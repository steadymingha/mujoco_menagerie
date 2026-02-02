import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import enum
import contextlib
from core.foot_mechanics import *
from core.ground_contact import ContactModel
from graph import FootContactPlotter

# --- Simulator Wrapper ---
class Go2Sim:
    def __init__(self, model_path: str, dt: float = 0.002):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # 1. Load Model & Data
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        
        self.last_render_time = time.time()
        
        # 2. Reset
        mujoco.mj_resetData(self.model, self.data)
        
        # --- [INIT] Force High Drop Position ---
        # Start high enough (0.8m) to see the leg extend before hitting ground
        self.data.qpos[0:3] = [0, 0, 0.8] 
        self.data.qpos[3:7] = [1, 0, 0, 0] # Quaternion identity

        # Manual Pose Init: Try to unfold slightly to avoid self-collision at start
        # Hip: 0.5, Knee: -1.0
        pose_init = [0.0, 0.5, -1.0]
        for i in range(4):
            base = 7 + i*3
            self.data.qpos[base:base+3] = pose_init

        mujoco.mj_forward(self.model, self.data)
        
        self.ctrl0 = np.zeros(self.model.nu)

    @contextlib.contextmanager
    def launch_viewer(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 1.5
            viewer.cam.lookat = [0, 0, 0.3]
            viewer.opt.geomgroup[3] = 1 # Show collision geoms
            yield viewer 

    def add_text(self, viewer, contents):
        data = self.data
        if viewer.user_scn:
                viewer.user_scn.ngeom = 0 
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    text_pos = np.array([data.qpos[0], data.qpos[1], data.qpos[2] + 0.5])
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LABEL,
                        np.zeros(3),
                        text_pos,
                        np.zeros(9),
                        np.array([1, 1, 1, 1]) 
                    )
                    geom.label = contents
                    viewer.user_scn.ngeom += 1

    def step(self, ctrl_input: np.ndarray = None):
        if ctrl_input is not None:
            self.data.ctrl[:] = ctrl_input
        else:
            self.data.ctrl[:] = self.ctrl0 
        mujoco.mj_step(self.model, self.data)
    
    def sync(self, viewer):
        viewer.sync()
        time_until_next_step = self.model.opt.timestep - (time.time() - self.last_render_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        self.last_render_time = time.time()

def get_foot_ids(model):
    foot_names = ["FL", "FR", "RL", "RR"] 
    foot_geom_ids = []
    for name in foot_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1: foot_geom_ids.append(gid)
    return foot_geom_ids

# --- [MODIFIED] Fix: Only count contact with FLOOR ---
def get_ground_truth_contact(model, data, foot_geom_ids):
    ground_truth = np.zeros(4)
    
    # Check for floor geometry (Usually ID 0, or named 'floor')
    # If your XML doesn't name the floor, usually the plane is the first geom (id 0)
    # Let's try to find it by name, if not, assume it's geom_id 0 if type is plane
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    
    # If not named 'floor', look for the first plane
    if floor_id == -1:
        for i in range(model.ngeom):
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                floor_id = i
                break
    
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        
        # Check contact: Foot <-> Floor
        # Ignore Foot <-> Robot Body (Self collision)
        
        for leg_idx, foot_id in enumerate(foot_geom_ids):
            # Case 1: g1 is foot, g2 is floor
            if g1 == foot_id and g2 == floor_id:
                ground_truth[leg_idx] = 1.0
            # Case 2: g2 is foot, g1 is floor
            elif g2 == foot_id and g1 == floor_id:
                ground_truth[leg_idx] = 1.0
            
    return ground_truth

SIMUL_TIME = 100 

def main():
    model_file = './unitree_go2/scene_mjx.xml'
    sim = Go2Sim(model_file)
    
    ff = FootForce(sim.model)
    fh = FootHeight()
    cm = ContactModel()
    
    plotter = FootContactPlotter(max_len=100, draw_interval=5)
    foot_ids = get_foot_ids(sim.model)

    print("Simulation Loop Started...")
    
    with sim.launch_viewer() as viewer:
        while viewer.is_running():

            # --- [A] Algorithms ---
            fz = ff.get_foot_force(sim.data)
            pz = fh.get_foot_height(sim.data)
            p_foot_contact = cm.prob_contact(sim.data, pz, fz)
            ground_truth = get_ground_truth_contact(sim.model, sim.data, foot_ids)
            
            plotter.update(p_foot_contact, ground_truth)
            
            # --- [B] Visualization ---
            for i, gid in enumerate(foot_ids):
                sim.model.geom_matid[gid] = -1
                if p_foot_contact[i] > 0.5:
                    sim.model.geom_rgba[gid] = [1.0, 0.0, 0.0, 1.0] # Red
                else:
                    sim.model.geom_rgba[gid] = [0.0, 0.0, 1.0, 1.0] # Blue

            # --- [C] Controller: Brute Force Extension (FIXED) ---
            sim.ctrl0[:] = 0.0 
            
            # 1. Extend FL Knee: Tried +20, failed. Trying -25.0
            sim.ctrl0[2] = -25.0 
            
            # 2. Extend FL Hip (Thigh) to help: Index 1
            # Usually + extends thigh downwards
            sim.ctrl0[1] = 10.0 
            
            # Debug: Check actual angle
            knee_angle = sim.data.qpos[7+2] # FL Knee
            
            sim.step(sim.ctrl0)
            
            txt = f"GT:{ground_truth[0]} | Ang:{knee_angle:.2f}"
            sim.add_text(viewer, txt)
            sim.sync(viewer)

if __name__ == "__main__":
    main()