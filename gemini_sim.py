import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import enum
import contextlib
from mh.foot_mechanics import *
from mh.ground_contact import ContactModel
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
        
        # Unlock Control Limits for strong torque
        self.model.actuator_ctrlrange[:, 0] = -100.0 
        self.model.actuator_ctrlrange[:, 1] = 100.0  
        
        self.last_render_time = time.time()
        
        # 2. Reset & Manual Pose Init
        mujoco.mj_resetData(self.model, self.data)
        
        # Start Position: High enough
        self.data.qpos[0:3] = [0, 0, 0.5] 
        self.data.qpos[3:7] = [1, 0, 0, 0]

        # Init Pose for ALL legs: [Abd, Hip, Knee] -> [0.0, 0.8, -1.5]
        # This matches our target controller to minimize startup jump
        pose_init = [0.0, 0.8, -1.5]
        for i in range(4):
            base = 7 + i*3
            self.data.qpos[base:base+3] = pose_init

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.ctrl0 = np.zeros(self.model.nu)

    @contextlib.contextmanager
    def launch_viewer(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 2.0
            viewer.cam.lookat = [0, 0, 0.2]
            viewer.opt.geomgroup[3] = 1 
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

def get_ground_truth_contact(model, data, foot_geom_ids):
    ground_truth = np.zeros(4)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if floor_id == -1:
        for i in range(model.ngeom):
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                floor_id = i
                break
    
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        for leg_idx, foot_id in enumerate(foot_geom_ids):
            if (g1 == foot_id and g2 == floor_id) or (g2 == foot_id and g1 == floor_id):
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

    # Gains for PD Control
    kp = 100.0
    kd = 5.0

    # Leg Indices Mapping
    # Each leg has 3 motors: [Abd, Hip, Knee]
    legs_indices = [
        [0, 1, 2],   # FL
        [3, 4, 5],   # FR
        [6, 7, 8],   # RL
        [9, 10, 11]  # RR
    ]

    print("Simulation Loop Started...")
    
    with sim.launch_viewer() as viewer:
        while viewer.is_running():

            # --- [A] Algorithms ---
            fz = ff.get_foot_force(sim.data)
            pz = fh.get_foot_height(sim.data)
            p_foot_contact = cm.prob_contact(sim.data, pz, fz)
            
            ground_truth = get_ground_truth_contact(sim.model, sim.data, foot_ids)
            plotter.update(p_foot_contact, ground_truth)
            
            # --- [C] Controller: All Legs Stiff Standing ---
            sim.ctrl0[:] = 0.0 
            
            # Loop through all 4 legs
            target_abds = [0.174, -0.174, 0.174, -0.174] 
            target_thigh = [-0.1, -0.1, 0.8, 0.8]
            for leg_idx in range(4):
                indices = legs_indices[leg_idx] # [Abd, Hip, Knee] indices for this leg
                
                # 1. Abduction: PD Control -> Hold at 0.0
                i_abd = indices[0]
                curr_abd = sim.data.qpos[7 + i_abd]
                vel_abd = sim.data.qvel[6 + i_abd]
                sim.ctrl0[i_abd] = kp * (target_abds[leg_idx] - curr_abd) - kd * vel_abd
                
                # 2. Hip (Thigh): PD Control -> Hold at 0.8
                i_hip = indices[1]
                curr_hip = sim.data.qpos[7 + i_hip]
                vel_hip = sim.data.qvel[6 + i_hip]
                sim.ctrl0[i_hip] = kp * (target_thigh[leg_idx] - curr_hip) - kd * vel_hip
                
                # 3. Knee (Calf): Brute Force -> Extend (+30 Nm)
                i_knee = indices[2]
                sim.ctrl0[i_knee] = 30.0 
            
            # Display FL info for debugging
            debug_txt = f"GT(FL):{ground_truth[0]} | Prob(FL):{p_foot_contact[0].item():.2f}"
            
            sim.step(sim.ctrl0)
            sim.add_text(viewer, debug_txt)
            sim.sync(viewer)

if __name__ == "__main__":
    main()