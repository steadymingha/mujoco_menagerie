import os
os.environ['DISPLAY'] = ':0'
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import enum
import contextlib
import argparse
from datetime import datetime
import mediapy as media
from core.foot_mechanics_claude import *
from core.ground_contact import ContactModel
from graph import FootContactPlotter

# --- Simulator Wrapper ---
class Go2Sim:
    def __init__(self, model_path: str, dt: float = 0.001):
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

        # Init Pose for ALL legs: [Abd, Hip, Knee] -> [0.0, 0.8, -1.8]
        # Knee is set to -1.8 to stay within joint limit [-2.72, -0.84]
        pose_init = [0.0, 0.8, -1.8]
        for i in range(4):
            base = 7 + i*3
            self.data.qpos[base:base+3] = pose_init

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.ctrl0 = np.zeros(self.model.nu)
        
        # Initialize simulation time tracking for automated lift
        self.sim_time = 0.0

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

    def apply_automated_lift(self, data):
        """
        Applies automated lifting force to robot base periodically.
        Cycle: 2s ground -> 1s lift, repeated.
        """
        cycle_period = 2.0  # Total cycle: 2s ground + 1s lift
        ground_duration = 1.0

        # Time within current cycle
        t_in_cycle = self.sim_time % cycle_period

        # Lift during last 1s of each cycle (i.e., t_in_cycle >= 2.0)
        if t_in_cycle >= ground_duration:
            lift_force = 150.0  # Newtons
            data.qfrc_applied[2] = lift_force
        else:
            data.qfrc_applied[:] = 0.0

    def step(self, ctrl_input: np.ndarray = None):
        if ctrl_input is not None:
            self.data.ctrl[:] = ctrl_input
        else:
            self.data.ctrl[:] = self.ctrl0 
        
        # Apply automated lift before stepping
        self.apply_automated_lift(self.data)
        
        mujoco.mj_step(self.model, self.data)
        
        # Update simulation time
        self.sim_time += self.model.opt.timestep
    
    def sync(self, viewer=None):
        """
        Synchronizes with viewer and maintains real-time speed.
        If viewer is None (headless mode), only maintains timing.
        """
        if viewer is not None:
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

SIMUL_TIME = 5.0  # Simulation duration in seconds

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MuJoCo Go2 Robot Simulation')
    parser.add_argument('--headless', action='store_true',
                        help='Run simulation without viewer (headless mode)', default=False)
    parser.add_argument('--record', type=str, default="output.mp4",
                        help='Record video to specified file (e.g., output.mp4). If not specified, no recording.')
    args = parser.parse_args()

    model_file = './unitree_go2/scene_mjx.xml'
    sim = Go2Sim(model_file)

    ff = FootForce(sim.model)
    fh = FootHeight()
    cm = ContactModel()

    # Disable real-time display, will save plot at the end
    max_len = int(SIMUL_TIME / sim.model.opt.timestep)
    plotter = FootContactPlotter(max_len=max_len, draw_interval=1, enable_display=False)
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
    print(f"Running for {SIMUL_TIME} seconds...")
    if args.headless:
        print("Running in HEADLESS mode (no viewer)")

    # Video recording setup
    frames = []
    renderer = None
    record_fps = 30
    frame_interval = 1.0 / record_fps
    next_frame_time = 0.0
    if not args.headless and args.record:
        print(f"Recording video to: {args.record}")
        renderer = mujoco.Renderer(sim.model, height=480, width=640)

    # ============================================
    # Target positions for controller
    # All joints use PD control to stay within joint limits
    # ============================================
    # Joint limits from XML:
    #   - Abd (hip_joint): [-1.05, 1.05]
    #   - Hip (thigh_joint): [-1.57, 3.49]
    #   - Knee (calf_joint): [-2.72, -0.84]
    # ============================================
    target_abds = [0.0, 0.0, 0.0, 0.0]
    target_hips = [0.8, 0.8, 0.8, 0.8]
    target_knees = [-1.8, -1.8, -1.8, -1.8]  # Within limit [-2.72, -0.84]

    # Joint limit margins (to avoid hitting limits)
    KNEE_LIMIT_UPPER = -0.85  # Slightly inside the limit -0.84
    KNEE_LIMIT_LOWER = -2.70  # Slightly inside the limit -2.72

    def run_step(viewer=None):
        """Common simulation step logic for both headless and viewer modes."""
        nonlocal next_frame_time

        # --- [A] Algorithms ---
        fz = ff.get_foot_force(sim.data)
        pz = fh.get_foot_height(sim.data)
        p_foot_contact = cm.prob_contact(sim.data, pz, fz)
        ground_truth = get_ground_truth_contact(sim.model, sim.data, foot_ids)
        plotter.update(p_foot_contact, ground_truth, fz=fz, pz=pz)

        # --- [B] Video Recording ---
        if renderer and sim.sim_time >= next_frame_time:
            renderer.update_scene(sim.data)
            frames.append(renderer.render().copy())
            next_frame_time += frame_interval

        # --- [C] Controller: All Legs PD Control ---
        sim.ctrl0[:] = 0.0
        for leg_idx in range(4):
            indices = legs_indices[leg_idx]  # [Abd, Hip, Knee] indices for this leg

            # 1. Abduction: PD Control
            i_abd = indices[0]
            curr_abd = sim.data.qpos[7 + i_abd]
            vel_abd = sim.data.qvel[6 + i_abd]
            sim.ctrl0[i_abd] = kp * (target_abds[leg_idx] - curr_abd) - kd * vel_abd

            # 2. Hip (Thigh): PD Control
            i_hip = indices[1]
            curr_hip = sim.data.qpos[7 + i_hip]
            vel_hip = sim.data.qvel[6 + i_hip]
            sim.ctrl0[i_hip] = kp * (target_hips[leg_idx] - curr_hip) - kd * vel_hip

            # 3. Knee (Calf): PD Control with joint limit protection
            i_knee = indices[2]
            curr_knee = sim.data.qpos[7 + i_knee]
            vel_knee = sim.data.qvel[6 + i_knee]
            
            # Clamp target to stay within joint limits
            target_knee_clamped = np.clip(target_knees[leg_idx], KNEE_LIMIT_LOWER, KNEE_LIMIT_UPPER)
            
            sim.ctrl0[i_knee] = kp * (target_knee_clamped - curr_knee) - kd * vel_knee

        sim.step(sim.ctrl0)

        # Viewer-specific: display debug text
        if viewer:
            lift_status = "LIFT" if 2.0 <= sim.sim_time <= 3.0 else "GROUND"
            debug_txt = f"T:{sim.sim_time:.1f}s | {lift_status} | GT(FL):{ground_truth[0]} | Prob(FL):{p_foot_contact[0].item():.2f}"
            sim.add_text(viewer, debug_txt)

        sim.sync(viewer)

    # Run simulation with or without viewer
    if args.headless:
        while sim.sim_time < SIMUL_TIME:
            run_step()
    else:
        with sim.launch_viewer() as viewer:
            while viewer.is_running() and sim.sim_time < SIMUL_TIME:
                run_step(viewer)

    # Save plot after simulation ends
    print(f"\nSimulation completed at t={sim.sim_time:.2f}s")

    # Save video if recording was enabled
    if args.record and frames:
        media.write_video(args.record, frames, fps=record_fps)
        print(f"Video saved: {args.record} ({len(frames)} frames)")
        renderer.close()

    # Generate timestamp filename (format: YYMMDD_HHMM.png)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    filename = f"results/{timestamp}.png"

    plotter.save(filename)
    plotter.close()
    print("Done!")

if __name__ == "__main__":
    main()
