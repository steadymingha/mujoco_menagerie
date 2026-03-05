import os
os.environ['DISPLAY'] = ':0'
import time
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
from core.control import JointController, GaitController
from graph import FootContactPlotter
from claude_sim import *
from core.fsm import QuadrupedContactFSM

SIMUL_TIME = 5.0  # Simulation duration in seconds

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MuJoCo Go2 Robot Simulation')
    parser.add_argument('--headless', action='store_true',
                        help='Run simulation without viewer (headless mode)', default=True)
    parser.add_argument('--record', type=str, default="output.mp4",
                        help='Record video to specified file (e.g., output.mp4). If not specified, no recording.')
    args = parser.parse_args()

    model_file = './unitree_go2/scene_mjx.xml'
    sim = Go2Sim(model_file)

    ff = FootForce(sim.model)
    fh = FootHeight()
    cm = ContactModel()

    gait = GaitController()
    ctrl = JointController(sim)




    # Disable real-time display, will save plot at the end
    max_len = int(SIMUL_TIME / sim.model.opt.timestep)
    plotter = FootContactPlotter(max_len=max_len, draw_interval=1, enable_display=False)
    foot_ids = get_foot_ids(sim.model)

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

    def run_step(viewer=None):
        nonlocal next_frame_time

        # --- [A] Algorithms ---
        fz = ff.get_foot_force(sim.data)
        pz = fh.get_foot_height(sim.data)
        p_foot_contact = cm.prob_contact(sim.data, pz, fz)

        # --- [A] verification ---
        ground_truth = get_ground_truth_contact(sim.model, sim.data, foot_ids)
        plotter.update(p_foot_contact, ground_truth, fz=fz, pz=pz)

        # --- [B] Video Recording ---
        if renderer and sim.sim_time >= next_frame_time:
            renderer.update_scene(sim.data)
            frames.append(renderer.render().copy())
            next_frame_time += frame_interval

        # --- [C] Controller ---

        gait.high_level_controller(sim, fh.get_foot_position(), cm.s_phi, cm.phi )

        ctrl.compute()
        sim.step(sim.ctrl0)

        # Viewer-specific: display debug text
        if viewer:
            debug_txt = f"T:{sim.sim_time:.1f}s | GT(FL):{ground_truth[0]} | Prob(FL):{p_foot_contact[0].item():.2f}"
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
