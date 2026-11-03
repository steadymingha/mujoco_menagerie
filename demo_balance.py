"""QP balance controller push-recovery test, with video recording.

Checks that the Cheetah3 balance controller
(GaitController.balance_controller -> BalanceQPSolver) keeps the robot upright
while external pushes are applied. This script produced the blog video
results/qp_balance_test.mp4.

Flow
    1) Until PD_SETTLE_TIME: JointController (joint-space PD) to land and settle
    2) After that: GRF from the QP, mapped to torques and written straight into
       sim.ctrl0 (unlike main.py, where PD overwrites the QP torques)
    3) Pushes from PUSH_SCHEDULE applied to the base body via xfrc_applied
    4) Every frame rendered and saved as mp4 to RECORD_PATH

No gait here: all four feet are held in stance (np.full(4, 0.5)) and only
balance is tested.

Viewer overlay
    green sphere  : pcom_d  (predictive support polygon target CoM)
    red sphere    : pcom    (actual CoM)
    orange arrow  : push direction

Frame convention (Go2 XML): +x = toward the head, +y = to the left
"""
import os
os.environ['DISPLAY'] = ':0'
import time
import numpy as np
import mujoco
import mediapy as media
from claude_sim import Go2Sim, get_foot_ids
from core.foot_mechanics_claude import FootForce, FootHeight
from core.ground_contact import ContactModel
from core.control import GaitController, JointController

SIMUL_TIME    = 15.0
PD_SETTLE_TIME = 1.5   # seconds of PD control before switching to QP
SLOW_FACTOR   = 1.0    # viewer playback speed (1.0 = real-time, 3.0 = 3x slower)
RECORD_PATH   = "results/qp_balance_test.mp4"
RECORD_FPS    = 30

# Push schedule: (t_start, t_end, force_xyz)
# Go2 XML: +x = 머리 방향, +y = 왼쪽
# 힘 방향 = "미는 방향" (힘이 로봇에 작용하는 방향)
# PUSH_SCHEDULE = [
#     (3.0,  3.5,  np.array([-15,   0,  0])),  # 머리→꼬리 (앞에서 뒤로 밀기)
#     (5.0,  5.5,  np.array([ 15,   0,  0])),  # 꼬리→머리 (뒤에서 앞으로 밀기)
#     (7.0,  7.5,  np.array([  0, -11,  0])),  # 왼쪽→오른쪽 (왼쪽에서 밀기)
#     (9.0,  9.5,  np.array([  0,  11,  0]))  # 오른쪽→왼쪽 (오른쪽에서 밀기)
#     # (11.0, 11.5, np.array([ 5,  5,  0])),  # 대각선 (뒤+오른쪽에서 밀기)
# ]
# PUSH_SCHEDULE = [
#     (3.0,  3.5,  np.array([ -15,   0,  0])),  # 꼬리→머리
#     (5.0,  5.5,   np.array([18,   0,  0])),  # 머리→꼬리
# ]
PUSH_SCHEDULE = [
    (3.0,  3.5,  np.array([  0, -20,  0])),  # 왼쪽→오른쪽
    (6.0,  6.5,  np.array([  0,  12,  0])),  # 오른쪽→왼쪽
    (7.0,  7.5,  np.array([ 15,   0,  0]))  # 꼬리→머리
    # (9.0,  9.5,  np.array([-8,   0,  0])),  # 머리→꼬리
]
# for rechord
# PUSH_SCHEDULE = [
#     (5.0,  5.5,  np.array([  0, -15,  0])),  # 왼쪽→오른쪽
#     (8.0,  8.5,  np.array([  0,  12,  0])),  # 오른쪽→왼쪽
#     (11.0, 11.5,  np.array([ 15,   0,  0]))  # 꼬리→머리
#     # (9.0,  9.5,  np.array([-8,   0,  0])),  # 머리→꼬리
# ]

# PUSH_SCHEDULE = [

#     (3.0,  3.5,  np.array([ -20,   0,  0])),  # 꼬리→머리
#     (5.0,  5.5,  np.array([-20,   0,  0])),  # 머리→꼬리
# ]
PUSH_LABELS = {
    0: "head→tail (-X)",
    1: "tail→head (+X)",
    2: "left→right (-Y)",
    3: "right→left (+Y)",
    4: "diagonal (+X+Y)",
}


def get_push(t):
    for i, (t0, t1, f) in enumerate(PUSH_SCHEDULE):
        if t0 <= t < t1:
            return f, PUSH_LABELS[i]
    return np.zeros(3), "none"



def main():
    sim = Go2Sim('./unitree_go2/scene_mjx.xml')
    ff  = FootForce(sim.model)
    fh  = FootHeight()
    cm  = ContactModel()
    gait = GaitController()
    ctrl = JointController(sim)  # PD controller for settling phase

    # Base body for xfrc_applied
    base_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    if base_id == -1:
        base_id = 1  # fallback: first non-world body

    xfrc = np.zeros(6)  # reused buffer: [torque(3), force(3)]

    frames = []
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    frame_interval = 1.0 / RECORD_FPS
    next_frame_time = 0.0

    def run_step(viewer=None):
        nonlocal next_frame_time
        # --- Sensing ---
        fz = ff.get_foot_force(sim.data)       # also populates ff.Jacobians
        pz = fh.get_foot_height(sim.data)      # also populates fh.foot_world
        p_foot_contact = cm.prob_contact(sim.data, pz, fz)

        # --- Controller: PD settle → QP balance ---
        if sim.sim_time < PD_SETTLE_TIME:
            pcom_d = sim.data.subtree_com[0].copy()
            pcom   = pcom_d
            ctrl.compute()
        else:
            pcom_d, pcom, tq_d = gait.balance_controller(
                sim,
                fh.get_foot_position(),
                p_foot_contact.ravel(),
                np.full(4, 0.5),
                ff.Jacobians,
                np.zeros(3),
                0.0,
            )
            sim.ctrl0[:] = tq_d.flatten()

        # --- Push Disturbance ---
        push_force, push_label = get_push(sim.sim_time)
        sim.data.xfrc_applied[base_id, :3] = push_force

        # # === GRF 디버그 ===
        # if np.linalg.norm(push_force) > 0 and sim.sim_time % 0.1 < sim.model.opt.timestep:
        #     F = gait.qp.F_prev  # 마지막 QP 결과
        #     print(f"\n=== GRF t={sim.sim_time:.2f} push={push_force} ===")
        #     names = ['FL', 'FR', 'RL', 'RR']
        #     F_total = np.zeros(3)
        #     for i in range(4):
        #         fi = F[3*i:3*i+3]
        #         F_total += fi
        #         print(f"  {names[i]}: Fx={fi[0]:7.2f} Fy={fi[1]:7.2f} Fz={fi[2]:7.2f}")
        #     print(f"  합계: Fx={F_total[0]:7.2f} Fy={F_total[1]:7.2f} Fz={F_total[2]:7.2f}")
        #     print(f"  필요: Fx={-push_force[0]:7.2f} Fy={-push_force[1]:7.2f} (push 반대)")
        
        # if 3.0 < sim.sim_time < 3.02:
        #     F = gait.qp.F_prev
        #     for i in range(4):
        #         F_leg = F[3*i:3*i+3]
        #         tau = -ff.Jacobians[i].T @ F_leg
        #         F_check = np.linalg.pinv(ff.Jacobians[i].T) @ tau
        #         names = ['FL','FR','RL','RR']
        #         print(f"{names[i]}: GRF={F_leg} → tau={tau} → F_recovered={F_check}")

        sim.step(sim.ctrl0)

        # --- Video Frame Capture ---
        if sim.sim_time >= next_frame_time:
            renderer.update_scene(sim.data)
            frames.append(renderer.render().copy())
            next_frame_time += frame_interval

        # --- Viewer Overlay ---
        if viewer:
            sim.clear_overlay(viewer)

            # Green sphere: pcom_d (predictive support polygon target)
            sim.add_sphere(viewer, pcom_d, radius=0.05, rgba=[0.0, 1.0, 0.0, 0.9])
            # Red sphere: actual CoM
            sim.add_sphere(viewer, pcom, radius=0.04, rgba=[1.0, 0.2, 0.2, 0.9])

            # if np.linalg.norm(push_force) > 0:
            #     push_dir = push_force / np.linalg.norm(push_force)
            #     body_pos = sim.data.xpos[base_id].copy()
            #     indicator_pos = body_pos - push_dir * 0.35
            #     sim.add_sphere(viewer, indicator_pos, radius=0.04, rgba=[1.0, 0.6, 0.0, 0.9])
            if np.linalg.norm(push_force) > 0:
                push_dir = push_force / np.linalg.norm(push_force)
                arrow_from = pcom - push_dir * 0.4
                arrow_from[2] = pcom[2]
                arrow_to = arrow_from + push_dir * 0.3
                
                viewer.user_scn.ngeom += 1
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.015,
                    arrow_from.astype(np.float64),
                    arrow_to.astype(np.float64),
                )
                viewer.user_scn.geoms[viewer.user_scn.ngeom - 1].rgba = np.array([1.0, 0.6, 0.0, 0.9], dtype=np.float32)

            # debug_txt = (
            #     f"T:{sim.sim_time:.2f}s | push:{push_label} {np.linalg.norm(push_force):.0f}N | "
            #     f"pcom_d:({pcom_d[0]:.2f},{pcom_d[1]:.2f},{pcom_d[2]:.2f}) | "
            #     f"pcom:({pcom[0]:.2f},{pcom[1]:.2f},{pcom[2]:.2f})"
            # )
            # sim.add_text(viewer, debug_txt)

        sim.sync(viewer)
        if viewer and SLOW_FACTOR > 1.0:
            time.sleep((SLOW_FACTOR - 1.0) * sim.model.opt.timestep)

    print("=== QP Balance Test ===")
    print("Green sphere : pcom_d  (predictive support polygon target)")
    print("Red sphere   : pcom    (actual CoM)")
    print("Orange arrow : push direction")
    print()
    for i, (t0, t1, f) in enumerate(PUSH_SCHEDULE):
        print(f"  t={t0:.1f}-{t1:.1f}s : {PUSH_LABELS[i]}  {np.linalg.norm(f):.0f}N")
    print()

    with sim.launch_viewer() as viewer:
        while viewer.is_running() and sim.sim_time < SIMUL_TIME:
            run_step(viewer)

    renderer.close()
    if frames:
        os.makedirs(os.path.dirname(RECORD_PATH), exist_ok=True)
        media.write_video(RECORD_PATH, frames, fps=RECORD_FPS)
        print(f"Video saved: {RECORD_PATH} ({len(frames)} frames)")

    print(f"Test complete at t={sim.sim_time:.2f}s")


if __name__ == '__main__':
    main()