"""
tau_d 계산 디버깅 - 왜 knee 값이 동일한지 확인
"""
import numpy as np
import mujoco

LEG_INDICES = {
    'FL': [6, 7, 8],
    'FR': [9, 10, 11],
    'RL': [12, 13, 14],
    'RR': [15, 16, 17]
}

xml_path = '../unitree_go2/scene_mjx.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
model.opt.timestep = 0.002
model.actuator_ctrlrange[:, 0] = -100.0
model.actuator_ctrlrange[:, 1] = 100.0

mujoco.mj_resetData(model, data)
data.qpos[0:3] = [0, 0, 0.5]
data.qpos[3:7] = [1, 0, 0, 0]
pose_init = [0.0, 0.8, -1.8]
for i in range(4):
    base = 7 + i * 3
    data.qpos[base:base + 3] = pose_init
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

M = np.zeros((model.nv, model.nv))
y_pre = np.zeros((18, 1))

gamma, beta = 0.828, 103.7

kp, kd = 100.0, 5.0
legs_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
target_abds = [0.0, 0.0, 0.0, 0.0]
target_hips = [0.8, 0.8, 0.8, 0.8]
target_knees = [-1.8, -1.8, -1.8, -1.8]
ctrl = np.zeros(12)

print("=" * 80)
print("tau_d 계산 디버깅")
print("=" * 80)

for step in range(2500):
    for leg_idx in range(4):
        indices = legs_indices[leg_idx]
        i_abd, i_hip, i_knee = indices
        ctrl[i_abd] = kp * (target_abds[leg_idx] - data.qpos[7 + i_abd]) - kd * data.qvel[6 + i_abd]
        ctrl[i_hip] = kp * (target_hips[leg_idx] - data.qpos[7 + i_hip]) - kd * data.qvel[6 + i_hip]
        ctrl[i_knee] = kp * (target_knees[leg_idx] - data.qpos[7 + i_knee]) - kd * data.qvel[6 + i_knee]
    
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)
    
    # Disturbance torque 계산
    mujoco.mj_fullM(model, M, data.qM)
    q_dot = data.qvel[:, np.newaxis]
    p = M @ q_dot
    qfrc_bias = data.qfrc_bias[:, np.newaxis]
    qfrc_actuator = data.qfrc_actuator[:, np.newaxis]
    
    dyn_terms = beta * p + qfrc_actuator - qfrc_bias
    y = (1 - gamma) * dyn_terms + gamma * y_pre
    tau_d = beta * p - y
    
    y_pre = y
    
    if step == 2000:
        print(f"\n--- Step {step} ---")
        
        print(f"\n=== 입력값 분석 ===")
        print(f"ctrl (12): {ctrl}")
        print(f"qfrc_actuator[6:18]: {qfrc_actuator[6:18].flatten()}")
        print(f"qfrc_bias[6:18]: {qfrc_bias[6:18].flatten()}")
        
        print(f"\n=== Knee 관련 값 비교 (indices 8, 11, 14, 17) ===")
        knee_indices = [8, 11, 14, 17]
        
        print(f"\n{'항목':<20} {'FL(8)':<12} {'FR(11)':<12} {'RL(14)':<12} {'RR(17)':<12}")
        print("-" * 68)
        print(f"{'qfrc_actuator':<20} {qfrc_actuator[8].item():<12.4f} {qfrc_actuator[11].item():<12.4f} {qfrc_actuator[14].item():<12.4f} {qfrc_actuator[17].item():<12.4f}")
        print(f"{'qfrc_bias':<20} {qfrc_bias[8].item():<12.4f} {qfrc_bias[11].item():<12.4f} {qfrc_bias[14].item():<12.4f} {qfrc_bias[17].item():<12.4f}")
        print(f"{'beta*p':<20} {(beta*p)[8].item():<12.4f} {(beta*p)[11].item():<12.4f} {(beta*p)[14].item():<12.4f} {(beta*p)[17].item():<12.4f}")
        print(f"{'dyn_terms':<20} {dyn_terms[8].item():<12.4f} {dyn_terms[11].item():<12.4f} {dyn_terms[14].item():<12.4f} {dyn_terms[17].item():<12.4f}")
        print(f"{'y':<20} {y[8].item():<12.4f} {y[11].item():<12.4f} {y[14].item():<12.4f} {y[17].item():<12.4f}")
        print(f"{'tau_d':<20} {tau_d[8].item():<12.4f} {tau_d[11].item():<12.4f} {tau_d[14].item():<12.4f} {tau_d[17].item():<12.4f}")
        
        print(f"\n=== 전체 tau_d ===")
        print(f"tau_d[6:9] (FL):   {tau_d[6:9].flatten()}")
        print(f"tau_d[9:12] (FR):  {tau_d[9:12].flatten()}")
        print(f"tau_d[12:15] (RL): {tau_d[12:15].flatten()}")
        print(f"tau_d[15:18] (RR): {tau_d[15:18].flatten()}")
        
        print(f"\n=== Joint 위치 (대칭성 확인) ===")
        print(f"FL knee (qpos[9]):  {data.qpos[9]:.4f}")
        print(f"FR knee (qpos[12]): {data.qpos[12]:.4f}")
        print(f"RL knee (qpos[15]): {data.qpos[15]:.4f}")
        print(f"RR knee (qpos[18]): {data.qpos[18]:.4f}")