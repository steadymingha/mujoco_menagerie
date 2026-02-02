"""
Contact probability 디버깅
z1, z2 값 및 Kalman filter 동작 확인
"""
import numpy as np
import mujoco
from scipy.special import erf
import math

LEG_INDICES = {
    'FL': [6, 7, 8],
    'FR': [9, 10, 11],
    'RL': [12, 13, 14],
    'RR': [15, 16, 17]
}

def prob_contact_given_foot_height(pz):
    mu_zg = 0
    sigma_zg = math.sqrt(0.1)  # ≈ 0.316
    p_c_pz = 0.5 * (1 + erf((mu_zg - pz) / (sigma_zg * math.sqrt(2))))
    return p_c_pz

def prob_contact_given_contact_force(fz):
    mu_fc = 40
    sigma_fc = math.sqrt(25)  # = 5
    p_c_fz = 0.5 * (1 + erf((fz - mu_fc) / (sigma_fc * math.sqrt(2))))
    return p_c_fz

def get_foot_height_simple(data):
    """간단한 foot height 계산 (site 사용)"""
    heights = []
    for leg in ['FL', 'FR', 'RL', 'RR']:
        site_name = f"{leg}_foot"
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id != -1:
            heights.append(data.site_xpos[site_id][2])
        else:
            heights.append(0.0)
    return np.array(heights)[:, np.newaxis]

def get_mujoco_contact_forces(model, data):
    contact_forces = np.zeros(4)
    leg_names = ["FL", "FR", "RL", "RR"]
    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        body_name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world"
        body_name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world"
        c_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, c_force)
        for leg_idx, leg in enumerate(leg_names):
            if f"{leg}_calf" in body_name1 or f"{leg}_calf" in body_name2:
                contact_forces[leg_idx] += abs(c_force[0])
    return contact_forces[:, np.newaxis]

# Observer로 계산한 force
def get_observer_force(model, data, M, y_pre):
    gamma, beta = 0.828, 103.7
    
    mujoco.mj_fullM(model, M, data.qM)
    q_dot = data.qvel[:, np.newaxis]
    p = M @ q_dot
    qfrc_bias = data.qfrc_bias[:, np.newaxis]
    qfrc_actuator = data.qfrc_actuator[:, np.newaxis]
    
    dyn_terms = beta * p + qfrc_actuator - qfrc_bias
    y = (1 - gamma) * dyn_terms + gamma * y_pre
    tau_d = beta * p - y
    
    # Torque to force
    foot_forces = []
    for leg_name in ['FL', 'FR', 'RL', 'RR']:
        site_name = f"{leg_name}_foot"
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        idx = LEG_INDICES[leg_name]
        J_leg = jacp[:, idx]
        tau_d_leg = tau_d[idx].flatten()
        f_foot = np.linalg.pinv(J_leg.T) @ tau_d_leg
        foot_forces.append(float(f_foot[2]))
    
    return np.array(foot_forces)[:, np.newaxis], y

# Setup
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

kp, kd = 100.0, 5.0
legs_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
target_abds = [0.0, 0.0, 0.0, 0.0]
target_hips = [0.8, 0.8, 0.8, 0.8]
target_knees = [-1.8, -1.8, -1.8, -1.8]
ctrl = np.zeros(12)

print("=" * 80)
print("Contact Probability 디버깅")
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
    
    # Observer update
    fz_observer, y_pre = get_observer_force(model, data, M, y_pre)
    
    if step == 2000:
        print(f"\n--- Step {step} ---")
        
        # Foot height (from site)
        pz = get_foot_height_simple(data)
        
        # Foot force (from observer)
        fz = fz_observer
        
        # MuJoCo ground truth force
        fz_mujoco = get_mujoco_contact_forces(model, data)
        
        print(f"\n=== Raw Values ===")
        print(f"Foot Height (pz): {pz.flatten()}")
        print(f"Observer Force (fz): {fz.flatten()}")
        print(f"MuJoCo Force: {fz_mujoco.flatten()}")
        
        print(f"\n=== Probability Calculations ===")
        
        # z1: prob from height
        z1 = prob_contact_given_foot_height(pz)
        print(f"\nz1 (from height):")
        print(f"  Formula: 0.5 * (1 + erf((0 - pz) / (0.316 * sqrt(2))))")
        print(f"  pz = {pz.flatten()}")
        print(f"  z1 = {z1.flatten()}")
        
        # z2: prob from force
        z2 = prob_contact_given_contact_force(fz)
        print(f"\nz2 (from force):")
        print(f"  Formula: 0.5 * (1 + erf((fz - 40) / (5 * sqrt(2))))")
        print(f"  fz = {fz.flatten()}")
        print(f"  z2 = {z2.flatten()}")
        
        print(f"\n=== Analysis ===")
        print(f"Height model expects: pz < 0 for high contact prob")
        print(f"  Current pz ≈ 0.01~0.02m → (0 - 0.015) / 0.447 ≈ -0.034")
        print(f"  erf(-0.034) ≈ -0.038 → prob ≈ 0.48")
        
        print(f"\nForce model expects: fz > 40N for high contact prob")
        print(f"  Current fz ≈ 29~46N")
        print(f"  For fz=29: (29-40)/(5*1.414) ≈ -1.56 → erf ≈ -0.97 → prob ≈ 0.015")
        print(f"  For fz=46: (46-40)/(5*1.414) ≈ 0.85 → erf ≈ 0.77 → prob ≈ 0.89")
        
        print(f"\n=== Parameter Suggestions ===")
        print(f"Current foot height range: {pz.min():.4f} ~ {pz.max():.4f}m")
        print(f"Current force range: {fz.min():.1f} ~ {fz.max():.1f}N")
        
        # 더 적절한 파라미터 제안
        print(f"\nSuggested height params:")
        print(f"  mu_zg = 0.02 (typical standing height)")
        print(f"  sigma_zg = 0.01 (tighter threshold)")
        
        print(f"\nSuggested force params:")
        print(f"  mu_fc = 30 (lower threshold for lighter robot)")
        print(f"  sigma_fc = 10 (wider acceptance)")

