"""
Joint limit 내에서 동작하는 적절한 자세로 테스트
"""
import numpy as np
import mujoco

LEG_INDICES = {
    'FL': [6, 7, 8],
    'FR': [9, 10, 11],
    'RL': [12, 13, 14],
    'RR': [15, 16, 17]
}

def torque_to_force(model, data, tau_d):
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
    return np.array(foot_forces)

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
    return contact_forces

def main():
    xml_path = '../unitree_go2/scene_mjx.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    model.actuator_ctrlrange[:, 0] = -100.0 
    model.actuator_ctrlrange[:, 1] = 100.0  
    
    # Joint limits 확인
    print("=== Knee Joint Limits ===")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if 'calf' in str(name):
            print(f"{name}: range={model.jnt_range[i]}")
    
    # 적절한 초기 자세: joint limit 내에서
    # knee range: -2.7227 ~ -0.8378
    # -1.5는 범위 내이지만, ctrl=30으로 밀면 limit에 도달
    
    # 해결책: PD 제어로 knee도 위치 제어
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 0.5]
    data.qpos[3:7] = [1, 0, 0, 0]
    
    # 초기 자세 (knee를 limit 중간으로)
    pose_init = [0.0, 0.8, -1.8]  # knee = -1.8 (limit 중간)
    for i in range(4):
        base = 7 + i*3
        data.qpos[base:base+3] = pose_init
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    
    M = np.zeros((model.nv, model.nv))
    y_pre = np.zeros((18, 1))
    
    gamma, beta = 0.828, 103.7
    S_T = np.block([[np.zeros([6,12])], [np.eye(12)]])
    
    kp, kd = 100.0, 5.0
    legs_indices = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
    
    # 타겟: joint limit 내에서 유지
    target_abds = [0.0, 0.0, 0.0, 0.0]
    target_hip = [0.8, 0.8, 0.8, 0.8]  # 모든 다리 동일
    target_knee = [-1.8, -1.8, -1.8, -1.8]  # limit 중간
    
    ctrl = np.zeros(12)
    
    print("\n" + "=" * 80)
    print("PD 제어로 Joint Limit 내 유지 테스트")
    print("=" * 80)
    
    for step in range(2500):
        for leg_idx in range(4):
            indices = legs_indices[leg_idx]
            i_abd, i_hip, i_knee = indices
            
            # 모든 관절 PD 제어
            ctrl[i_abd] = kp * (target_abds[leg_idx] - data.qpos[7 + i_abd]) - kd * data.qvel[6 + i_abd]
            ctrl[i_hip] = kp * (target_hip[leg_idx] - data.qpos[7 + i_hip]) - kd * data.qvel[6 + i_hip]
            ctrl[i_knee] = kp * (target_knee[leg_idx] - data.qpos[7 + i_knee]) - kd * data.qvel[6 + i_knee]
        
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        
        # Observer (qfrc_actuator 사용)
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
            
            print(f"\n=== Joint 상태 (limit 확인) ===")
            print(f"FL: abd={data.qpos[7]:.3f}, hip={data.qpos[8]:.3f}, knee={data.qpos[9]:.3f}")
            print(f"Knee limit: [-2.72, -0.84]")
            
            # Limit 위반 확인
            for i, name in enumerate(['FL', 'FR', 'RL', 'RR']):
                knee_pos = data.qpos[9 + i*3]
                if knee_pos > -0.8378:
                    print(f"  ⚠️ {name} knee at limit!")
            
            print(f"\n=== Constraint 수 ===")
            print(f"nefc: {data.nefc}, ncon: {data.ncon}")
            
            # Equality constraint (joint limit) 개수
            eq_count = sum(1 for i in range(data.nefc) if data.efc_type[i] == 3)
            limit_forces = [data.efc_force[i] for i in range(data.nefc) if data.efc_type[i] == 3]
            print(f"Equality constraints: {eq_count}")
            if limit_forces:
                print(f"Equality forces: {limit_forces}")
            
            print(f"\n=== 토크 ===")
            print(f"ctrl[0:3] (FL): {ctrl[0:3]}")
            print(f"qfrc_actuator[6:9] (FL): {data.qfrc_actuator[6:9]}")
            
            print(f"\n=== tau_d (FL) ===")
            print(f"tau_d[6:9]: {tau_d[6:9].flatten()}")
            
            # MuJoCo 실제 힘 -> 토크
            mujoco_forces = get_mujoco_contact_forces(model, data)
            for i, leg_name in enumerate(['FL']):
                site_name = f"{leg_name}_foot"
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                idx = LEG_INDICES[leg_name]
                J_leg = jacp[:, idx]
                f_actual = np.array([0, 0, mujoco_forces[i]])
                tau_from_mujoco = J_leg.T @ f_actual
                print(f"MuJoCo 힘->토크 (FL): {tau_from_mujoco}")
            
            forces = torque_to_force(model, data, tau_d)
            
            print(f"\n=== 힘 비교 ===")
            print(f"{'방법':<25} {'FL':<10} {'FR':<10} {'RL':<10} {'RR':<10} {'합계':<10}")
            print("-" * 75)
            print(f"{'Observer':<25} {forces[0]:<10.1f} {forces[1]:<10.1f} {forces[2]:<10.1f} {forces[3]:<10.1f} {sum(forces):<10.1f}")
            print(f"{'MuJoCo 실제':<25} {mujoco_forces[0]:<10.1f} {mujoco_forces[1]:<10.1f} {mujoco_forces[2]:<10.1f} {mujoco_forces[3]:<10.1f} {sum(mujoco_forces):<10.1f}")

if __name__ == "__main__":
    main()