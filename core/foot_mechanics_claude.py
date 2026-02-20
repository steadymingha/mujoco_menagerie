import numpy as np
import mujoco
from typing import Tuple


# 상수 정의
X_BASE_OFFSET = 0.1934
Y_BASE_OFFSET = 0.0465
Y_OFFSET = 0.0955
L_THIGH = 0.213
L_CALF = 0.213

# 다리 위치 리스트
LEGS = ['front_left', 'front_right', 'hind_left', 'hind_right']
# 관절 유형 리스트
JOINTS = ['abduction', 'hip', 'knee']

class FootHeight:
    def __init__(self):
        pass
    def cal_foot_body_position(self, data):
        # 결과 딕셔너리
        joint_data = {}

        for joint in JOINTS:
            # 해당 관절의 모든 다리 데이터를 리스트로 수집
            joint_angles = [
                data.sensor(f'{joint}_{leg}_pos').data[0]
                for leg in LEGS
            ]
            # np.array로 변환하여 딕셔너리에 저장
            joint_data[f'theta_{joint}'] = np.array(joint_angles)

        # 결과 추출 (원래 변수명과 일치하게)
        theta_abduction = joint_data['theta_abduction']
        theta_hip = joint_data['theta_hip']
        theta_knee = joint_data['theta_knee']

        # 삼각함수 값 미리 계산 
        s1 = np.sin(theta_abduction)
        c1 = np.cos(theta_abduction)
        s2 = np.sin(theta_hip)
        c2 = np.cos(theta_hip)
        s23 = np.sin(theta_hip + theta_knee)
        c23 = np.cos(theta_hip + theta_knee)

        # 다리별 오프셋 (FL, FR, RL, RR 순서)
        X_BASE_OFFSETS = np.array([+0.1934, +0.1934, -0.1934, -0.1934])
        Y_BASE_OFFSETS = np.array([+0.0465, -0.0465, +0.0465, -0.0465])
        Y_OFFSETS = np.array([+0.0955, -0.0955, +0.0955, -0.0955])
        FOOT_OFFSET_X = -0.002

        # 발 위치 계산 (hip frame 기준)
        leg_z = -L_THIGH * c2 - L_CALF * c23
        leg_x = -L_THIGH * s2 - L_CALF * s23 + FOOT_OFFSET_X

        foot_x_hip = leg_x
        foot_y_hip = Y_OFFSETS * c1 - leg_z * s1
        foot_z_hip = Y_OFFSETS * s1 + leg_z * c1

        # base frame 기준
        foot_x = X_BASE_OFFSETS + foot_x_hip
        foot_y = Y_BASE_OFFSETS + foot_y_hip
        foot_z = foot_z_hip

        return np.vstack([foot_x, foot_y, foot_z])

    def get_foot_position(self, data):
        pass
    def get_foot_height(self, data):
        """월드 프레임에서의 발 높이 계산"""
        foot_body = self.cal_foot_body_position(data)
        base_z = data.qpos[2]
        
        # 쿼터니언으로 회전 행렬 생성
        quat = data.qpos[3:7]
        rot_matrix = self.quaternion_to_rotation_matrix(quat)
        
        # 발 위치를 월드 프레임으로 변환
        foot_world = rot_matrix @ foot_body
        foot_z_world = foot_world[2, :] + base_z
        
        return foot_z_world[:, np.newaxis]

    def quaternion_to_rotation_matrix(self, quat):
        """쿼터니언을 회전 행렬로 변환"""
        w, x, y, z = quat
        
        r11 = w**2 + x**2 - y**2 - z**2
        r12 = 2 * (x * y - w * z)
        r13 = 2 * (x * z + w * y)
        r21 = 2 * (x * y + w * z)
        r22 = w**2 - x**2 + y**2 - z**2
        r23 = 2 * (y * z - w * x)
        r31 = 2 * (x * z - w * y)
        r32 = 2 * (y * z + w * x)
        r33 = w**2 - x**2 - y**2 + z**2
        
        rot_matrix = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])
        
        return rot_matrix

LEG_INDICES = {
    'FL': [6, 7, 8],
    'FR': [9, 10, 11],
    'RL': [12, 13, 14],
    'RR': [15, 16, 17]
}

class FootForce:
    def __init__(self, model):
        self.y_pre = 0
        self.model = model
        self.M = np.zeros((model.nv, model.nv))
    
    def get_foot_force_jacobian(self, data, leg_name='FL'):
        model = self.model
        site_name = f"{leg_name}_foot"
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        idx = LEG_INDICES[leg_name]
        J_leg = jacp[:, idx]

        return J_leg

    def torque_to_force(self, data, tau_d):
        foot_force_z = []
        for leg_name in LEG_INDICES:
            leg_idx = LEG_INDICES[leg_name]
            tau_d_1leg = tau_d[leg_idx]
            J = self.get_foot_force_jacobian(data, leg_name)
            foot_force = np.linalg.pinv(J.T) @ tau_d_1leg
            foot_force_z.append(foot_force[-1])

        return np.vstack(foot_force_z)

    def get_foot_force(self, data):
        tau_d = self.get_disturbance_torque(data)
        fz = self.torque_to_force(data, tau_d)
        return fz

    def get_disturbance_torque(self, data):
        """
        Discrete-time Disturbance Observer (논문 Eq. 10)
        
        핵심 수정사항:
        1. qfrc_actuator 사용 (ctrl 대신 실제 적용된 토크)
           - forcerange 제한이 반영된 실제 토크
        2. gamma, beta는 15Hz cutoff, 500Hz sampling 기준
           - gamma = e^(-2π*15*0.002) ≈ 0.828
           - beta = (1-gamma)/(gamma*dt) ≈ 103.7
        """
        gamma = 0.828
        beta = 103.7
        
        # Mass matrix
        mujoco.mj_fullM(self.model, self.M, data.qM)
        
        # Generalized momentum: p = M * qdot
        q_dot = data.qvel[:, np.newaxis]
        p = self.M @ q_dot
        
        # Coriolis + Gravity (MuJoCo: qfrc_bias = C*qdot + g)
        qfrc_bias = data.qfrc_bias[:, np.newaxis]
        
        # 실제 적용된 actuator 토크 (forcerange 제한 반영)
        # qfrc_actuator는 이미 18x1 (base 6 + joints 12)
        qfrc_actuator = data.qfrc_actuator[:, np.newaxis]
        
        # Dynamic effects for filter
        # 논문: dyn = beta*p + S*tau + C*qdot - g
        # MuJoCo에서 qfrc_bias = C*qdot + g 이므로
        # dyn = beta*p + qfrc_actuator - qfrc_bias
        dyn_terms = beta * p + qfrc_actuator - qfrc_bias
        
        # Low-pass filter
        y = (1 - gamma) * dyn_terms + gamma * self.y_pre
        
        # Disturbance torque
        tau_d = beta * p - y
        
        # Save for next iteration
        self.y_pre = y

        return tau_d  # 18x1


if __name__ == "__main__":
    # 테스트 코드
    xml_path = './unitree_go2/scene_mjx.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    fh = FootHeight()
    ff = FootForce(model)

    for i in range(10):
        mujoco.mj_step(model, data)
        foot_height = fh.get_foot_height(data)
        foot_force = ff.get_foot_force(data)
        print(f"Step {i}: height={foot_height.flatten()}, force={foot_force.flatten()}")
