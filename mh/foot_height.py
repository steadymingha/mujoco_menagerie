import numpy as np
import mujoco
from typing import Tuple



# 상수 정의
X_BASE_OFFSET = 0.1934
Y_BASE_OFFSET = 0.0465
Y_OFFSET = 0.0955
L_THIGH = 0.213
L_CALF = 0.213

class FootHeight:
    def __init__(self):
        pass

    def calculate_fl_foot_z(self, d):
        """
        MuJoCo data를 기반으로 FL(앞 왼쪽) 다리 발끝의 Z축 높이를 계산합니다.
        (몸통 base 좌표계 기준)

        Args:
            d: MuJoCo의 MjData 객체

        Returns:
            계산된 발끝의 Z축 높이
        """
        # 1. 시뮬레이션에서 현재 관절 각도를 가져옵니다.
        theta_abduction = d.joint('FL_hip_joint').qpos[0]
        theta_hip = d.joint('FL_thigh_joint').qpos[0]
        theta_knee = d.joint('FL_calf_joint').qpos[0]

        # 2. 1단계: Sagittal plane (옆에서 본) Z 높이 계산
        z_side = -L_THIGH * np.cos(theta_hip) - L_CALF * np.cos(theta_hip + theta_knee)

        # 3. 2단계: Abduction 효과를 적용하여 최종 Z 높이 계산
        z_final = z_side * np.cos(theta_abduction) + Y_OFFSET * np.sin(theta_abduction)
        
        return z_final

    def calculate_fl_foot_pos(self, d):
        """
        MuJoCo data를 기반으로 FL(앞 왼쪽) 다리 발끝의
        3D 위치 (X, Y, Z)를 계산합니다. (몸통 base 좌표계 기준)

        Args:
            d: MuJoCo의 MjData 객체

        Returns:
            (x, y, z) 위치를 담은 튜플
        """
        # 1. 시뮬레이션에서 현재 관절 각도를 가져옵니다.
        theta_abduction = d.joint('FL_hip_joint').qpos[0]
        theta_hip = d.joint('FL_thigh_joint').qpos[0]
        theta_knee = d.joint('FL_calf_joint').qpos[0]

        # 삼각함수 값 미리 계산 (효율성)
        s1 = np.sin(theta_abduction)
        c1 = np.cos(theta_abduction)
        s2 = np.sin(theta_hip)
        c2 = np.cos(theta_hip)
        s23 = np.sin(theta_hip + theta_knee)
        c23 = np.cos(theta_hip + theta_knee)

        # 2. X, Y, Z 위치 계산
        foot_x = X_BASE_OFFSET - L_THIGH * s2 - L_CALF * s23
        
        z_side = -L_THIGH * c2 - L_CALF * c23
        foot_y = Y_BASE_OFFSET + Y_OFFSET * c1 - z_side * s1
        foot_z = Y_OFFSET * s1 + z_side * c1
        
        return (foot_x, foot_y, foot_z)

    def quat_to_rot_matrix(self, q):
        """
        단위 쿼터니언을 3x3 회전 행렬로 변환합니다.

        Args:
            q (np.ndarray): (w, x, y, z) 순서의 쿼터니언.

        Returns:
            np.ndarray: 3x3 회전 행렬.
        """
        # 쿼터니언 성분 추출
        w, x, y, z = q[0], q[1], q[2], q[3]

        # 수식에 기반한 회전 행렬의 각 성분 계산
        # R11, R12, R13 (첫 번째 행)
        r11 = w**2 + x**2 - y**2 - z**2
        r12 = 2 * (x * y - w * z)
        r13 = 2 * (x * z + w * y)

        # R21, R22, R23 (두 번째 행)
        r21 = 2 * (x * y + w * z)
        r22 = w**2 - x**2 + y**2 - z**2
        r23 = 2 * (y * z - w * x)

        # R31, R32, R33 (세 번째 행)
        r31 = 2 * (x * z - w * y)
        r32 = 2 * (y * z + w * x)
        r33 = w**2 - x**2 - y**2 + z**2
        
        # 3x3 행렬 구성
        rot_matrix = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])
        
        return rot_matrix

if __name__ == "__main__":
    # --- 시뮬레이션 루프 (예시) ---
    # --- 사전 설정 ---
    # 모델 로드
    xml_path = './unitree_go2/go2_mjx.xml'
    model = mujoco.MjModel.from_xml_path(xml_path) # go2.xml 파일 경로
    data = mujoco.MjData(model)

    fh = FootHeight()

    for i in range(100):
        mujoco.mj_step(model, data)
        
                
        #   # 현재 스텝의 FL 발끝 3D 위치 계산 및 출력
        #   x, y, z = calculate_fl_foot_pos(data)
        #   print(f"FL foot position: (X={x:.4f}, Y={y:.4f}, Z={z:.4f})")
        # 현재 스텝의 FL 발끝 Z 높이 계산 및 출력
        foot_z = fh.calculate_fl_foot_z(data)
        print(f"FL foot Z height: {foot_z:.4f}")

        
        # --- 예제 및 결과 확인 ---
        # 예시: Z축을 기준으로 90도(pi/2 라디안) 회전하는 쿼터니언
        # w = cos(theta/2), (x,y,z) = sin(theta/2) * (axis_x, axis_y, axis_z)
        # theta = pi/2 -> theta/2 = pi/4
        # w = cos(pi/4) = 0.7071...
        # x = sin(pi/4) * 0 = 0
        # y = sin(pi/4) * 0 = 0
        # z = sin(pi/4) * 1 = 0.7071...
        # World 좌표계 기준 'base' 몸통의 위치 벡터
        p_body_world = np.array([0.5, 1.0, 0.4]) 

        # World 좌표계 기준 'base' 몸통의 방향 (Z축 90도 회전 쿼터니언)
        q_body_world = np.array([0.7071, 0, 0, 0.7071])

        # Body 좌표계 기준 발의 위치 벡터 (사용자가 계산한 값)
        p_foot_body = np.array([0.19, 0.15, -0.3])


        # 2. 쿼터니언을 회전 행렬로 변환 (위에서 만든 함수 사용)
        rot_matrix = fh.quat_to_rot_matrix(q_body_world)


        # 3. 최종 변환 수식 적용하여 World 좌표 계산 (지적해주신 바로 그 부분!)
        p_foot_world = p_body_world + rot_matrix @ p_foot_body


        # 4. 결과 출력
        print(f"Body 위치 (World): {p_body_world}")
        print(f"발 위치 (Body):    {p_foot_body}")
        print("-" * 30)
        # print("계산된 회전 행렬:")
        # print(np.round(rot_matrix, 5))
        print("-" * 30)
        print(f"최종 발 위치 (World): {np.round(p_foot_world, 5)}")
        print(f"World 기준 최종 발 높이(z): {np.round(p_foot_world[2], 5)}")







