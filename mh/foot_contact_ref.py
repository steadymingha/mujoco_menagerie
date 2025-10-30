import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import erf
import mujoco
from foot_height import quat_to_rot_matrix

THRESHOLD = 0.6
# 링크 길이 등 상수 정의
X_BASE_OFFSET = 0.1934
Y_BASE_OFFSET = 0.0465
Y_OFFSET = 0.0955
L_THIGH = 0.213
L_CALF = 0.213

def get_current_phase(t0, t, T):
    phi = (t-t0)/T

    if phi < THRESHOLD:
        s_phi = 0
    else:
        s_phi = 1
    
    return phi, s_phi

def expected_contact_prob(): #  probabilistic model for the expectation of contact given the scheduled leg state & subphase during stance
    ## model parameter ##
    mean_cbar = np.array([0, 1])
    var_cbar_sq = 0.05
    mean_c = np.array([0, 1])
    var_c_sq = 0.05
    t0 = 0
    phi, s_phi = get_current_phase(t0, t, T)
    
    if s_phi: # stance state (0)
        prior_p = 0.5 * (erf((phi-mean_c[0])/math.sqrt(var_c_sq*2)) + erf((mean_c[1]-phi)/math.sqrt(var_c_sq*2)))
    else: # swing state(1)
        prior_p = 0.5 * (2 + erf((mean_cbar[0]-phi)/math.sqrt(var_cbar_sq*2)) + erf((phi-mean_cbar[1])/math.sqrt(var_cbar_sq*2)))
    
    return prior_p

def cal_disturbance_torque():
    ##  p = M(q)q_dot
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)


    u = beta*p + S.T*tau + C.T*q_dot - g
    y = (1-gamma) * u + gamma*y_prev
    tau_d = beta*p - y
    
    y_prev = y
    return tau_d
    
def get_foot_contact_force(mjc):
    
    
    return z_poc_meas

def cal_foot_body_position(data):
    # 다리 위치 리스트
    LEGS = ['front_left', 'front_right', 'hind_left', 'hind_right']
    # 관절 유형 리스트
    JOINTS = ['abduction', 'hip', 'knee']
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
    
    feet_pos = np.stack([foot_x, foot_y, foot_z], axis=1)
    return feet_pos


def get_foot_height(data):
    feet_b = cal_foot_body_position(data)
    # World 좌표계 기준 'base' 몸통의 위치 벡터
    body_w = np.array([0.5, 1.0, 0.4]) 

    # World 좌표계 기준 'base' 몸통의 방향 (Z축 90도 회전 쿼터니언)
    q_body_w = np.array([0.7071, 0, 0, 0.7071])
    rot_matrix = quat_to_rot_matrix(q_body_w)

    foot_w = []
    for foot_b in feet_b:
        foot_w.append(body_w + rot_matrix @ foot_b)

    return np.array(foot_w)[:,-1] # pick foot z height only

def prob_contact_from_foot_height(data):
    mu_zg = 0 # mean
    sigma_zg = math.sqrt(0.1) # var
    pz = get_foot_height(data)
    
    p_c_given_pz = []
    for pz_i in pz:
        p_c_given_pz.append( 0.5 * (1+ erf((mu_zg-pz_i)/(sigma_zg*math.sqrt(2)))))
    
    return np.array(p_c_given_pz)

# def get_contact_force():
#     return c_f

def kalman_filter(z_meas, x_esti, P, u):
    """Kalman Filter Algorithm for One Variable.
       Return Kalman Gain for Drawing.
    """
    # Initialization for system model.
    A = 0
    H = 1
    Q = 0
    R = 4
    B = np.eye(4)
    
    # Initialization for estimation.
    x_0 = np.array([0., 0., 0., 0.])  # 14 for book.
    P_0 = np.eye(4)*0.1
    K_0 = np.eye(4)

    if i == 0:
            x_esti, P, K = x_0, P_0, K_0
    else:
    # (1) Prediction.
    x_pred = A * x_esti + B * u
    P_pred = A * P * A + Q

    # (2) Kalman Gain.
    K = P_pred * H / (H * P_pred * H + R)

    # (3) Estimation.
    x_esti = x_pred + K * (z_meas - H * x_pred)

    # (4) Error Covariance.
    P = P_pred - K * H * P_pred

    return x_esti, P, K

def main():
    xml_path = './unitree_go2/go2_mjx.xml'
    model = mujoco.MjModel.from_xml_path(xml_path) # go2.xml 파일 경로
    data = mujoco.MjData(model)

    # Input parameters.
    time_end = 10
    dt = 0.2

    

    # Create time array and initialize storage arrays
    time = np.arange(0, time_end, dt)
    n_samples = len(time)
    poc_meas_save = np.zeros(n_samples)
    poc_esti_save = np.zeros(n_samples)
    P_save = np.zeros(n_samples)
    K_save = np.zeros(n_samples)

    # Run Kalman filter
    x_esti, P, K = None, None, None
    for i in range(n_samples):
        # u = expected_contact_prob()
        z_1 = prob_contact_from_foot_height(data)
        # z_2 = 

        x_esti, P, K = kalman_filter(z_meas, x_esti, P, A, H, Q, R, B, u)

        poc_meas_save[i] = z_meas
        poc_esti_save[i] = x_esti
        P_save[i] = P
        K_save[i] = K


    # Create plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 10))

    # Plot 1: Measurements vs Estimation
    plt.subplot(1, 3, 1)
    plt.plot(time, poc_meas_save, 'r*--', label='Measurements', markersize=15)
    plt.plot(time, poc_esti_save, 'bo-', label='Kalman Filter', markersize=15)
    plt.legend(loc='upper left', fontsize=20)
    plt.title('Measurements v.s. Estimation (Kalman Filter)', fontsize=20)
    plt.xlabel('Time [sec]', fontsize=25)
    plt.ylabel('pocage [V]', fontsize=25)

    # Plot 2: Error Covariance
    plt.subplot(1, 3, 2)
    plt.plot(time, P_save, 'go-', markersize=15)
    plt.title('Error Covariance (Kalman Filter)', fontsize=20)
    plt.xlabel('Time [sec]', fontsize=25)
    plt.ylabel('Error Covariance (P)', fontsize=25)

    # Plot 3: Kalman Gain
    plt.subplot(1, 3, 3)
    plt.plot(time, K_save, 'ko-', markersize=15)
    plt.title('Kalman Gain (Kalman Filter)', fontsize=20)
    plt.xlabel('Time [sec]', fontsize=25)
    plt.ylabel('Kalman Gain (K)', fontsize=25)

    # Save the plot
    plt.tight_layout()
    plt.savefig('png/simple_kalman_filter.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print final results
    print(f"Final estimated pocage: {x_esti:.2f} V")
    print(f"True pocage: 14.4 V")
    print(f"Final error covariance: {P:.4f}")
    print(f"Final Kalman gain: {K:.4f}")
    print("Plot saved to: png/simple_kalman_filter.png")

if __name__ == "__main__":
    # main()

    # 모델과 데이터 로드
    xml_path = './unitree_go2/go2_mjx.xml'
    model = mujoco.MjModel.from_xml_path(xml_path) # go2.xml 파일 경로
    data = mujoco.MjData(model)

    ## 1. Contact Probability by Kalman



    ## 2. Contact State Determination



    ## 3. Building Contact State FSM