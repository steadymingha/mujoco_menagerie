import numpy as np
import mujoco
from scipy.special import erf
import math
from core.util import *
from core.qp import BalanceQPSolver
from core.foot_mechanics_claude import LEG_INDICES
old_settings = np.seterr(all='raise')

class JointController:
    """Joint-space PD controller for Go2.

    Computes torque commands and writes them into sim.ctrl0.
    Joint order per leg: [Abd, Hip, Knee]
    """

    legs_indices = [
        [0, 1, 2],    # FL
        [3, 4, 5],    # FR
        [6, 7, 8],    # RL
        [9, 10, 11],  # RR
    ]

    # Joint limits from XML (with small margin)
    KNEE_LIMIT_UPPER = -0.85   # calf_joint upper: -0.84
    KNEE_LIMIT_LOWER = -2.70   # calf_joint lower: -2.72

    def __init__(self, sim, kp: float = 100.0, kd: float = 5.0):
        self.sim = sim
        self.kp = kp
        self.kd = kd
        self.target_abds  = [0.0,  0.0,  0.0,  0.0]
        self.target_hips  = [0.8,  0.8,  0.8,  0.8]
        self.target_knees = [-1.8, -1.8, -1.8, -1.8]
        # self.target_hips  = [0.6,  0.6,  0.6,  0.6]
        # self.target_knees = [-1.4, -1.4, -1.4, -1.4]

    def compute(self):
        """PD 토크 계산 후 sim.ctrl0 에 기록."""
        sim = self.sim
        kp, kd = self.kp, self.kd
        sim.ctrl0[:] = 0.0

        for leg_idx in range(4):
            i_abd, i_hip, i_knee = self.legs_indices[leg_idx]

            curr_abd = sim.data.qpos[7 + i_abd]
            vel_abd  = sim.data.qvel[6 + i_abd]
            sim.ctrl0[i_abd] = kp * (self.target_abds[leg_idx] - curr_abd) - kd * vel_abd

            curr_hip = sim.data.qpos[7 + i_hip]
            vel_hip  = sim.data.qvel[6 + i_hip]
            sim.ctrl0[i_hip] = kp * (self.target_hips[leg_idx] - curr_hip) - kd * vel_hip

            curr_knee = sim.data.qpos[7 + i_knee]
            vel_knee  = sim.data.qvel[6 + i_knee]
            target_knee = np.clip(self.target_knees[leg_idx], self.KNEE_LIMIT_LOWER, self.KNEE_LIMIT_UPPER)
            sim.ctrl0[i_knee] = kp * (target_knee - curr_knee) - kd * vel_knee


class GaitController:
    def __init__(self):#, leg_position):
        # p = leg_position

        self.pcom_d_prev = np.zeros(3)
        self.pcom_prev = np.zeros(3)
        self.yaw_dprev = 0.0
    # {FR, FL, BR, BL}

        self.qp = BalanceQPSolver(mu=0.6, fz_min=10, fz_max=500)

        self.legs_indices = [
        [0, 1, 2],    # FL
        [3, 4, 5],    # FR
        [6, 7, 8],    # RL
        [9, 10, 11],  # RR
    ]
        self.com_offset = None
    
    def weighting_factor(self, s_phi, phi):
        sigma_c0_sq = 0.1 # 0.05 ~ 0.2 var = sigma_c0_sq = sigma_c1_sq
        sigma_cbar0_sq = 0.1 # var = sigma_cbar0_sq = sigma_cbar1_sq

        denom_c0 = denom_c1 = np.sqrt(sigma_c0_sq * 2) # 
        denom_cbar0 = denom_cbar1 = np.sqrt(sigma_cbar0_sq * 2)

        K_c_phi = 0.5 * (erf(phi / denom_c0) + erf((1 - phi) / denom_c1))
        K_cbar_phi = 0.5 * (2 + erf(-phi / denom_cbar0)    
                              + erf((phi - 1) / denom_cbar1))
        
        Phi = s_phi * K_c_phi + (1-s_phi) * K_cbar_phi
        
        return Phi

    def predictive_support_polygon(self, p_4legs, s_phi, phi): # needs all leg positions
        Phi = self.weighting_factor(s_phi, phi)
        p_4legs = p_4legs.copy()
        Phi = Phi.copy()
        
        # # FL,FR, BL, BR -> FL FR BR BL 
        # Phi[2], Phi[3] = Phi[3], Phi[2]
        # p_4legs[2], p_4legs[3] = p_4legs[3], p_4legs[2]

        # FL,FR,RL,RR → FR,RR,RL,FL  (논문 CW: FR→BR→BL→FL)
        order = [1, 3, 2, 0]
        p_4legs = p_4legs[order]
        Phi = Phi[order]

         # cw : -, next / ccw : + , prev
        xi_i = []
        for i in range(len(p_4legs)):
            p_iplus  = p_4legs[(i - 1) % len(p_4legs)]
            p_i       = p_4legs[i]
            p_iminus = p_4legs[(i + 1) % len(p_4legs)]

            Phi_iplus = Phi[(i - 1) % len(p_4legs)]
            Phi_i     = Phi[i]
            Phi_iminus = Phi[(i + 1) % len(p_4legs)]
            
            xi_iminus = p_i * Phi[i] + p_iminus * (1 - Phi[i])                   
            xi_iplus = p_i * Phi[i] + p_iplus * (1- Phi[i])

            xi_i.append((Phi_i * p_i + Phi_iminus * xi_iminus + Phi_iplus * xi_iplus) / (Phi_i + Phi_iminus + Phi_iplus))
        
        p_CoM_desired = sum(xi_i) / len(xi_i)

        return p_CoM_desired

    def posture_adjustment(self, p_feet):  # for desired posture
        px = p_feet[0,:][:, np.newaxis]
        py = p_feet[1,:][:, np.newaxis]
        pz = p_feet[2,:][:, np.newaxis]
        W = np.concatenate((np.ones((4, 1)), px, py) , axis=1)
        a = np.linalg.lstsq(W, pz, rcond=None)[0]

        pitch_d = -np.arctan(a[1].item()) # Slope along the x-axis
        roll_d = np.arctan(a[2].item())    # Slope along the y-axis
        
        # z(x, y) = a0 + a1x + a2y
        return pitch_d, roll_d, a

        #### p_feet format ####  3x4
        #      | 발1 | 발2 | 발3| 발4
        # 0행(x)| x1 ​| x2 ​| x3​ | x4​
        # 1행(y)| y1 ​| y2 ​| y3​ | y4​
        # 2행(z)| z1 | z2 ​| z3 ​| z4​
    
    def translational_acceleration_cmd(self, data, pcom_d, pcom, pcomdot_d=0):
        # Kp_p = 50#30.0
        # Kd_p = 50#40.0
        # Kp_p = np.array([50.0, 50.0, 50.0])
        # Kd_p = np.array([50.0, 50.0, 30.0])
        Kp_p = np.array([100.0, 100.0, 100.0])
        Kd_p = np.array([100.0, 50.0,100.0])

        if (self.pcom_d_prev == 0).all(): 
            self.pcom_d_prev = pcom_d
            self.pcom_prev = pcom

        # pcomdot_d = (pcom_d - self.pcom_d_prev) / dt
        self.pcomdot = data.subtree_linvel[0] #(pcom - self.pcom_prev) / dt


        pcom2dot_d = Kp_p * (pcom_d - pcom) + Kd_p * (pcomdot_d - self.pcomdot)
        
        self.pcom_d_prev = pcom_d
        self.pcom_prev = pcom

        return pcom2dot_d
    
    def angular_acceleration_cmd(self, sim, data, dt, roll_d, pitch_d, yaw_d=0):
        # Kp_w = 50.0
        # kd_w = 25.0
        Kp_w = np.array([100.0, 100.0, 100.0])
        Kd_w = np.array([5.0, 100.0,50.0])
        q = data.sensor('orientation').data
        R = quaternion_to_R(q)
        R_d = euler123_to_R(roll_d, pitch_d, yaw_d)

        # 목표 오일러각 속도 (유한차분)
        euler_d = np.array([roll_d, pitch_d, yaw_d])
        if not hasattr(self, '_prev_euler_d'):
            self._prev_euler_d = euler_d.copy()

        eulerdot_d = (euler_d - self._prev_euler_d) / dt
        self._prev_euler_d = euler_d

        # 자코비안으로 body frame 목표 각속도 변환
        B = euler123_jacobian(pitch_d, yaw_d)
        # omega_d = B @ eulerdot_d
        omega_d_body = B @ eulerdot_d
        omega_d_world = R_d @ omega_d_body  # desired body → world

        omega = data.sensor('gyro').data  # body frame 현재 각속도 [wx, wy, wz]

        omega_world = R @ omega  # body → world
        omegadot_world = Kp_w * SO3_logmap(R_d @ R.T) + Kd_w * (omega_d_world - omega_world)
        
        return omegadot_world
        # omegadot_b_d = Kp_w * SO3_logmap(R_d @ R.T) + kd_w * (omega_d - omega)
        
        # return omegadot_b_d
    
    def balance_controller(self, sim, p, s_phi, phi, J, pcomdotb_d, yaw_rate_d = 0): # p : estimation, q : sensor(gyro)
        model = sim.model
        data = sim.data
        dt = model.opt.timestep
        pcom = data.subtree_com[0]
        pcom_d = self.predictive_support_polygon(p.T, s_phi, phi)
        pitch_d, roll_d, slope = self.posture_adjustment(p)
        
        if self.com_offset is None:
            foot_center = pcom_d.copy()
            self.com_offset = pcom[:2] - foot_center[:2]

        pcom_d[0] += self.com_offset[0]
        pcom_d[1] += self.com_offset[1]

        ## desired robot height calculation
        z0 = 0.225#25#0.225 #0.27
        # z_gnd = ( np.array([1, pcom[0], pcom[1]]) @ slope ).item()
        z_gnd = slope[0].item()
        pcom_d_z = z_gnd + z0
        pcom_d[2] = pcom_d_z

        # if 1.5 < sim.sim_time < 1.52:
        #     print(f"feet z: {p[2,:]}")
        #     print(f"z_gnd: {z_gnd:.4f}, z0: {z0}, pcom_d_z: {pcom_d_z:.4f}")
        #     print(f"pcom_z - z_gnd: {pcom[2]-z_gnd:.4f}")
        # yaw_d calculation ##
        yaw_d = self.yaw_dprev + yaw_rate_d * dt
        self.yaw_dprev = yaw_d

        ## p dot com d calculation (body -> world)
        R_yaw = euler123_to_R(0, 0, yaw_d)
        self.pcomdot_d = R_yaw @ pcomdotb_d

        # pcomdotb_d = 0 # translate user input pcomdotb_d to world frame
        # yaw_d = 0 # integrate user input yaw_rate_d


        pcom2dot_d = self.translational_acceleration_cmd(sim.data, pcom_d, pcom, self.pcomdot_d)
        # omegadotb_d = self.angular_acceleration_cmd(sim.data, dt, roll_d, pitch_d, yaw_d)
        omegadotw_d = self.angular_acceleration_cmd(sim ,sim.data, dt, roll_d, pitch_d, yaw_d)
        # omegadotw_d = np.zeros(3)
        # b_d = self.force_desired(sim, pcom2dot_d, omegadotb_d)
        b_d = self.force_desired(sim, pcom2dot_d, omegadotw_d)

        # if 3.0 < sim.sim_time < 5.0:
        #     m = sim.model.body_subtreemass[0]
        #     b_d = np.array([0, 0, m*9.8, 0, 5.0, 0])
            
        # S = np.diag([1, 1, 10, 5, 5, 5])
        S = np.diag([10, 10, 10, 5, 5, 5])
        tq_d = self.qp_solution(pcom, p, s_phi, b_d, S, J)
        # if sim.sim_time % 0.5 < sim.model.opt.timestep and sim.sim_time > 1.5:
        #     print(f"t={sim.sim_time:.1f} pcom_z={pcom[2]:.4f} pcom_d_z={pcom_d[2]:.4f}")
        return pcom_d, pcom, tq_d
    
    # def force_desired(self, sim, pcom2dot_d, omegadot_b_d):
    def force_desired(self, sim, pcom2dot_d, omegadot_w_d):
        m = sim.model.body_subtreemass[0]
        g = np.array([0, 0, 9.8]).T

        # cinert_data = sim.data.cinert[1]  # 인덱스 1이 보통 base body.

        # # 2. 3x3 회전 관성 행렬(Inertia Tensor) 성분 추출
        # # 대각 성분 (Ixx, Iyy, Izz)
        # ixx = cinert_data[4]
        # iyy = cinert_data[5]
        # izz = cinert_data[6]

        # # 비대각 성분 (Ixy, Ixz, Iyz)
        # ixy = cinert_data[7]
        # ixz = cinert_data[8]
        # iyz = cinert_data[9]

        # # 3. I_G 행렬 (world frame, 3x3 Symmetric Matrix)
        # I_G_world = np.array([
        #     [ixx, ixy, ixz],
        #     [ixy, iyy, iyz],
        #     [ixz, iyz, izz]
        # ])

        M = np.zeros((sim.model.nv, sim.model.nv))
        mujoco.mj_fullM(sim.model, M, sim.data.qM)
        I_G_world = M[3:6, 3:6]  # 회전-회전 블록

        # 4. world frame → body frame 변환: I_body = R.T @ I_world @ R
        q = sim.data.sensor('orientation').data
        R = quaternion_to_R(q)
        # I_G = R.T @ I_G_world @ R

        # b_d = np.concatenate([m * (pcom2dot_d + g), I_G @ omegadot_b_d])
        b_d = np.concatenate([m * (pcom2dot_d + g), I_G_world @ omegadot_w_d])
        # if 3.0 < sim.sim_time < 3.02:
        #     print(f"omegadot_w_d: {omegadot_w_d}")
        #     print(f"I_G_world:\n{I_G_world}")
        #     print(f"I_G_world @ omegadot: {I_G_world @ omegadot_w_d}")
        return b_d
        
    def qp_solution(self, p_com, p_feet, contact, bd, S, J):
        F, ok = self.qp.solve(p_com, p_feet.T, contact, bd, S)
        torques = []
        for i, leg_idx in enumerate(self.legs_indices):
            F_1leg = F[:, np.newaxis][leg_idx]
            torques.append(-J[i].T @ F_1leg)
        
        return np.vstack(torques)

        #### p_feet format ####  3x4
        #      | 발1 | 발2 | 발3| 발4
        # 0행(x)| x1 ​| x2 ​| x3​ | x4​
        # 1행(y)| y1 ​| y2 ​| y3​ | y4​
        # 2행(z)| z1 | z2 ​| z3 ​| z4​

        ## p_hip format 2x4 ( FL FR BL BR )
        #      | 힙1 | 힙2 | 힙3| 힙4
        # 0행(x)| x1 ​| x2 ​| x3​ | x4​
        # 1행(y)| y1 ​| y2 ​| y3​ | y4​
 

    def swing_controller(self, p_hip, T_cphi):
        # the step location on the 2D ground plane for foot i
        # self.pcomdot_d, pcomdot -> Values retrieved from balance_controller
        # Tc_phi : nominal scheduled contact phase (stance) time
        # p_hip : position of the corresponding hip i
        # 2D projection
        g = 9.8
        z0 = 0.225  # Keep in sync with balance_controller

        p_step = []
        for i in range(4): # p_hip.T[i] 2x1
            p_step.append(p_hip.T[i] + 0.5 * T_cphi * self.pcomdot_d[0:2] + math.sqrt(z0/g) * (self.pcomdot[0:2] - self.pcomdot_d[0:2]))

        #T_stance = T_gait * duty_factor
        # duty_factor: 한 주기에서 스탠스 비율
        # 예) trot: duty_factor ≈ 0.5

        

    

