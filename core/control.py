import numpy as np
from scipy.special import erf
import math

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
    def __init__(self, leg_position):
        p = leg_position
    # {FR, FL, BR, BL}
    
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

        # FL,FR, BL, BR -> FL FR BR BL 
        Phi[2], Phi[3] = Phi[3], Phi[2]
        p_4legs[2], p_4legs[3] = p_4legs[3], p_4legs[2]

         # cw : -, next / ccw : + , prev
        xi_i = []
        for i in range(len(p_4legs)):
            p_iplus  = p_4legs[(i - 1) % len(p_4legs)]
            p_i       = p_4legs[i]
            p_iminus = p_4legs[(i + 1) % len(p_4legs)]

            Phi_iplus = Phi[(i - 1) % len(p_4legs)]
            Phi_i     = Phi[i]
            Phi_iminus = Phi[(i + 1) % len(p_4legs)]
            
            xi_iminus = p_i * Phi + p_iminus * (1 - Phi)                   
            xi_iplus = p_i * Phi + p_iplus * (1- Phi)

            xi_i.append((Phi_i * p_i + Phi_iminus * xi_iminus + Phi_iplus * xi_iplus) / (Phi_i + Phi_iminus + Phi_iplus))
        
        p_CoM_desired = sum(xi_i) / len(xi_i)

        return p_CoM_desired

    def posture_adjustment(self, p_feet):  # for desired posture
        px = p_feet[0,:][:, np.newaxis]
        py = p_feet[1,:][:, np.newaxis]
        pz = p_feet[2,:][:, np.newaxis]
        W = np.concatenate((np.ones((4, 1)), px, py) , axis=1)
        a = np.linalg.lstsq(W, pz, rcond=None)[0]

        return a

        #### p_feet format ####
        #      | 발1 | 발2 | 발3| 발4
        # 0행(x)| x1 ​| x2 ​| x3​ | x4​
        # 1행(y)| y1 ​| y2 ​| y3​ | y4​
        # 2행(z)| z1 | z2 ​| z3 ​| z4​
        
    def force_PD_ctrl(self, p_c_d, ):
        pass

    

