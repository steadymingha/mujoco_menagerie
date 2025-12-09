import numpy as np
from scipy.special import erf
import math
# np.set_printoptions(precision=2, suppress=True)

class ContactModel:
    def __init__(self):
        self.kalman_param()
        self.T = 3 #0.6 # gait period [sec]
        self.THRESHOLD = 0.9
        

    def kalman_param(self):
        # System model
        n = 4
        self.A = np.zeros((4,4))
        self.H = np.vstack((np.eye(n), np.eye(n)))
        self.Sigma_w = 0.998 * np.eye(n)
        self.B = np.eye(n)

        # Measurement
        self.R = 4
        self.Sigma_v1 = 0.841 * np.eye(n)    
        self.Sigma_v2 = 0.930 * np.eye(n)
        self.Sigma_v = np.block([[self.Sigma_v1,np.zeros((n,n))], [np.zeros((n,n)),self.Sigma_v2]])
        
        # Initialization for estimation.
        self.x_esti = np.zeros((4, 1))
        self.Sigma = np.eye(4) * 0.1
        self.K = np.eye(4)
        self.z1, self.z2 = np.zeros((n,1)), np.zeros((n,1))
        
    def update(self, data, foot_height, foot_force):
        self.u = self.prediction_prob_model(data)   #prob_contact_given_state_subphase
        self.z1 = self.prob_contact_given_foot_height(foot_height)
        self.z2 = self.prob_contact_given_contact_force(foot_force)
        self.z = np.block([[self.z1],[self.z2]])
    
    def kalman(self):
        # (1) Prediction.
        x_pred = self.A @ self.x_esti + self.B @ self.u
        Sigma_pred = self.A @ self.Sigma @ self.A.T + self.Sigma_w

        # (2) Kalman Gain.
        self.K = Sigma_pred @ self.H.T @ (self.H @ Sigma_pred @ self.H.T + self.Sigma_v)

        # (3) Estimation.
        self.x_esti = x_pred + self.K @ (self.z - self.H @ x_pred)

        # (4) Error Covariance.
        self.Sigma = Sigma_pred - self.K @ self.H @ Sigma_pred
        self.x_esti = np.clip(self.x_esti, 0.0, 1.0)

        return self.x_esti # , self.K, self.Sigma

    def prob_contact(self, data, foot_height, foot_force):
        # model update
        self.update(data, foot_height, foot_force)
        p_foot_contact = self.kalman()
        return p_foot_contact
    
    # LEGS = ['front_left', 'front_right', 'hind_left', 'hind_right']
    def get_current_phase(self, t):
        phi0 = (t % self.T) / self.T # [0,1)
        offset = 0.5
        phi0_offset = (phi0 + offset) % 1.0
        phi = np.array([phi0_offset, phi0, phi0, phi0_offset])

        s_phi = np.where(phi < self.THRESHOLD, 1, 0) # phi < threshold, stance state(1)
        
        return phi, s_phi
    
    def prediction_prob_model(self, data): #prob_contact_given_state_subphase
        ## model parameter ##
        mean_cbar = np.array([0, 1])
        var_cbar_sq = 0.05
        mean_c = np.array([0, 1])
        var_c_sq = 0.05
        
        t = data.time # current time

        phi, s_phi = self.get_current_phase(t)
        
        denom_c = np.sqrt(var_c_sq * 2)
        denom_cbar = np.sqrt(var_cbar_sq * 2)

        # Stance(1)
        prob_stance = 0.5 * (erf((phi - mean_c[0]) / denom_c) + erf((mean_c[1] - phi) / denom_c))
        
        # Swing(0)
        prob_swing = 0.5 * (2 + erf((mean_cbar[0] - phi) / denom_cbar) + erf((phi - mean_cbar[1]) / denom_cbar))

        prior_p = s_phi * prob_stance + (1 - s_phi) * prob_swing
        
        return prior_p[:, np.newaxis]
    
    def prob_contact_given_foot_height(self, pz):
        mu_zg = 0 # mean
        sigma_zg = math.sqrt(0.1) # var

        p_c_pz = 0.5 * (1 + erf((mu_zg-pz)/(sigma_zg*math.sqrt(2))))
        # print(f'foot height : \n {p_c_pz[0].item():.2f}, {p_c_pz[1].item():.2f}')
        return p_c_pz

    def prob_contact_given_contact_force(self, fz):
        mu_fc = 40
        sigma_fc = math.sqrt(25)

        p_c_fz = 0.5 * (1 + erf((fz-mu_fc)/(sigma_fc*math.sqrt(2))))
        # print(f'contact force : \n {p_c_fz[0].item():.2f}, {p_c_fz[1].item():.2f}')

        return p_c_fz
