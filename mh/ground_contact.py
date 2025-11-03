import numpy as np
import math

class ContactModel:
    def __init__(self):
        self.kalman_param()

    
    def kalman_param(self):
        # System model
        n = 4
        self.A = 0
        self.H = np.vstack((np.eyes(n), np.eyes(n)))
        self.Sigma_w = 0.998 * np.eye(n)
        self.B = np.eye(n)

        # Measurement
        self.R = 4
        self.Sigma_v1 = 0.841 * np.eye(n)    
        self.Sigma_v2 = 0.930 * np.eye(n)
        self.Sigma_v = np.block([[self.Sigma_v1,np.zeros((n,n))], [np.zeros((n,n)),self.Sigma_v2]])
        
        # Initialization for estimation.
        self.x = np.array([0., 0., 0., 0.])  
        self.Sigma = np.eye(4) * 0.1
        self.K = np.eye(4)
        self.z1, self.z2 = np.zeros((n,1)), np.zeros((n,1))
        self.z = np.block([[self.z1],[self.z2]])

    def update(self, foot_height, foot_force):
        self.u = self.prediction_prob_model()   #prob_contact_given_state_subphase
        self.z1 = self.prob_contact_given_foot_height(foot_height)
        self.z2 = self.prob_contact_given_contact_force(foot_force)


    def prob_contact(self, foot_height, foot_force):
        # model update
        self.update(foot_height, foot_force)

        # (1) Prediction.
        x_pred = self.A @ x_esti + self.B @ self.u
        Sigma_pred = self.A @ self.Sigma @ self.A.T + self.Sigma_w

        # (2) Kalman Gain.
        self.K = Sigma_pred @ self.H.T @ (self.H @ Sigma_pred @ self.H.T + self.Sigma_v)

        # (3) Estimation.
        x_esti = x_pred + self.K * (self.z - self.H * x_pred)

        # (4) Error Covariance.
        self.Sigma = Sigma_pred - self.K * self.H * Sigma_pred

    def get_current_phase(t0, t, T):
        phi0 = (t-t0)/T # [0,1)
        phi = np.array([phi0, phi0])

        if phi < THRESHOLD:
            s_phi = 0
        else:
            s_phi = 1
        
        return phi, s_phi
    
    def prediction_prob_model(self): #prob_contact_given_state_subphase
        ## model parameter ##
        mean_cbar = np.array([0, 1])
        var_cbar_sq = 0.05
        mean_c = np.array([0, 1])
        var_c_sq = 0.05
        t0 = 0
        phi, s_phi = self.get_current_phase(t0, t, T)
        
        if s_phi: # stance state (0)
            prior_p = 0.5 * (erf((phi-mean_c[0])/math.sqrt(var_c_sq*2)) + erf((mean_c[1]-phi)/math.sqrt(var_c_sq*2)))
        else: # swing state(1)
            prior_p = 0.5 * (2 + erf((mean_cbar[0]-phi)/math.sqrt(var_cbar_sq*2)) + erf((phi-mean_cbar[1])/math.sqrt(var_cbar_sq*2)))
        
        return prior_p
    
    def prob_contact_given_foot_height(self, pz):
        mean_zg = 0 # mu
        var_zg = math.sqrt(0.1) # sigma

        p_c_pz = 0.5 * (1 + erf((mean_zg-pz)/(var_zg*math.sqrt(2))))

        return p_c_pz

    def prob_contact_given_contact_force(self, fz):
        return 0
