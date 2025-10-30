import numpy as np

class Kalman:
    def __init__(self):
        self.kalman_param()

        
    
    def kalman_param(self):
        self.A = 0
        self.H = 1
        self.Q = 0
        self.R = 4
        self.B = np.eye(4)
        
        # Initialization for estimation.
        self.x = np.array([0., 0., 0., 0.])  
        self.Sigma = np.eye(4)*0.1
        self.K = np.eye(4)

    def update(self):
        self.u = self.prediction_prob_model()

    def prob_contact(self):
        # model update
        self.update()

        # (1) Prediction.
        x_pred = self.A * x_esti + self.B * self.u
        Sigma_pred = self.A * self.Sigma * self.A + self.Q

        # (2) Kalman Gain.
        self.K = Sigma_pred * self.H / (self.H * Sigma_pred * self.H + self.R)

        # (3) Estimation.
        x_esti = x_pred + self.K * (z_meas - self.H * x_pred)

        # (4) Error Covariance.
        self.Sigma = Sigma_pred - self.K * self.H * Sigma_pred

    def get_current_phase(t0, t, T)
        phi = (t-t0)/T

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
        phi, s_phi = get_current_phase(t0, t, T)
        
        if s_phi: # stance state (0)
            prior_p = 0.5 * (erf((phi-mean_c[0])/math.sqrt(var_c_sq*2)) + erf((mean_c[1]-phi)/math.sqrt(var_c_sq*2)))
        else: # swing state(1)
            prior_p = 0.5 * (2 + erf((mean_cbar[0]-phi)/math.sqrt(var_cbar_sq*2)) + erf((phi-mean_cbar[1])/math.sqrt(var_cbar_sq*2)))
        
        return prior_p