import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import erf

THRESHOLD = 0.6

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

    phi, s_phi = get_current_phase(t0, t, T)
    
    if s_phi: # stance state (0)
        prior_p = 0.5 * (erf((phi-mean_c[0])/math.sqrt(var_c_sq*2)) + erf((mean_c[1]-phi)/math.sqrt(var_c_sq*2)))
    else: # swing state(1)
        prior_p = 0.5 * (2 + erf((mean_cbar[0]-phi)/math.sqrt(var_cbar_sq*2)) + erf((phi-mean_cbar[1])/math.sqrt(var_cbar_sq*2)))
    
    return prior_p

def get_foot_contact_force():
    # """Measure pocage."""
    # v = np.random.normal(0, 2)   # v: measurement noise.
    # poc_true = 14.4             # poc_true: True pocage [V].
    # z_poc_meas = poc_true + v  # z_poc_meas: Measured pocage [V] (observable).

    return z_poc_meas

def get_ground_height():
    return g_h

def kalman_filter(z_meas, x_esti, P, A, H, Q, R, B, u):
    """Kalman Filter Algorithm for One Variable.
       Return Kalman Gain for Drawing.
    """
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
    # Input parameters.
    time_end = 10
    dt = 0.2

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
        u = expected_contact_prob()
        z_1 = get_ground_height()
        z_2 = 
        if i == 0:
            x_esti, P, K = x_0, P_0, K_0
        else:
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
    main()