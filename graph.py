import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class FootContactPlotter:
    """
    Class to plot foot contact data in real-time.
    Now supports comparison between Estimated Probability and Ground Truth.
    """
    def __init__(self, max_len=50, draw_interval=1):
        self.max_len = max_len
        self.draw_interval = draw_interval 
        self.step_count = 0
        
        # --- [MODIFIED] Create queues for both Estimate and Truth ---
        # 1. Queue for Estimated Probability
        self.queues_est = [deque([0]*max_len, maxlen=max_len) for _ in range(4)]
        # 2. Queue for Ground Truth (Physics engine data)
        self.queues_truth = [deque([0]*max_len, maxlen=max_len) for _ in range(4)]
        # ------------------------------------------------------------
        
        # Graph Setup
        plt.ion() 
        self.fig, self.axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        
        self.lines_est = []
        self.lines_truth = [] # --- [ADDED] List for truth lines ---
        
        titles = ['Front Left (FL)', 'Front Right (FR)', 'Rear Left (RL)', 'Rear Right (RR)']
        
        for i in range(4):
            # --- [MODIFIED] Plot two lines per subplot ---
            # Line 1: Estimated Probability (Blue, Solid)
            line_e, = self.axes[i].plot(np.arange(max_len), self.queues_est[i], 
                                      color='blue', lw=1.5, label='Estimate')
            
            # Line 2: Ground Truth (Orange, Dashed)
            line_t, = self.axes[i].plot(np.arange(max_len), self.queues_truth[i], 
                                      color='orange', lw=2, linestyle='--', alpha=0.7, label='Truth')
            
            self.lines_est.append(line_e)
            self.lines_truth.append(line_t)
            # ---------------------------------------------
            
            self.axes[i].set_ylabel(titles[i], fontsize=10)
            self.axes[i].set_ylim(-0.2, 1.2) 
            self.axes[i].grid(True, alpha=0.5)
            
            # Add legend to the first subplot
            if i == 0:
                self.axes[i].legend(loc='upper right')
            
        self.axes[0].set_title("Real-time Contact: Estimate vs Truth", fontsize=14, pad=10)
        self.axes[3].set_xlabel("Time Window", fontsize=12)
        plt.tight_layout()

    def update(self, p_foot_contact, ground_truth):
        """
        Updates the graph with new data.
        Args:
            p_foot_contact: Estimated probability (4x1 or 4,)
            ground_truth: Actual contact state from physics (4,)
        """
        # --- [MODIFIED] Update both queues ---
        flat_est = p_foot_contact.flatten()
        flat_truth = ground_truth.flatten()
        
        for i in range(4):
            self.queues_est[i].append(flat_est[i])
            self.queues_truth[i].append(flat_truth[i])
        # -------------------------------------

        # Redraw
        self.step_count += 1
        if self.step_count % self.draw_interval == 0:
            for i in range(4):
                self.lines_est[i].set_ydata(self.queues_est[i])
                self.lines_truth[i].set_ydata(self.queues_truth[i]) # --- [ADDED] Update truth line ---
            
            plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.show()
        print("Graph closed.")