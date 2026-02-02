import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class FootContactPlotter:
    """
    Class to plot foot contact data in real-time or save to file.
    Now supports comparison between Estimated Probability and Ground Truth.
    Also plots fz (foot force) and pz (foot height).
    """
    def __init__(self, max_len=50, draw_interval=1, enable_display=True):
        self.max_len = max_len
        self.draw_interval = draw_interval
        self.step_count = 0
        self.enable_display = enable_display

        # Queues for contact estimate and truth
        self.queues_est = [deque([0]*max_len, maxlen=max_len) for _ in range(4)]
        self.queues_truth = [deque([0]*max_len, maxlen=max_len) for _ in range(4)]

        # Queues for fz and pz (4 legs each)
        self.queues_fz = [deque([0]*max_len, maxlen=max_len) for _ in range(4)]
        self.queues_pz = [deque([0]*max_len, maxlen=max_len) for _ in range(4)]

        # Graph Setup: 6 subplots (4 contact + 1 fz + 1 pz)
        if self.enable_display:
            plt.ion()
        self.fig, self.axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)

        self.lines_est = []
        self.lines_truth = []
        self.lines_fz = []
        self.lines_pz = []

        leg_names = ['FL', 'FR', 'RL', 'RR']
        colors = ['blue', 'orange', 'green', 'red']

        # Contact subplots (0-3)
        for i in range(4):
            line_e, = self.axes[i].plot(np.arange(max_len), self.queues_est[i],
                                      color='blue', lw=1.5, linestyle=':', label='Estimate')
            line_t, = self.axes[i].plot(np.arange(max_len), self.queues_truth[i],
                                      color='orange', lw=2, linestyle='--', alpha=0.7, label='Truth')
            self.lines_est.append(line_e)
            self.lines_truth.append(line_t)
            self.axes[i].set_ylabel(leg_names[i], fontsize=10)
            self.axes[i].set_ylim(-0.2, 1.2)
            self.axes[i].grid(True, alpha=0.5)
            if i == 0:
                self.axes[i].legend(loc='upper right')

        self.axes[0].set_title("Contact: Estimate vs Truth", fontsize=12, pad=10)

        # fz subplot (index 4)
        for i in range(4):
            line_fz, = self.axes[4].plot(np.arange(max_len), self.queues_fz[i],
                                         color=colors[i], lw=1, label=leg_names[i])
            self.lines_fz.append(line_fz)
        self.axes[4].set_ylabel('Force (N)', fontsize=10)
        self.axes[4].set_title('Foot Force (fz)', fontsize=12)
        self.axes[4].legend(loc='upper right', fontsize=8)
        self.axes[4].grid(True, alpha=0.5)

        # pz subplot (index 5)
        for i in range(4):
            line_pz, = self.axes[5].plot(np.arange(max_len), self.queues_pz[i],
                                         color=colors[i], lw=1, label=leg_names[i])
            self.lines_pz.append(line_pz)
        self.axes[5].set_ylabel('Height (m)', fontsize=10)
        self.axes[5].set_title('Foot Height (pz)', fontsize=12)
        self.axes[5].legend(loc='upper right', fontsize=8)
        self.axes[5].grid(True, alpha=0.5)

        self.axes[5].set_xlabel("Time Window", fontsize=12)
        plt.tight_layout()

    def update(self, p_foot_contact, ground_truth, fz=None, pz=None):
        """
        Updates the graph with new data.
        Args:
            p_foot_contact: Estimated probability (4x1 or 4,)
            ground_truth: Actual contact state from physics (4,)
            fz: Foot force (4,) - optional
            pz: Foot height (4,) - optional
        """
        flat_est = p_foot_contact.flatten()
        flat_truth = ground_truth.flatten()

        for i in range(4):
            self.queues_est[i].append(flat_est[i])
            self.queues_truth[i].append(flat_truth[i])

        # Update fz/pz queues if provided
        if fz is not None:
            flat_fz = np.array(fz).flatten()
            for i in range(4):
                self.queues_fz[i].append(flat_fz[i])

        if pz is not None:
            flat_pz = np.array(pz).flatten()
            for i in range(4):
                self.queues_pz[i].append(flat_pz[i])

        # Redraw (only if display is enabled)
        self.step_count += 1
        if self.enable_display and self.step_count % self.draw_interval == 0:
            for i in range(4):
                self.lines_est[i].set_ydata(self.queues_est[i])
                self.lines_truth[i].set_ydata(self.queues_truth[i])
                self.lines_fz[i].set_ydata(self.queues_fz[i])
                self.lines_pz[i].set_ydata(self.queues_pz[i])

            # Auto-scale fz/pz axes
            self.axes[4].relim()
            self.axes[4].autoscale_view()
            self.axes[5].relim()
            self.axes[5].autoscale_view()

            plt.pause(0.001)

    def save(self, filename='foot_contact_plot.png'):
        """
        Saves the current plot to a file.
        Args:
            filename: Name of the file to save (default: 'foot_contact_plot.png')
        """
        # Update all lines before saving
        for i in range(4):
            self.lines_est[i].set_ydata(self.queues_est[i])
            self.lines_truth[i].set_ydata(self.queues_truth[i])
            self.lines_fz[i].set_ydata(self.queues_fz[i])
            self.lines_pz[i].set_ydata(self.queues_pz[i])

        # Auto-scale fz/pz axes
        self.axes[4].relim()
        self.axes[4].autoscale_view()
        self.axes[5].relim()
        self.axes[5].autoscale_view()

        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")

    def close(self):
        """
        Closes the plot (for real-time display mode).
        """
        if self.enable_display:
            plt.ioff()
            plt.show()
        plt.close(self.fig)
        print("Graph closed.")