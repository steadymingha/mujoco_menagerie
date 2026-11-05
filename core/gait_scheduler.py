import numpy as np

from core.ground_contact import ContactModel
from core.fsm import QuadrupedContactFSM

LEGS = ["FL", "FR", "RL", "RR"]

# input s_hat, s_phi(fixed), t_cbar, t_c
# output s_phi

class GaitScheduler:
    def __init__(self, contactmodel, data):
        self.cm = contactmodel
        
        self.t = data.time
        self.t0 =  0 # start of current state, need to update 
        self.phi = self.cm.get_current_phase(self.t)


        t_cbar = self.cm.phi * (1 - self.cm.THRESHOLD) # fixed swing duration
        t_c = self.cm.phi * self.cm.THRESHOLD  # fixed stance duration

        self.fsm = QuadrupedContactFSM(t_cbar, t_c, self.t0)


    def gait_scheduler(self, s_hat): # s_hat is p_foot_contact(0 or 1)
        prev = dict.fromkeys(LEGS)

        for i, name in enumerate(LEGS):
            t_leg = ((self.phi[i] - self.cm.THRESHOLD) % 1.0) * self.cm.T
            self.fsm.legs[name].update(int(s_hat[i]), t_leg)

        contact = np.array([self.fsm.legs[name].s_phi for name in LEGS])

        # for i, name in enumerate(LEGS):   # 상태가 바뀐 순간만 출력
        #     state = fsm.legs[name].state.name
        #     if state != prev[name]:
        #         print(f"{t:6.3f} {name:>4} {phi_vec[i]:5.2f} {s_vec[i]:>5}  "
        #               f"{state:<8} {contact[i]:>5}")
        #         prev[name] = state

        return prev

    t0 update 확인할것


    





        



        