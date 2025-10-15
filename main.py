# 0. User Cmd
# 1. High level planning ()
# 2. leg and body control
# 3. State Estimation

# p dot = CoM translational vel
# psi dot = CoM turning rate

class HighLevelPlanning:
    def __init__(self):
        pass
    def desired_CoM_cmd(self,psi_dot,p_dot):
        pass

    def gait_scheduler(self):
        pass

class Controller:
    def __init__(self):
        pass
    def force_controller(self):
        pass
    def swing_leg_controller(self):
        pass

    def joint_pd_controller(self):
        pass


class StateObserver:
    def __init__(self):
        pass
    def CoM_state(self): # KF
        pass
    def leg_contact_detector(self):
        pass

class Cheetah:
    def __init__(self):
        pass

    def force_model(self):
        pass
    def p_psi_to_torque(self):
        #jacobian transpose
        pass


## leg-independent phase variable to schedule nominal contact and swing phases.
if __name__ == "__main__":
    import mujoco
    import numpy as np

    xml_path = './unitree_go2/go2_mjx.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    for i in range(19):
        mujoco.mj_step(model, data)

        orientation_data = data.sensor('orientation').data
        position_data = data.sensor('global_position').data
        print(f"joint sensor data test {data.sensor('knee_front_left_pos').data}")

        # print(f"Orientation (Quaternion): {orientation_data}")
        # print(f"Global Position (x, y, z): {position_data}")