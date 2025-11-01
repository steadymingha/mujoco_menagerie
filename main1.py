import mujoco
import numpy as np
from mh.foot_mechanics import FootHeight
# from mh.ground_contact import ContactModel

SIMUL_TIME = 10

if __name__ == "__main__":
    xml_path = './unitree_go2/go2_mjx.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    fh = FootHeight()

    for i in range(SIMUL_TIME):
        mujoco.mj_step(model, data)

        pz = fh.get_foot_height(data)
        print(pz)
    