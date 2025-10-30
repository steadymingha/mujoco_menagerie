import mujoco
import numpy as np
from mh.foot_height import FootHeight
from mh.ground_contact import ContactModel

if __name__ == "__main__":
    xml_path = './unitree_go2/go2_mjx.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    fh = FootHeight
    pz = fh.calculate_fl_foot_z()
    