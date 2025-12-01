## Implementation of Contact Model Fusion for Event-Based Locomotion in Unstructured Terrains 

import mujoco
import numpy as np
from mh.foot_mechanics import *
from mh.ground_contact import ContactModel

SIMUL_TIME = 1 # sec

if __name__ == "__main__":
    xml_path = './unitree_go2/go2_mjx.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    fh = FootHeight()
    ff = FootForce(model)
    cm = ContactModel()

    print(model.opt.timestep)
    nsteps = int(np.ceil(SIMUL_TIME / model.opt.timestep))
    for i in range(nsteps):
        print(i)
        pz = fh.get_foot_height(data)
        fz = ff.get_foot_force(model, data)
        p_foot_contact = cm.prob_contact(data, pz, fz)
        mujoco.mj_step(model, data)


    