# # original : tutorial.py(mujoco.viewer), mujoco_viewer : unofficial viewer
from sim import Go2Sim 
import numpy as np
from mh.foot_mechanics import *
from mh.ground_contact import ContactModel

if __name__ == "__main__":
    # main()
    from sim import RandomController, Go2Sim
    model_file = "unitree_go1/scene.xml"
    sim = Go2Sim(model_file)
    
    # Sim initialization
    sim.reset()

    # my_controller = RandomController(sim.model)

    sim.run(controller=my_controller, estimator=None)




