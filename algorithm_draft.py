## Implementation of Contact Model Fusion for Event-Based Locomotion in Unstructured Terrains

import mujoco
import numpy as np
from core.foot_mechanics import *
from core.ground_contact import ContactModel
import matplotlib.pyplot as plt

SIMUL_TIME = 1 # sec

if __name__ == "__main__":
    xml_path = './unitree_go2/go2_mjx_mh.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    ff = FootForce(model)
    fh = FootHeight()
    cm = ContactModel()

    print(model.opt.timestep)
    nsteps = int(np.ceil(SIMUL_TIME / model.opt.timestep))
    p_foot_contact_flat = []
    for i in range(nsteps):
        fz = ff.get_foot_force(data)
        pz = fh.get_foot_height(data)
        p_foot_contact = cm.prob_contact(data, pz, fz)
        

        p_foot_contact_flat.append(p_foot_contact.flatten())
        
        mujoco.mj_step(model, data)


    data = np.array(p_foot_contact_flat) 

    # 3. 4행 1열짜리 그래프 생성 (sharex=True: x축 공유해서 깔끔하게)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    leg_names = ['FL (Idx 0)', 'FR (Idx 1)', 'RL (Idx 2)', 'RR (Idx 3)']
    colors = ['red', 'blue', 'green', 'orange']

    # 4. 반복문으로 4개의 그래프 그리기
    for i in range(4):
        # data[:, i] -> 모든 시간(행)에 대해 i번째 다리(열) 값만 가져옴
        axes[i].plot(data[:, i], color=colors[i])
        axes[i].set_ylabel(leg_names[i]) # y축 이름
        axes[i].grid(True, alpha=0.3)    # 격자

    # 맨 마지막 그래프에만 x축 라벨 붙이기
    axes[3].set_xlabel('Time Step')
    axes[0].set_title('Foot Contact History by Leg')
    plt.tight_layout()
    # plt.legend()
    plt.grid()
    plt.savefig('plot_test.png',dpi=300, bbox_inches='tight')
    plt.close()
    

