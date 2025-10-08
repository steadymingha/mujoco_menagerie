import mujoco
import mujoco_viewer
# from mujoco.usd import exporter
import numpy as np


j = {
    "RF_hip": 0,   # Right Front Hip
    "RF_thigh": 1, # Right Front Thigh
    "RF_calf": 2,  # Right Front Calf
    "LF_hip": 3,   # Left Front Hip
    "LF_thigh": 4, # Left Front Thigh
    "LF_calf": 5,  # Left Front Calf
    "RR_hip": 6,   # Right Rear Hip
    "RR_thigh": 7, # Right Rear Thigh
    "RR_calf": 8,  # Right Rear Calf
    "LR_hip": 9,   # Left Rear Hip
    "LR_thigh": 10,# Left Rear Thigh
    "LR_calf": 11  # Left Rear Calf
}

# left direction - 양수
# right direction - 음수



# Front right thigh, Front right calf, Rear right thigh, Rear right calf


def main():
    # 모델 로드
    # MODEL_XML = "unitree_go1/scene.xml"
    MODEL_XML = "unitree_go2/scene_mjx.xml"
    # pose_data = "dog_pace.npy"
    # [right_front_thigh_angle, right_front_calf_angle, right_rear_thigh_angle, right_rear_calf_angle]
    # poses = np.load(pose_data)
    # mujoco.MjsCamera.targetbody
    model = mujoco.MjModel.from_xml_path(MODEL_XML)


    data = mujoco.MjData(model)
    # sim->cam.type = mjCAMERA_TRACKING;
    # Viewer 실행
    viewer = mujoco_viewer.MujocoViewer(model, data)
    # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    # viewer._render_every_frame = False
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body("base").id  # 따라갈 바디 지정
    viewer.cam.distance = 3.0  # 카메라와 바디 사이 거리

    # qpos = data.qpos  # 상태 변수
    # qpos[3:7] = [0,1, 0, 0]  # 쿼터니언 (x축 90도 회전)
    # data.qpos[:] = qpos  # 초기 상태 적용

    # 모니터링할 관절 이름 설정
    # joint_name = "FR_thigh_joint"  # 예: "FR_hip_joint"

    joint_names = ['FR_thigh_joint', 'FR_calf_joint', 'RR_thigh_joint', 'RR_calf_joint', 'FL_thigh_joint', 'FL_calf_joint', 'RL_thigh_joint', 'RL_calf_joint']

    try:
        for i, joint_name in enumerate(joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_index = model.jnt_qposadr[joint_id]
            angle = data.qpos[qpos_index]
            print(f"Joint '{joint_name}' joint id : '{joint_id}', qpos id : '{qpos_index}', angle: {angle:.3f} rad")
    except mujoco.Error:
        print(f"Joint '{joint_name}' not found in the model.")

    cnt = 0
    while viewer.is_alive:
        # if cnt % 5 == 0:
            # data.ctrl[j["RF_thigh"]] = pace_data[alive_flag][0]
            # data.ctrl[j["RF_calf"]] = pace_data[alive_flag][1] *2
            # data.ctrl[j["RR_thigh"]] = pace_data[alive_flag][2]
            # data.ctrl[j["RR_calf"]] = pace_data[alive_flag][3] *2
            # data.ctrl[j["LF_thigh"]] = pace_data[alive_flag][4]
            # data.ctrl[j["LF_calf"]] = pace_data[alive_flag][5] *2
            # data.ctrl[j["LR_thigh"]] = pace_data[alive_flag][6]
            # data.ctrl[j["LR_calf"]] = pace_data[alive_flag][7]   *2
        mujoco.mj_step(model, data)
        viewer.render()
        cnt += 1



    viewer.close()


if __name__ == "__main__":
    main()