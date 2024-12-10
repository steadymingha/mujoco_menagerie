import mujoco
import mujoco_viewer
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






def main():
    # 모델 로드
    # MODEL_XML = "unitree_go1/scene.xml"
    MODEL_XML = "/Users/mhlee/study/mujoco_menagerie/unitree_go1/scene.xml"
    pose_data = "dog_pace.npy"
    # [right_front_thigh_angle, right_front_calf_angle, right_rear_thigh_angle, right_rear_calf_angle]
    poses = np.load(pose_data)

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)

    # Viewer 실행
    viewer = mujoco_viewer.MujocoViewer(model, data)
    # viewer._time_per_render = 1/100
    viewer._render_every_frame = False
    # viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_SHADOW] = False
    # viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_REFLECTION] = False
    # mjVIS_STATIC
    #mjVIS_SKIN
    first_flag = 1
    alive_flag = 0
    # 제어 루프
    while viewer.is_alive:
        # 관절 제어: 일정한 제어 입력을 설정
        if first_flag:
            # data.ctrl[j["RF_hip"]] = -0.03# * model.nu  # 모든 관절에 일정 토크 입력
            # data.ctrl[j["LF_hip"]] = 0.03 #* model.nu  # 모든 관절에 일정 토크 입력
            # data.ctrl[j["RR_hip"]] = -0.3 # * model.nu  # 모든 관절에 일정 토크 입력
            data.ctrl[j["RR_thigh"]] = 0 # * model.nu  # 모든 관절에 일정 토크 입력
            # data.ctrl[j["LR_hip"]] = 0.3 #* model.nu  # 모든 관절에 일정 토크 입력
            data.ctrl[j["LR_thigh"]] = 0 #* model.nu  # 모든 관절에 일정 토크 입력
            first_flag = 0
        # else:
            # data.ctrl[j["RF_thigh"]] = np.sin(data.time)
            # data.ctrl[j["RR_thigh"]] = np.sin(data.time)


        # 시뮬레이션 스텝
        mujoco.mj_step(model, data)
        alive_flag += 1
        # 렌더링
        viewer.render()


        print(f"{data.qpos[0], data.qpos[1], data.qpos[2], data.qpos[3], data.qpos[4], data.qpos[5]}")


    viewer.close()


if __name__ == "__main__":
    main()