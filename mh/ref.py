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

# Hip abduction link (엉덩관절-허벅지 연결부): FL_thigh body의 pos를 보면 0 0.0955 0 으로, 길이는 0.0955m 입니다.
#    1   <body name="FL_hip" pos="0.1934 0.0465 0">
#    2     ...
#    3     <body name="FL_thigh" pos="0 0.0955 0">

#    - Thigh link (허벅지): FL_calf body의 pos를 보면 0 0 -0.213 으로, 길이는 0.213m 입니다.

#    1   <body name="FL_thigh" pos="0 0.0955 0">
#    2     ...
#    3     <body name="FL_calf" pos="0 0 -0.213">

#    - Calf link (종아리): FL_calf body 내의 발(foot) 사이트 pos를 보면 ... 0 0 -0.213 으로, 길이는 0.213m 입니다.

#    1   <body name="FL_calf" pos="0 0 -0.213">
#    2     ...
#    3     <site name="FL_foot" class="go2foot"/>
#    4   ...
#    5   <default class="go2foot">
#    6     <site group="1" pos="-0.002 0 -0.213"/>
#    7     ...
#    8   </default>

#   요약하면 각 다리의 링크 길이는 다음과 같습니다.
#    * Hip abduction: 0.0955m
#    * Thigh: 0.213m
#    * Calf: 0.213m
#      1. 몸통의 기준점 (0, 0, 0)

#   제가 '몸통 정중앙'이라고 말씀드린 것은 base 라는 이름의 몸통 body가 가지는 자체적인 로컬 좌표계의 원점(0,0,0)을 의미합니다. 모든 자식 body(예: 
#   다리)의 위치는 이 base body의 로컬 원점을 기준으로 계산됩니다.

#   하지만 시뮬레이션이 시작될 때, 이 base body 자체는 월드(world) 좌표계의 pos="0 0 0.445" 위치에 놓이게 됩니다. 즉, 로봇 몸통의 기준점은 
#   지면에서 44.5cm 위에 떠 있는 상태로 시작합니다.

#   2. 좌표계 방향

#   이 모델의 좌표계 방향은 로봇의 다리 위치를 보면 명확히 알 수 있습니다.

#    * FL_hip (앞-왼쪽): pos="0.1934 0.0465 0"
#    * FR_hip (앞-오른쪽): pos="0.1934 -0.0465 0"
#    * RL_hip (뒤-왼쪽): pos="-0.1934 0.0465 0"
#    * RR_hip (뒤-오른쪽): pos="-0.1934 -0.0465 0"

#   이것을 바탕으로 방향을 정리하면 다음과 같습니다.

#    * +X 방향: 앞쪽 (Forward) (앞다리들의 x값이 양수)
#    * -X 방향: 뒤쪽 (Backward) (뒷다리들의 x값이 음수)
#    * +Y 방향: 왼쪽 (Left) (왼쪽 다리들의 y값이 양수)
#    * -Y 방향: 오른쪽 (Right) (오른쪽 다리들의 y값이 음수)
#    * +Z 방향: 위쪽 (Up)