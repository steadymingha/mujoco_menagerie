from mujoco import MjModel, MjData
import mujoco.viewer

xml_name = 'unitree_go2/scene_mjx.xml'
model = MjModel.from_xml_path(xml_name)
data = MjData(model)

sensor_name = "gyro"
sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
dim = model.sensor_dim[sensor_id]
addr = model.sensor_adr[sensor_id]

sensor_values = data.sensordata[addr : addr + dim]
print(f"{sensor_name} =", sensor_values)

sensor_info = []

for i in range(model.nsensor):
    adr = model.name_sensoradr[i]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    start = model.sensor_adr[i]
    dim = model.sensor_dim[i]
    sensor_info.append((name, start, dim))

for _ in range(10):
    mujoco.mj_step(model, data)
    for name, start, dim in sensor_info:
        value = data.sensordata[start:start+dim]
        print(f"{name}: {value}")
    print("----")

print(model.opt.timestep)


print("==================INERTIA & MASS=====================")
# 각 body별 질량
body_masses = model.body_mass  # 모든 body의 질량 배열
print(f"Body masses: {body_masses}")

# 전체 질량
total_mass = np.sum(model.body_mass)
print(f"Total mass: {total_mass} kg")

# 각 body별 관성 텐서 (3x3 matrix for each body)
body_inertias = model.body_inertia  # shape: (nbody, 3, 3)
print(f"Body inertias shape: {body_inertias.shape}")

# 특정 body의 관성 (예: base body - index 1)
base_inertia = model.body_inertia[1]  # 0은 world
print(f"Base inertia:\n{base_inertia}")

# Body 이름으로 인덱스 찾기
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)]
print(f"Body names: {body_names}")

# 특정 body ID 찾기
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
base_mass = model.body_mass[base_id]

##
# joint_pos = data.sensordata[0:12]  # abduction, hip, knee joints (FL, RL, FR, 
#   RR)

#   # 관절 속도 (12개)  
#   joint_vel = data.sensordata[12:24]

#   # IMU 센서
#   gyro = data.sensordata[24:27]           # 자이로스코프
#   accel = data.sensordata[27:30]          # 가속도계
#   orientation = data.sensordata[30:34]    # 쿼터니언 자세
#   position = data.sensordata[34:37]       # 글로벌 위치
#   linvel = data.sensordata[37:40]         # 선속도
#   angvel = data.sensordata[40:43]         # 각속도

#   # 또는 개별 관절 상태
#   qpos = data.qpos[7:19]  # 관절 위치 (freejoint 제외)
#   qvel = data.qvel[6:18]  # 관절 속도 (freejoint 제외)

#   토크 명령

#   # 토크 제어 (-24 ~ 24 Nm)
#   data.ctrl[0:12] = desired_torques  # 12개 관절 토크

#   # 관절 순서: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#   #           RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf

#   # 예시
#   data.ctrl[:] = [0, 0.9, -1.8, 0, 0.9, -1.8,  # Front legs
#                   0, 0.9, -1.8, 0, 0.9, -1.8]  # Rear legs
##


# if __name__ == "__main__":
#     main()