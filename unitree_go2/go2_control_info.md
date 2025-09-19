# Unitree Go2 (go2_mjx.xml) 보행 제어 정보

이 문서는 `unitree_go2/go2_mjx.xml` 모델을 사용하여 보행 제어기를 구현하는 데 필요한 주요 정보를 요약합니다.

## 1. 좌표계 (Coordinate System)

- **World Frame**: 시뮬레이션의 글로벌 좌표계입니다. 중력은 Z축의 음수 방향으로 작용합니다.
- **Body Frame**: 로봇의 `base` 링크에 부착된 로컬 좌표계입니다.
  - **원점**: `base` 링크의 기하학적 중심에 위치합니다. 모델 초기 로딩 시 `(0, 0, 0.445)` 위치에 스폰됩니다.
  - **IMU 센서 위치**: `base` 링크의 Body Frame 기준으로 `(-0.02557, 0, 0.04232)`에 위치합니다.

## 2. 조인트 (Joints)

총 12개의 회전 조인트가 있으며, 각 다리당 3개씩 (Abduction/Hip/Knee) 구성됩니다. 모든 각도는 라디안(radian) 단위입니다.

| Joint Name         | Leg          | Type      | Axis    | Range (radians)         |
| ------------------ | ------------ | --------- | ------- | ----------------------- |
| `FL_hip_joint`     | Front Left   | Abduction | `1 0 0` | `[-1.0472, 1.0472]`     |
| `FL_thigh_joint`   | Front Left   | Hip       | `0 1 0` | `[-1.5708, 3.4907]`     |
| `FL_calf_joint`    | Front Left   | Knee      | `0 1 0` | `[-2.7227, -0.83776]`    |
| `FR_hip_joint`     | Front Right  | Abduction | `1 0 0` | `[-1.0472, 1.0472]`     |
| `FR_thigh_joint`   | Front Right  | Hip       | `0 1 0` | `[-1.5708, 3.4907]`     |
| `FR_calf_joint`    | Front Right  | Knee      | `0 1 0` | `[-2.7227, -0.83776]`    |
| `RL_hip_joint`     | Rear Left    | Abduction | `1 0 0` | `[-1.0472, 1.0472]`     |
| `RL_thigh_joint`   | Rear Left    | Hip       | `0 1 0` | `[-1.5708, 3.4907]`     |
| `RL_calf_joint`    | Rear Left    | Knee      | `0 1 0` | `[-2.7227, -0.83776]`    |
| `RR_hip_joint`     | Rear Right   | Abduction | `1 0 0` | `[-1.0472, 1.0472]`     |
| `RR_thigh_joint`   | Rear Right   | Hip       | `0 1 0` | `[-1.5708, 3.4907]`     |
| `RR_calf_joint`    | Rear Right   | Knee      | `0 1 0` | `[-2.7227, -0.83776]`    |

## 3. 구동기 (Actuators)

12개의 `general` 타입 구동기가 각 조인트를 제어합니다.

| Actuator Name | Controlled Joint   | Control Range           | Force Range |
| ------------- | ------------------ | ----------------------- | ----------- |
| `FL_hip`      | `FL_hip_joint`     | `[-0.9472, 0.9472]`     | `[-24, 24]` |
| `FL_thigh`    | `FL_thigh_joint`   | `[-1.4, 2.5]`           | `[-24, 24]` |
| `FL_calf`     | `FL_calf_joint`    | `[-2.6227, -0.84776]`   | `[-24, 24]` |
| `FR_hip`      | `FR_hip_joint`     | `[-0.9472, 0.9472]`     | `[-24, 24]` |
| `FR_thigh`    | `FR_thigh_joint`   | `[-1.4, 2.5]`           | `[-24, 24]` |
| `FR_calf`     | `FR_calf_joint`    | `[-2.6227, -0.84776]`   | `[-24, 24]` |
| `RL_hip`      | `RL_hip_joint`     | `[-0.9472, 0.9472]`     | `[-24, 24]` |
| `RL_thigh`    | `RL_thigh_joint`   | `[-1.4, 2.5]`           | `[-24, 24]` |
| `RL_calf`     | `RL_calf_joint`    | `[-2.6227, -0.84776]`   | `[-24, 24]` |
| `RR_hip`      | `RR_hip_joint`     | `[-0.9472, 0.9472]`     | `[-24, 24]` |
| `RR_thigh`    | `RR_thigh_joint`   | `[-1.4, 2.5]`           | `[-24, 24]` |
| `RR_calf`     | `RR_calf_joint`    | `[-2.6227, -0.84776]`   | `[-24, 24]` |

## 4. 센서 (Sensors)

제어에 사용할 수 있는 센서 목록입니다.

### Body State Sensors

`imu` 사이트(`base` 링크에 위치)에서 측정된 값입니다.

| Sensor Name       | Type            | Description                               |
| ----------------- | --------------- | ----------------------------------------- |
| `gyro`            | `gyro`          | 각속도 (rad/s)                            |
| `accelerometer`   | `accelerometer` | 선형 가속도 (m/s^2)                       |
| `orientation`     | `framequat`     | IMU의 월드 좌표계 기준 방향 (Quaternion)    |
| `global_position` | `framepos`      | IMU의 월드 좌표계 기준 위치 (m)           |
| `global_linvel`   | `framelinvel`   | IMU의 월드 좌표계 기준 선형 속도 (m/s)    |
| `global_angvel`   | `frameangvel`   | IMU의 월드 좌표계 기준 각속도 (rad/s)     |

### Joint Sensors

각 조인트의 위치와 속도를 측정합니다.

- **Position Sensors (`jointpos`)**:
  - `abduction_front_left_pos`, `hip_front_left_pos`, `knee_front_left_pos`
  - `abduction_front_right_pos`, `hip_front_right_pos`, `knee_front_right_pos`
  - `abduction_hind_left_pos`, `hip_hind_left_pos`, `knee_hind_left_pos`
  - `abduction_hind_right_pos`, `hip_hind_right_pos`, `knee_hind_right_pos`

- **Velocity Sensors (`jointvel`)**:
  - `abduction_front_left_vel`, `hip_front_left_vel`, `knee_front_left_vel`
  - `abduction_front_right_vel`, `hip_front_right_vel`, `knee_front_right_vel`
  - `abduction_hind_left_vel`, `hip_hind_left_vel`, `knee_hind_left_vel`
  - `abduction_hind_right_vel`, `hip_hind_right_vel`, `knee_hind_right_vel`

## 5. 발 끝 (Foot Contact Points)

- **Contact Geometry**: `FL`, `FR`, `RL`, `RR` 이름의 `geom`이 충돌 감지를 위해 각 다리 끝에 정의되어 있습니다.
- **Foot Sites**: `FL_foot`, `FR_foot`, `RL_foot`, `RR_foot` 이름의 `site`가 있으며, `calf` 링크 기준으로 `(-0.002, 0, -0.213)` 위치에 있습니다. 이 `site`를 이용해 발 끝의 위치나 속도를 추적할 수 있습니다.
