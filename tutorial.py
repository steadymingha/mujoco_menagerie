import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import enum

# --- Helper Methods ---

class Resolution(enum.Enum):
    SD = (480, 640)
    HD = (720, 1280)
    UHD = (2160, 3840)

def unit_smooth(normalised_time: float) -> float:
    return 1 - np.cos(normalised_time * 2 * np.pi)

def azimuth(time: float, duration: float, total_rotation: float, offset: float) -> float:
    return offset + unit_smooth(time / duration) * total_rotation

# --- Parameters ---
res = Resolution.SD
fps = 60
duration = 100000.0
ctrl_rate = 2
ctrl_std = 0.05
total_rot = 60

# --- Loading Model ---
model_dir = Path("unitree_go1")
model_xml = model_dir / "scene.xml"

# 경로 확인
if not model_xml.exists():
    print(f"Error: Model file not found at {model_xml}")
    # 테스트를 위해 go2 경로가 있다면 사용 (없으면 에러)
    model_xml = Path("./unitree_go2/go2_mjx.xml")
    if not model_xml.exists():
        print("Please check your XML path.")
        exit(1)

model = mujoco.MjModel.from_xml_path(str(model_xml))
data = mujoco.MjData(model)

# --- Control Setup (Actuator Noise) ---
np.random.seed(12345)

nsteps = int(np.ceil(duration / model.opt.timestep))
perturb = np.random.randn(nsteps, model.nu)
width = int(nsteps * ctrl_rate / duration)
kernel = np.exp(-0.5 * np.linspace(-3, 3, width) ** 2)
kernel /= np.linalg.norm(kernel)

for i in range(model.nu):
    perturb[:, i] = np.convolve(perturb[:, i], kernel, mode="same")

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ctrl0 = data.ctrl.copy()
else:
    mujoco.mj_resetData(model, data)
    ctrl0 = np.mean(model.actuator_ctrlrange, axis=1)

# --- 발 Geom ID 찾기 ---
foot_names = ["FL", "FR", "RL", "RR"] 
foot_geom_ids = []
for name in foot_names:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid != -1:
        foot_geom_ids.append(gid)

# --- Real-time Visualization Loop ---

def main():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.vopt.geomgroup[3] = 1
        viewer.cam.distance = 1.0
        viewer.cam.lookat = [0, 0, 0.2]
        start_azimuth = viewer.cam.azimuth
        
        start_time = time.time()
        
        for i in range(nsteps):
            if not viewer.is_running():
                break

            step_start = time.time()

            # 1. 제어 입력 적용
            data.ctrl[:] = ctrl0 + ctrl_std * perturb[i]
            
            # 2. 시뮬레이션 스텝
            mujoco.mj_step(model, data)

            # --- [수정된 부분] 랜덤 값 생성 및 시각화 ---
            
            # (A) 랜덤 확률 값 생성
            random_prob = np.random.rand()

            # (B) 발 색깔 바꾸기 (확률 > 0.5 이면 빨강)
            if random_prob > 0.5:
                foot_color = [1.0, 0.0, 0.0, 1.0] # 빨강
            else:
                foot_color = [0.0, 0.0, 1.0, 1.0] # 파랑

            for gid in foot_geom_ids:
                model.geom_rgba[gid] = foot_color

            # (C) 3D 텍스트 라벨 추가 (add_overlay 대체)
            # viewer.user_scn을 사용하여 로봇 위에 글자를 띄웁니다.
            if viewer.user_scn:
                viewer.user_scn.ngeom = 0 # 이전 프레임의 유저 지오메트리 초기화
                
                # 라벨을 추가할 공간이 있는지 확인
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    
                    # 라벨 초기화 (위치: 로봇 위 0.5m)
                    text_pos = np.array([data.qpos[0], data.qpos[1], data.qpos[2] + 0.5])
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LABEL,
                        np.zeros(3),
                        text_pos,
                        np.zeros(9),
                        np.array([1, 1, 1, 1]) # 흰색 글자
                    )
                    # 텍스트 설정
                    geom.label = f"Prob: {random_prob:.4f}"
                    
                    viewer.user_scn.ngeom += 1

            # ------------------------------------------

            # 3. 카메라 업데이트
            current_sim_time = data.time
            if current_sim_time < duration:
                viewer.cam.azimuth = azimuth(current_sim_time, duration, total_rot, start_azimuth)

            # 4. 뷰어 동기화
            viewer.sync()

            # 5. FPS 제어
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()