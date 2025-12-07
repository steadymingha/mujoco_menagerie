# original : tutorial.py(mujoco.viewer), mujoco_viewer : unofficial viewer
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
duration = 10.0
ctrl_rate = 2
ctrl_std = 0.05
total_rot = 60
# blend_std는 실시간 뷰어에서는 픽셀 블렌딩을 하지 않으므로 제외했습니다.

# --- Loading Model ---
model_dir = Path("unitree_go1")
model_xml = model_dir / "scene.xml"

if not model_xml.exists():
    print(f"Error: Model file not found at {model_xml}")
    print("Please make sure the 'unitree_go1' directory is in the current folder.")
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

# 초기 컨트롤 값 설정
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ctrl0 = data.ctrl.copy()
else:
    mujoco.mj_resetData(model, data)
    ctrl0 = np.mean(model.actuator_ctrlrange, axis=1)

# --- Real-time Visualization Loop ---

def main():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # 뷰어 초기 설정
        viewer.cam.distance = 1.0  # 카메라 거리
        viewer.cam.lookat = [0, 0, 0.2] # 로봇을 바라보도록 조정
        start_azimuth = viewer.cam.azimuth
        
        # 렌더링 옵션 (Collision을 보고 싶다면 아래 주석을 조정하세요)
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
        
        start_time = time.time()
        
        for i in range(nsteps):
            # 1. 뷰어가 닫혔으면 종료
            if not viewer.is_running():
                break

            step_start = time.time()

            # 2. 제어 입력 적용
            data.ctrl[:] = ctrl0 + ctrl_std * perturb[i]
            
            # 3. 시뮬레이션 스텝
            mujoco.mj_step(model, data)

            # 4. 카메라 업데이트 (원래 코드의 azimuth 로직 적용)
            current_sim_time = data.time
            if current_sim_time < duration:
                # 카메라 회전 적용
                viewer.cam.azimuth = azimuth(current_sim_time, duration, total_rot, start_azimuth)

            # 5. 뷰어 동기화 (화면 갱신)
            viewer.sync()

            # 6. Real-time 속도 맞추기 (FPS 제어)
            # 시뮬레이션이 너무 빠르면 잠시 대기
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()

# model_dir = Path("unitree_go1")
# model_xml = model_dir / "scene.xml"
# model = mujoco.MjModel.from_xml_path(str(model_xml))
# data = mujoco.MjData(model)

# # 간단히 실행 (가장 안정적)
# if __name__ == "__main__":
#     print("MuJoCo Viewer를 실행합니다. (마우스로 로봇을 드래그해보세요)")
#     mujoco.viewer.launch(model, data)