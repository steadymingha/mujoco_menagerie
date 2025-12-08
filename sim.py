import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import enum
import contextlib
from mh.foot_mechanics import *
from mh.ground_contact import ContactModel

# --- Utils & Helpers ---
class Resolution(enum.Enum):
    SD = (480, 640)
    HD = (720, 1280)
    UHD = (2160, 3840)

class CameraController:
    """Helper class to calculate camera movement"""
    def __init__(self, duration: float, total_rotation: float):
        self.duration = duration
        self.total_rot = total_rotation
        self.start_azimuth = 0.0

    def unit_smooth(self, normalised_time: float) -> float:
        return 1 - np.cos(normalised_time * 2 * np.pi)

    def get_azimuth(self, current_time: float, offset: float) -> float:
        if current_time >= self.duration:
            return offset + self.total_rot
        return offset + self.unit_smooth(current_time / self.duration) * self.total_rot

# --- Simulator Wrapper (도구 상자 역할) ---
class Go2Sim:
    def __init__(self, model_path: str, dt: float = 0.002):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # 1. Load Model & Data
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        
        # 2. Camera & Render Internal State
        self.last_render_time = time.time()
        
        # Reset Data
        mujoco.mj_resetData(self.model, self.data)
        self.ctrl0 = np.mean(self.model.actuator_ctrlrange, axis=1) # 초기 자세

    @contextlib.contextmanager
    def launch_viewer(self):
        """
        뷰어를 실행하고 제어권을 main으로 넘겨주는 Context Manager
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 뷰어 초기 카메라 설정
            viewer.cam.distance = 1.0
            viewer.cam.lookat = [0, 0, 0.2]
            
            # main 함수의 with 구문 안으로 viewer 객체를 전달
            yield viewer 

    def add_text(self, viewer, contents):
        data = self.data
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
                    # geom.label = f"Prob: {contents:.4f}"
                    geom.label = contents
                    
                    viewer.user_scn.ngeom += 1

    def step(self, ctrl_input: np.ndarray = None):
        """물리 연산 1스텝 진행"""
        if ctrl_input is not None:
            self.data.ctrl[:] = ctrl_input
        else:
            self.data.ctrl[:] = self.ctrl0 # 입력 없으면 기본 자세 유지
            
        mujoco.mj_step(self.model, self.data)
    
    def sync(self, viewer):
        """
        화면 갱신 및 시뮬레이션 속도 조절 (Real-time Sync)
        """
        viewer.sync()
        
        # 시간 동기화 (FPS 조절)
        time_until_next_step = self.model.opt.timestep - (time.time() - self.last_render_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
        self.last_render_time = time.time()

SIMUL_TIME = 1 # sec

def main():
    model_file = './unitree_go2/scene_mjx.xml'#"unitree_go1/scene.xml" # 경로 확인 필요
    sim = Go2Sim(model_file)
    
    ff = FootForce(sim.model)
    fh = FootHeight()
    cm = ContactModel()
    # # 2. 발 접촉 확률을 저장할 변수 (예시)
    # foot_contact_probs = [0.0, 0.0, 0.0, 0.0] # FR, FL, RR, RL
    
    print("Simulation Loop Started on Main Thread...")
    
    # nsteps = int(np.ceil(SIMUL_TIME / model.opt.timestep))
    with sim.launch_viewer() as viewer:
        while viewer.is_running():
            
            # --- [A] 센서 데이터 수집 (Sensor Data Acquisition) ---
            # 예: 발 끝의 힘 센서나 가속도계 값 가져오기
            # foot_force = sim.data.sensordata[...] 
            
            # --- [B] 발 접촉 확률 계산 알고리즘 (Algorithm Area) ---
            fz = ff.get_foot_force(sim.data)
            pz = fh.get_foot_height(sim.data)
            p_foot_contact = cm.prob_contact(sim.data, pz, fz)
            contents = f'FL, FR : {p_foot_contact[0].item():.2f}, {p_foot_contact[1].item():.2f}'

            # 여기에 작성하신 알고리즘을 넣으시면 됩니다.
            # 지금은 예시로 랜덤값을 넣습니다.
            # for i in range(4):
            #     # (예시) z축 반력이나 위치를 기반으로 확률 계산
            #     foot_contact_probs[i] = np.random.uniform(0, 1) 
            
            # # 디버깅: 확률이 0.8 이상일 때만 출력해보기
            # if foot_contact_probs[0] > 0.9:
            #     print(f"FR Foot Contact Probability High: {foot_contact_probs[0]:.2f}")

            current_ctrl = sim.ctrl0 
            
            sim.step(current_ctrl)



            sim.add_text(viewer, contents)




            sim.sync(viewer)

if __name__ == "__main__":
    main()