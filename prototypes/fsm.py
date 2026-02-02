import time # 시뮬레이션을 위해 import

class EventBasedGaitFSM:
    """
    논문(그림 10)의 Event-Based FSM을 구현한 클래스입니다.
    매 제어 루프마다 update() 함수가 호출되어야 합니다.
    """
    
    def __init__(self, initial_state='Swing'):
        self.current_state = initial_state
        self.outputs = {} # (s_hat, s_phi) 출력을 저장할 딕셔너리
        
        # FSM 상태에 따른 출력(모터 제어 신호) 정의
        # [cite: 408] 다이어그램의 (s_hat, s_phi) 값
        self.state_outputs = {
            'Swing':   {'s_hat': 0, 's_phi': 0},
            'Late':    {'s_hat': 0, 's_phi': 1},
            'Early':   {'s_hat': 1, 's_phi': 0}, # [cite: 392, 393]
            'Contact': {'s_hat': 1, 's_phi': 1}  # [cite: 399, 402]
        }
        
        self.set_outputs(initial_state) # 초기 상태 출력 설정
        print(f"FSM Initialized. State: {self.current_state}")

    def set_outputs(self, state):
        """현재 FSM 상태에 맞는 (s_hat, s_phi) 값을 설정합니다."""
        self.outputs = self.state_outputs.get(state, {})
        print(f"ACTION: State={state}. Outputs set to {self.outputs}")

    def transition_to(self, next_state):
        """상태를 안전하게 전이시키고 새 상태의 출력을 설정합니다."""
        if self.current_state != next_state:
            print(f"\n!!! TRANSITION: {self.current_state} -> {next_state}")
            self.current_state = next_state
            self.set_outputs(self.current_state)
            return True # 전이 발생
        return False # 상태 유지

    def update(self, s_hat_sensor, t, t_c_bar, t_0):
        """
        [ 중요 ] 매 제어 루프(cycle)마다 이 함수를 호출해야 합니다.
        
        Args:
            s_hat_sensor (int): 실제 접촉 센서의 값 (0 or 1). 
                                논문의 'estimated contact state' [cite: 413]
            t (float): 현재 시간
            t_c_bar (float): 스케줄 임계 시간 1
            t_0 (float): 스케줄 임계 시간 2
        """
        print(f"--- UPDATE: State={self.current_state}, t={t:.2f}, sensor_s_hat={s_hat_sensor}")

        # 다이어그램의 시간 조건들을 명확하게 변수로 정의 [cite: 408]
        time_cond_ContactEarly = (t >= t_c_bar + t_0)
        time_cond_SwingLate = (t > t_c_bar + t_0)
        time_cond_SwingContact = (t == t_c_bar + t_0)

        # --- FSM 전이 로직 ---
        # (현재 상태)
        if self.current_state == 'Swing':
            # [우선순위 1] '이벤트' 기반 전이 (센서 감지) [cite: 408]
            if s_hat_sensor == 1:
                self.transition_to('Early')
            # [우선순위 2] '시간' 기반 전이 (너무 늦음) [cite: 408]
            elif time_cond_SwingLate:
                self.transition_to('Late')
            # [우선순위 3] '시간' 기반 전이 (정확한 타이밍) [cite: 408]
            elif time_cond_SwingContact:
                # !!! 현실 경고 !!!
                # 실제 로봇 코드에서 float 시간(t)이 특정 값과
                # '==' 비교로 일치할 확률은 0에 가깝습니다.
                # 보통 (t > T - 0.01) and (t < T + 0.01) 처럼
                # '구간'으로 처리해야 합니다.
                print("INFO: 't == t_c_bar + t_0'의 기적적인 일치!")
                self.transition_to('Contact')
            # (else: 모든 조건 불만족 시 'Swing' 상태 유지)

        elif self.current_state == 'Late':
            # 'Late' 상태는 오직 센서 이벤트만 기다림 [cite: 408]
            if s_hat_sensor == 1:
                self.transition_to('Contact')

        elif self.current_state == 'Early':
            # 'Early' 상태는 디바운싱 타이머(시간 조건)를 기다림 [cite: 408, 417]
            if time_cond_ContactEarly:
                self.transition_to('Contact')

        elif self.current_state == 'Contact':
            # 'Contact' 상태는 발을 뗄 시간(시간 조건)을 기다림 [cite: 408]
            if time_cond_ContactEarly:
                self.transition_to('Swing')
        
        return self.current_state, self.outputs

# --- FSM 시뮬레이션 테스트 ---

# 1. FSM 인스턴스 생성
fsm = EventBasedGaitFSM(initial_state='Swing')

# 2. 제어 파라미터 (임의의 값)
T_C_BAR = 1.0
T_0 = 0.5
TIME_LIMIT = T_C_BAR + T_0 # 1.5

# 3. 시뮬레이션 변수
sensor_s_hat = 0 # 0: 접촉 없음, 1: 접촉 감지
current_time = 0.0
dt = 0.1 # 0.1초마다 update() 호출

print("\n\n--- [시뮬레이션 1: 예상보다 '일찍' 닿는 경우] ---")
sensor_s_hat = 0
current_time = 0.0
while current_time < 2.5:
    if current_time >= 0.8: # 0.8초에 'Early' 접촉 발생!
        sensor_s_hat = 1
        
    fsm.update(sensor_s_hat, current_time, T_C_BAR, T_0)
    current_time += dt
    # 결과 예측: 
    # t=0.8 'Swing' -> 'Early' (센서 감지)
    # t=1.5 'Early' -> 'Contact' (시간 조건 만족)
    # t=1.5 'Contact' -> 'Swing' (동일한 시간 조건)
    # t=1.6 'Swing' -> 'Early' (센서가 계속 1이므로)
    # ... (Early와 Contact/Swing을 빠르게 반복할 수 있음 -> 실제론 이래서 로직이 더 복잡함)


print("\n\n--- [시뮬레이션 2: 예상보다 '늦게' 닿는 경우] ---")
fsm = EventBasedGaitFSM(initial_state='Swing') # FSM 리셋
sensor_s_hat = 0
current_time = 0.0
while current_time < 2.5:
    if current_time >= 1.8: # 1.8초에 'Late' 접촉 발생! (TIME_LIMIT 1.5초보다 늦음)
        sensor_s_hat = 1

    fsm.update(sensor_s_hat, current_time, T_C_BAR, T_0)
    current_time += dt
    # 결과 예측:
    # t=1.6 'Swing' -> 'Late' (시간 t > 1.5 만족)
    # t=1.8 'Late' -> 'Contact' (센서 감지)
    # t=1.8 'Contact' -> 'Swing' (시간 t >= 1.5 만족)