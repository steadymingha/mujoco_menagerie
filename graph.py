import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class FootContactPlotter:
    """
    실시간으로 4개의 다리 접촉(Contact) 데이터를 그리는 클래스
    """
    def __init__(self, max_len=50, draw_interval=1):
        self.max_len = max_len
        self.draw_interval = draw_interval # 몇 번 스텝마다 화면을 갱신할지 (속도 조절용)
        self.step_count = 0
        
        # 1. 데이터를 저장할 큐(Queue) 생성 (오래된 데이터 자동 삭제)
        self.queues = [deque([0]*max_len, maxlen=max_len) for _ in range(4)]
        
        # 2. 그래프 초기 설정
        plt.ion() # Interactive Mode On
        self.fig, self.axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        
        self.lines = []
        colors = ['red', 'blue', 'green', 'orange']
        titles = ['Front Left (FL)', 'Front Right (FR)', 'Rear Left (RL)', 'Rear Right (RR)']
        
        for i in range(4):
            # 초기 빈 라인 생성
            line, = self.axes[i].plot(np.arange(max_len), self.queues[i], 
                                      color=colors[i], lw=2)
            self.lines.append(line)
            
            # 스타일 설정
            self.axes[i].set_ylabel(titles[i], fontsize=10)
            self.axes[i].set_ylim(-0.2, 1.2) # 0과 1이 잘 보이도록 여유 둠
            self.axes[i].grid(True, alpha=0.5)
            
        self.axes[0].set_title("Real-time Foot Contact Sensor", fontsize=14, pad=10)
        self.axes[3].set_xlabel("Time Window", fontsize=12)
        plt.tight_layout()

    def update(self, p_foot_contact):
        """
        외부에서 4x1 np.array 데이터를 받아 그래프를 갱신함
        """
        # 1. 데이터 업데이트 (큐에 추가)
        flat_data = p_foot_contact.flatten() # (4, 1) -> (4,)
        for i in range(4):
            self.queues[i].append(flat_data[i])

        # 2. 화면 그리기 (성능을 위해 draw_interval 마다 한 번씩만 수행)
        self.step_count += 1
        if self.step_count % self.draw_interval == 0:
            for i in range(4):
                self.lines[i].set_ydata(self.queues[i]) # 데이터 교체
            
            plt.pause(0.001) # 화면 갱신 트리거

    def close(self):
        """종료 처리"""
        plt.ioff()
        plt.show()
        print("그래프 종료.")

# ---------------------------------------------------------
# [Main 함수] - 실제 사용 예시
# ---------------------------------------------------------
def main():
    print("시뮬레이션을 시작합니다. (중단하려면 Ctrl+C)")
    
    # 1. 플로터 객체 생성
    # draw_interval=5 -> 데이터는 매번 넣지만, 그림은 5번에 1번만 그림 (속도 향상)
    plotter = FootContactPlotter(max_len=100, draw_interval=5)
    
    try:
        # 가상의 시뮬레이션 루프 (예: 500 스텝)
        for i in range(500):
            
            # --- [데이터 생성] ---
            # (4, 1) 형태의 랜덤 데이터 (0 또는 1)
            # 실제 로봇 코드에서는 센서 값을 여기에 넣으면 됩니다.
            p_foot_contact = np.random.randint(0, 2, size=(4, 1))
            
            # --- [그래프 업데이트] ---
            # 그냥 이 한 줄만 부르면 알아서 그려줍니다.
            plotter.update(p_foot_contact)
            
            # (시뮬레이션 속도 제어용 - 실제 코드엔 불필요할 수 있음)
            import time
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    finally:
        # 안전하게 종료
        plotter.close()

if __name__ == "__main__":
    main()