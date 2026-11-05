"""debug only: QuadrupedContactFSM 사용 예제.

ContactModel 의 phase(phi) + 접촉 추정(s_hat) -> FSM -> QP 에 넘길 stance 벡터.
main.py 에 넣을 부분은 아래 make_fsm() / fsm_step() 두 개뿐이고,
나머지는 동작을 눈으로 보기 위한 테스트 코드다.

실행: 프로젝트 루트에서  uv run python dbug/fsm_demo.py
"""
import sys
import numpy as np
sys.path.insert(0, ".")
from core.ground_contact import ContactModel
from core.fsm import QuadrupedContactFSM

LEGS = ["FL", "FR", "RL", "RR"]


# ── main.py 에 가져다 쓸 부분 ──────────────────────────────────────────────

def make_fsm(T, threshold, dt):
    """gait 주기에 맞춘 FSM 생성.

    t_c_bar = swing 이 끝나는 시각, t_c = stance 가 끝나는 시각.
    둘 다 'swing 시작' 기준이다. t_c 를 한 스텝 당기지 않으면
    다리 시계가 0 으로 되감기기 전에 Contact->Swing 이 걸리지 않는다.
    """
    return QuadrupedContactFSM(t_c_bar=(1.0 - threshold) * T, t_c=T - dt, t_0=0.0)


def fsm_step(fsm, s_vec, phi_vec, T, threshold):
    """다리별로 FSM 을 갱신하고 QP 에 넘길 stance 벡터를 돌려준다.

    s_vec, phi_vec : (4,) [FL, FR, RL, RR] 순
    반환           : (4,) s_phi, 1 = stance

    다리마다 phase 시간이 다르므로 QuadrupedContactFSM.update() 는 쓰지 않는다
    (그쪽은 네 다리에 같은 t 를 넘긴다).
    """
    for i, name in enumerate(LEGS):
        t_leg = ((phi_vec[i] - threshold) % 1.0) * T   # swing 시작 기준 경과 시간
        fsm.legs[name].update(int(s_vec[i]), t_leg)
    return np.array([fsm.legs[name].s_phi for name in LEGS])


# ── 여기부터는 데모용 ──────────────────────────────────────────────────────

def fake_s_hat(phi_vec, t, threshold):
    """접촉 추정값 대신 쓰는 가짜 센서.
    기본은 스케줄과 동일, 단 FL 은 일찍 닿고(Early) RR 은 늦게 닿는다(Late)."""
    s = np.where(phi_vec < threshold, 1, 0)
    if 2.0 < t < 4.0 and phi_vec[LEGS.index("FL")] > threshold + 0.15:
        s[LEGS.index("FL")] = 1      # swing 후반에 미리 착지
    if 4.0 < t < 6.0 and phi_vec[LEGS.index("RR")] < 0.15:
        s[LEGS.index("RR")] = 0      # stance 시작인데 아직 못 닿음
    return s


def main():
    cm = ContactModel()
    T, TH, dt = cm.T, cm.THRESHOLD, 0.002
    fsm = make_fsm(T, TH, dt)

    print(f"T={T}  THRESHOLD={TH}  swing={(1 - TH) * T:.3f}s")
    print(f"{'t':>6} {'leg':>4} {'phi':>5} {'s_hat':>5}  {'state':<8} {'s_phi':>5}")
    print("-" * 42)

    prev = dict.fromkeys(LEGS)
    for k in range(int(7.0 / dt)):
        t = k * dt
        phi_vec, _ = cm.get_current_phase(t)
        s_vec = fake_s_hat(phi_vec, t, TH)

        contact = fsm_step(fsm, s_vec, phi_vec, T, TH)

        for i, name in enumerate(LEGS):   # 상태가 바뀐 순간만 출력
            state = fsm.legs[name].state.name
            if state != prev[name]:
                print(f"{t:6.3f} {name:>4} {phi_vec[i]:5.2f} {s_vec[i]:>5}  "
                      f"{state:<8} {contact[i]:>5}")
                prev[name] = state

    print(f"\ncontact vector for QP (FL,FR,RL,RR) = {contact}")


if __name__ == "__main__":
    main()
