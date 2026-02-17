"""
MIT Cheetah 3 - Contact State Finite State Machine
Based on "Contact Model Fusion for Event-Based Locomotion in Unstructured Terrains"

4-state FSM for contact estimation per leg:
  Early   (ŝ=1, s_φ=0) : 접지 감지됐지만 phase는 아직 swing
  Contact (ŝ=1, s_φ=1) : 접지 확인, phase도 contact
  Late    (ŝ=0, s_φ=1) : 접지 해제됐지만 phase는 아직 contact
  Swing   (ŝ=0, s_φ=0) : 완전한 swing phase

Transition conditions:
  Swing   → Early   : ŝ = 1
  Swing   → Late    : t > t̄_c + t_0
  Swing   → Contact : ŝ = 1 and t = t̄_c + t_0
  Early   → Contact : t >= t̄_c + t_0
  Late    → Contact : ŝ = 1
  Contact → Swing   : t >= t_c + t_0
"""

from enum import Enum, auto
from dataclasses import dataclass


class ContactState(Enum):
    SWING = auto()
    EARLY = auto()
    CONTACT = auto()
    LATE = auto()


# 각 상태의 출력값 (ŝ, s_φ)
STATE_OUTPUT = {
    ContactState.EARLY:   (1, 0),
    ContactState.CONTACT: (1, 1),
    ContactState.LATE:    (0, 1),
    ContactState.SWING:   (0, 0),
}


@dataclass
class ContactFSM:
    """단일 다리의 contact state FSM"""

    state: ContactState = ContactState.SWING
    t_c_bar: float = 0.0  # expected swing duration (swing phase threshold)
    t_c: float = 0.0      # expected stance duration (stance phase threshold)
    t_0: float = 0.0      # time offset

    def update(self, s_hat: int, t: float) -> ContactState:
        """
        매 제어 루프마다 호출.

        Args:
            s_hat: contact estimate (0 or 1) from observer
            t: current phase time
        Returns:
            updated ContactState
        """
        t_swing = self.t_c_bar + self.t_0   # swing phase threshold
        t_stance = self.t_c + self.t_0       # stance phase threshold

        if self.state == ContactState.SWING:
            if s_hat == 1 and t == t_swing:
                # 접지 감지 + swing 시간 정확히 일치 → Contact 직행
                self.state = ContactState.CONTACT
            elif s_hat == 1:
                # 접지 감지 → Early
                self.state = ContactState.EARLY
            elif t > t_swing:
                # 접지 미감지 + swing 시간 초과 → Late
                self.state = ContactState.LATE

        elif self.state == ContactState.EARLY:
            if t >= t_swing:
                # swing 시간 충족 → Contact 승격
                self.state = ContactState.CONTACT

        elif self.state == ContactState.CONTACT:
            if t >= t_stance:
                # stance 시간 초과 → Swing
                self.state = ContactState.SWING

        elif self.state == ContactState.LATE:
            if s_hat == 1:
                # 재접지 감지 → Contact
                self.state = ContactState.CONTACT

        return self.state

    @property
    def s_hat_output(self) -> int:
        """현재 상태의 contact estimate 출력"""
        return STATE_OUTPUT[self.state][0]

    @property
    def s_phi(self) -> int:
        """현재 상태의 phase signal 출력"""
        return STATE_OUTPUT[self.state][1]

    def reset(self):
        self.state = ContactState.SWING


class QuadrupedContactFSM:
    """4족 로봇 전체 다리의 contact FSM 관리"""

    LEG_NAMES = ["FR", "FL", "RR", "RL"]

    def __init__(self, t_c_bar: float = 0.0, t_c: float = 0.0, t_0: float = 0.0):
        self.legs = {
            name: ContactFSM(t_c_bar=t_c_bar, t_c=t_c, t_0=t_0)
            for name in self.LEG_NAMES
        }

    def update(self, s_hats: dict[str, int], t: float) -> dict[str, ContactState]:
        """
        전체 다리 FSM 업데이트.

        Args:
            s_hats: {"FR": 0, "FL": 1, "RR": 1, "RL": 0}
            t: current phase time
        Returns:
            각 다리의 현재 상태
        """
        return {
            name: fsm.update(s_hats[name], t)
            for name, fsm in self.legs.items()
        }

    def get_contact_pattern(self) -> dict[str, tuple[int, int]]:
        """각 다리의 (ŝ, s_φ) 출력"""
        return {
            name: (fsm.s_hat_output, fsm.s_phi)
            for name, fsm in self.legs.items()
        }

    def set_gait_params(self, t_c_bar: float, t_c: float, t_0: float):
        """gait 파라미터 일괄 업데이트"""
        for fsm in self.legs.values():
            fsm.t_c_bar = t_c_bar
            fsm.t_c = t_c
            fsm.t_0 = t_0

    def reset(self):
        for fsm in self.legs.values():
            fsm.reset()


# ── 검증 테스트 ──
if __name__ == "__main__":

    def run_test(name, fsm, sequence):
        print(f"\n=== {name} ===")
        print(f"t_c_bar={fsm.t_c_bar}, t_c={fsm.t_c}, t_0={fsm.t_0}")
        print(f"  t_swing  = t_c_bar + t_0 = {fsm.t_c_bar + fsm.t_0}")
        print(f"  t_stance = t_c + t_0     = {fsm.t_c + fsm.t_0}")
        print(f"{'time':>6} {'s_hat':>5} {'state':<10} {'ŝ_out':>5} {'s_φ':>3}  transition")
        print("-" * 55)
        for t, s_hat, expected_state in sequence:
            prev = fsm.state.name
            state = fsm.update(s_hat, t)
            arrow = f"{prev} → {state.name}" if prev != state.name else f"{state.name} (유지)"
            print(f"{t:6.2f} {s_hat:>5} {state.name:<10} {fsm.s_hat_output:>5} {fsm.s_phi:>3}  {arrow}")
            assert state.name == expected_state, \
                f"FAIL: expected {expected_state}, got {state.name}"
        print("✓ PASSED")

    # Test 1: 정상 순환 Swing → Early → Contact → Swing
    run_test(
        "Test 1: Swing → Early → Contact → Swing",
        ContactFSM(t_c_bar=0.2, t_c=0.3, t_0=0.05),
        [   # (time, s_hat, expected_state)
            (0.00, 0, "SWING"),     # Swing 유지
            (0.10, 1, "EARLY"),     # ŝ=1 → Early
            (0.20, 1, "EARLY"),     # t < 0.25, Early 유지
            (0.25, 1, "CONTACT"),   # t >= t_swing(0.25) → Contact
            (0.30, 1, "CONTACT"),   # Contact 유지
            (0.35, 1, "SWING"),     # t >= t_stance(0.35) → Swing
        ]
    )

    # Test 2: Swing → Contact 직행 (s_hat=1, t == t_swing)
    run_test(
        "Test 2: Swing → Contact (direct)",
        ContactFSM(t_c_bar=0.2, t_c=0.3, t_0=0.05),
        [
            (0.00, 0, "SWING"),
            (0.25, 1, "CONTACT"),   # ŝ=1, t == t_swing → Contact 직행
            (0.35, 1, "SWING"),     # t >= t_stance → Swing
        ]
    )

    # Test 3: Swing → Late → Contact → Swing
    run_test(
        "Test 3: Swing → Late → Contact → Swing",
        ContactFSM(t_c_bar=0.1, t_c=0.3, t_0=0.05),
        [
            (0.00, 0, "SWING"),
            (0.10, 0, "SWING"),     # t < 0.15, Swing 유지
            (0.16, 0, "LATE"),      # t > t_swing(0.15), ŝ=0 → Late
            (0.20, 0, "LATE"),      # Late 유지
            (0.25, 1, "CONTACT"),   # ŝ=1 → Contact
            (0.35, 1, "SWING"),     # t >= t_stance → Swing
        ]
    )

    # Test 4: Late에서 접지 안되면 Late 유지
    run_test(
        "Test 4: Late stays without contact",
        ContactFSM(t_c_bar=0.1, t_c=0.5, t_0=0.05),
        [
            (0.00, 0, "SWING"),
            (0.16, 0, "LATE"),      # → Late
            (0.20, 0, "LATE"),      # ŝ=0, Late 유지
            (0.30, 0, "LATE"),      # Late 유지
            (0.40, 1, "CONTACT"),   # 드디어 접지 → Contact
            (0.55, 1, "SWING"),     # t >= t_stance(0.55) → Swing
        ]
    )

    # Test 5: Early에서 시간 지나면 바로 Contact (s_hat 무관)
    run_test(
        "Test 5: Early → Contact regardless of s_hat",
        ContactFSM(t_c_bar=0.2, t_c=0.5, t_0=0.05),
        [
            (0.00, 0, "SWING"),
            (0.10, 1, "EARLY"),     # → Early
            (0.25, 0, "CONTACT"),   # t >= t_swing → Contact (s_hat 무관)
        ]
    )

    print("\n✅ All tests passed!")