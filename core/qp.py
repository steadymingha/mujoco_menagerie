"""
Balance Controller QP Solver for Cheetah3
=========================================
- 클래스 기반: solver 인스턴스 재사용 + warm start
- 접촉 상태 변경 시에만 재setup
- friction pyramid + unilateral constraint 포함
"""

import numpy as np
import osqp
from scipy import sparse


class BalanceQPSolver:
    def __init__(self, mu=0.6, fz_min=10.0, fz_max=500.0,
                 alpha=1e-4, beta=1e-5,
                 eps_abs=1e-5, eps_rel=1e-5, max_iter=4000):
        """
        Parameters
        ----------
        mu : 마찰 계수
        fz_min : 최소 법선력 (unilateral constraint)
        fz_max : 최대 법선력
        alpha : ||F||^2 regularization weight
        beta : ||F - F_prev||^2 smoothing weight
        """
        self.mu = mu
        self.fz_min = fz_min
        self.fz_max = fz_max
        self.alpha = alpha
        self.beta = beta

        # OSQP 설정
        self.osqp_settings = {
            'verbose': False,
            'warm_start': True,
            'max_iter': max_iter,
            'eps_abs': eps_abs,
            'eps_rel': eps_rel,
            'polish': True,
            'adaptive_rho': True,
        }

        # 상태 추적
        self.solver = None
        self.prev_contact = None       # 이전 접촉 상태
        self.F_prev = np.zeros(12)     # 이전 최적 힘 (4다리 full)
        self.n_stance = 0
        self.stance_idx = np.array([], dtype=int)

    def _build_friction_constraints(self, n_stance):
        """
        Friction pyramid + unilateral constraints 구성

        다리당 5개 제약:
          fx - mu*fz <= 0
         -fx - mu*fz <= 0
          fy - mu*fz <= 0
         -fy - mu*fz <= 0
          fz_min <= fz <= fz_max
        """
        n_vars = 3 * n_stance
        n_con = 5 * n_stance
        C = np.zeros((n_con, n_vars))
        l = np.zeros(n_con)
        u = np.zeros(n_con)

        for i in range(n_stance):
            col = 3 * i
            row = 5 * i

            # fx - mu*fz <= 0
            C[row, col] = 1.0
            C[row, col + 2] = -self.mu
            l[row] = -np.inf
            u[row] = 0.0

            # -fx - mu*fz <= 0
            C[row + 1, col] = -1.0
            C[row + 1, col + 2] = -self.mu
            l[row + 1] = -np.inf
            u[row + 1] = 0.0

            # fy - mu*fz <= 0
            C[row + 2, col + 1] = 1.0
            C[row + 2, col + 2] = -self.mu
            l[row + 2] = -np.inf
            u[row + 2] = 0.0

            # -fy - mu*fz <= 0
            C[row + 3, col + 1] = -1.0
            C[row + 3, col + 2] = -self.mu
            l[row + 3] = -np.inf
            u[row + 3] = 0.0

            # fz_min <= fz <= fz_max
            C[row + 4, col + 2] = 1.0
            l[row + 4] = self.fz_min
            u[row + 4] = self.fz_max

        return C, l, u

    @staticmethod
    def skew(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def _build_A_dyn(self, p_feet_stance, p_com):
        """
        동역학 A 행렬 구성: [I ... I; [r1x] ... [rcx]]

        Parameters
        ----------
        p_feet_stance : (n_stance, 3) stance 발 위치
        p_com : (3,) CoM 위치
        """
        n = 3 * self.n_stance
        A = np.zeros((6, n))
        for i in range(self.n_stance):
            A[0:3, 3*i:3*i+3] = np.eye(3)
            r = p_feet_stance[i] - p_com
            A[3:6, 3*i:3*i+3] = self.skew(r)
        return A

    def _need_rebuild(self, contact_state):
        """접촉 상태가 바뀌면 solver 재구성 필요"""
        if self.prev_contact is None:
            return True
        return not np.array_equal(contact_state, self.prev_contact)

    def _setup_solver(self, P, q, C, l, u):
        """OSQP solver 초기 설정 또는 재설정"""
        self.solver = osqp.OSQP()
        self.solver.setup(
            P=sparse.csc_matrix(P),
            q=q,
            A=sparse.csc_matrix(C),
            l=l, u=u,
            **self.osqp_settings
        )

    def solve(self, p_com, p_feet, contact_state, bd, S):
        """
        Balance QP를 풀어 최적 GRF를 반환

        Parameters
        ----------
        p_com : (3,) CoM 위치 (world frame)
        p_feet : (4, 3) 4개 발 위치 (world frame), FR/FL/BR/BL 순
        contact_state : (4,) 접촉 상태 (1=stance, 0=swing)
        bd : (6,) desired wrench [force(3); torque(3)]
        S : (6, 6) wrench 추종 가중치 행렬

        Returns
        -------
        F_full : (12,) 4다리 힘 벡터 (swing 다리는 0)
        solved : bool 성공 여부
        """
        # --- stance 다리 파악 ---
        self.stance_idx = np.where(contact_state == 1)[0]
        self.n_stance = len(self.stance_idx)

        if self.n_stance == 0:
            # flight phase: 힘 없음
            self.F_prev = np.zeros(12)
            return self.F_prev.copy(), True

        n_vars = 3 * self.n_stance

        # --- stance 다리만 추출 ---
        p_feet_stance = p_feet[self.stance_idx]

        # F_prev에서 stance 다리 부분만 추출
        F_prev_stance = np.zeros(n_vars)
        for i, leg in enumerate(self.stance_idx):
            F_prev_stance[3*i:3*i+3] = self.F_prev[3*leg:3*leg+3]

        # --- A 행렬 ---
        A_dyn = self._build_A_dyn(p_feet_stance, p_com)

        # --- P, q 구성 ---
        P = 2.0 * (A_dyn.T @ S @ A_dyn
                    + (self.alpha + self.beta) * np.eye(n_vars))
        q = -2.0 * (A_dyn.T @ S @ bd + self.beta * F_prev_stance)

        # --- 접촉 상태 변경 → 전체 재구성 ---
        if self._need_rebuild(contact_state):
            C, l, u = self._build_friction_constraints(self.n_stance)
            self._setup_solver(P, q, C, l, u)
            self.prev_contact = contact_state.copy()
        else:
            # 접촉 동일 → P, q만 업데이트 (warm start 유지)
            P_sparse = sparse.csc_matrix(P)
            self.solver.update(
                Px=sparse.triu(P_sparse).data,
                q=q
            )

        # --- 풀기 ---
        result = self.solver.solve()

        # --- 결과 처리 ---
        if result.info.status_val == 1:  # solved
            F_stance = result.x

            # full 12차원 벡터로 복원
            F_full = np.zeros(12)
            for i, leg in enumerate(self.stance_idx):
                F_full[3*leg:3*leg+3] = F_stance[3*i:3*i+3]

            self.F_prev = F_full.copy()
            return F_full, True
        else:
            # 실패 시 이전 값 유지
            return self.F_prev.copy(), False


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":

    # --- 로봇 파라미터 ---
    m = 45.0
    I_body = np.diag([0.35, 2.1, 2.1])

    # --- 상태 ---
    p_com = np.array([0.0, 0.0, 0.4])
    p_feet = np.array([
        [ 0.30, -0.13, 0.0],
        [ 0.30,  0.13, 0.0],
        [-0.30, -0.13, 0.0],
        [-0.30,  0.13, 0.0],
    ])

    # --- PD desired acceleration ---
    Kp_pos = np.diag([100, 100, 100])
    Kd_pos = np.diag([20, 20, 20])
    Kp_ori = np.diag([100, 100, 100])
    Kd_ori = np.diag([20, 20, 20])

    p_ddot_d = Kp_pos @ np.array([0, 0, 0]) + Kd_pos @ np.array([0.5, 0, 0])
    omega_dot_d = Kp_ori @ np.array([0, 0.02, 0]) + Kd_ori @ np.zeros(3)

    bd = np.zeros(6)
    bd[0:3] = m * (p_ddot_d + np.array([0, 0, 9.81]))
    bd[3:6] = I_body @ omega_dot_d

    S = np.diag([1, 1, 10, 5, 5, 5])

    # --- 솔버 생성 ---
    qp = BalanceQPSolver(mu=0.6, fz_min=10, fz_max=500)

    # --- Test 1: trot (FR+BL stance) ---
    print("=== Test 1: Trot (FR + BL) ===")
    contact = np.array([1, 0, 0, 1])
    F, ok = qp.solve(p_com, p_feet, contact, bd, S)
    print(f"Solved: {ok}")
    names = ['FR', 'FL', 'BR', 'BL']
    for i in range(4):
        f = F[3*i:3*i+3]
        if contact[i]:
            fn = f[2]
            ft = np.sqrt(f[0]**2 + f[1]**2)
            print(f"  {names[i]}: fx={f[0]:7.2f} fy={f[1]:7.2f} fz={f[2]:7.2f}  |ft|/fn={ft/fn:.3f}")
        else:
            print(f"  {names[i]}: (swing)")

    achieved = np.zeros(6)
    for i, leg in enumerate(qp.stance_idx):
        fi = F[3*leg:3*leg+3]
        achieved[0:3] += fi
        r = p_feet[leg] - p_com
        achieved[3:6] += np.cross(r, fi)
    print(f"  bd target: {bd}")
    print(f"  achieved:  {achieved}")
    print(f"  error:     {achieved - bd}")

    # --- Test 2: 같은 접촉, bd 변경 (warm start) ---
    print("\n=== Test 2: Warm start (같은 접촉, bd 변경) ===")
    bd2 = bd.copy()
    bd2[0] += 20
    F2, ok2 = qp.solve(p_com, p_feet, contact, bd2, S)
    print(f"Solved: {ok2}")

    # --- Test 3: 접촉 변경 (FL+BR) → 재setup ---
    print("\n=== Test 3: 접촉 변경 (FL + BR) ===")
    contact2 = np.array([0, 1, 1, 0])
    F3, ok3 = qp.solve(p_com, p_feet, contact2, bd, S)
    print(f"Solved: {ok3}")
    for i in range(4):
        f = F3[3*i:3*i+3]
        if contact2[i]:
            fn = f[2]
            ft = np.sqrt(f[0]**2 + f[1]**2)
            print(f"  {names[i]}: fx={f[0]:7.2f} fy={f[1]:7.2f} fz={f[2]:7.2f}  |ft|/fn={ft/fn:.3f}")
        else:
            print(f"  {names[i]}: (swing)")

    # --- Test 4: 4다리 stance ---
    print("\n=== Test 4: 4다리 stance ===")
    contact3 = np.array([1, 1, 1, 1])
    F4, ok4 = qp.solve(p_com, p_feet, contact3, bd, S)
    print(f"Solved: {ok4}")
    for i in range(4):
        f = F4[3*i:3*i+3]
        fn = f[2]
        ft = np.sqrt(f[0]**2 + f[1]**2)
        print(f"  {names[i]}: fx={f[0]:7.2f} fy={f[1]:7.2f} fz={f[2]:7.2f}  |ft|/fn={ft/fn:.3f}")

    print("\n완료!")