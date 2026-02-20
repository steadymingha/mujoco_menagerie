"""
OSQP 사용 예제: Cheetah3 Balance Controller QP

풀어야 할 QP (Cheetah3 논문 식 (4)):
  F* = argmin_F  (AF - bd)^T S (AF - bd) + alpha||F||^2
  s.t.  CF <= d

OSQP 표준형:
  min  (1/2) x^T P x + q^T x
  s.t.  l <= A_osqp x <= u
"""

import numpy as np
import osqp
from scipy import sparse

# ============================================================
# 1. 로봇 파라미터
# ============================================================
m = 45.0
I_body = np.diag([0.35, 2.1, 2.1])
mu = 0.6

# ============================================================
# 2. 현재 상태 (예시)
# ============================================================
p_com = np.array([0.0, 0.0, 0.4])

p_feet = np.array([
    [ 0.30, -0.13, 0.0],   # FR
    [ 0.30,  0.13, 0.0],   # FL
    [-0.30, -0.13, 0.0],   # BR
    [-0.30,  0.13, 0.0],   # BL
])

# trot: FR+BL stance
contact_state = np.array([1, 0, 0, 1])

# ============================================================
# 3. Desired 가속도 (PD Control Law, 논문 식 (2))
# ============================================================
Kp_pos = np.diag([500, 500, 500])
Kd_pos = np.diag([100, 100, 100])
Kp_ori = np.diag([300, 300, 300])
Kd_ori = np.diag([50, 50, 50])

p_com_d = np.array([0.0, 0.0, 0.4])
p_com_dot = np.array([0.0, 0.0, 0.0])
p_com_dot_d = np.array([0.5, 0.0, 0.0])
omega_b = np.array([0.0, 0.0, 0.0])
omega_b_d = np.array([0.0, 0.0, 0.0])
ori_error = np.array([0.0, 0.02, 0.0])

p_ddot_d = Kp_pos @ (p_com_d - p_com) + Kd_pos @ (p_com_dot_d - p_com_dot)
omega_dot_d = Kp_ori @ ori_error + Kd_ori @ (omega_b_d - omega_b)

# bd 벡터 (논문 식 (3))
bd = np.zeros(6)
bd[0:3] = m * (p_ddot_d + np.array([0, 0, 9.81]))
bd[3:6] = I_body @ omega_dot_d

print("=== Desired Wrench (bd) ===")
print(f"  Force:  {bd[0:3]}")
print(f"  Torque: {bd[3:6]}")

# ============================================================
# 4. A 행렬 (Controller Model, 논문 식 (1))
# ============================================================
def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

stance_idx = np.where(contact_state == 1)[0]
n_stance = len(stance_idx)
n_vars = 3 * n_stance

print(f"\n=== Stance legs: {stance_idx}, 변수 수: {n_vars} ===")

A_dyn = np.zeros((6, n_vars))
for i, leg_idx in enumerate(stance_idx):
    A_dyn[0:3, 3*i:3*i+3] = np.eye(3)
    r = p_feet[leg_idx] - p_com
    A_dyn[3:6, 3*i:3*i+3] = skew(r)

# ============================================================
# 5. QP P, q 구성
# ============================================================
# Cost: (AF-bd)^T S (AF-bd) + alpha||F||^2
# -> (1/2) F^T P F + q^T F
#    P = 2(A^T S A + alpha*I)
#    q = -2 A^T S bd

S = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
alpha = 1e-4

P = 2.0 * (A_dyn.T @ S @ A_dyn + alpha * np.eye(n_vars))
q = -2.0 * A_dyn.T @ S @ bd

# ============================================================
# 6. 제약조건 (Friction Pyramid + Normal Force Bounds)
# ============================================================
# 다리당 5개 제약:
#   fx - mu*fz <= 0
#  -fx - mu*fz <= 0
#   fy - mu*fz <= 0
#  -fy - mu*fz <= 0
#   fz_min <= fz <= fz_max

fz_min = 10.0
fz_max = 500.0

n_constraints = 5 * n_stance
C_ineq = np.zeros((n_constraints, n_vars))
l_bound = np.zeros(n_constraints)
u_bound = np.zeros(n_constraints)

for i in range(n_stance):
    col = 3 * i
    row = 5 * i
    
    # fx - mu*fz <= 0
    C_ineq[row+0, col+0] = 1.0
    C_ineq[row+0, col+2] = -mu
    l_bound[row+0] = -np.inf
    u_bound[row+0] = 0.0
    
    # -fx - mu*fz <= 0
    C_ineq[row+1, col+0] = -1.0
    C_ineq[row+1, col+2] = -mu
    l_bound[row+1] = -np.inf
    u_bound[row+1] = 0.0
    
    # fy - mu*fz <= 0
    C_ineq[row+2, col+1] = 1.0
    C_ineq[row+2, col+2] = -mu
    l_bound[row+2] = -np.inf
    u_bound[row+2] = 0.0
    
    # -fy - mu*fz <= 0
    C_ineq[row+3, col+1] = -1.0
    C_ineq[row+3, col+2] = -mu
    l_bound[row+3] = -np.inf
    u_bound[row+3] = 0.0
    
    # fz_min <= fz <= fz_max
    C_ineq[row+4, col+2] = 1.0
    l_bound[row+4] = fz_min
    u_bound[row+4] = fz_max

# ============================================================
# 7. OSQP 솔버 실행
# ============================================================
P_sparse = sparse.csc_matrix(P)
C_sparse = sparse.csc_matrix(C_ineq)

solver = osqp.OSQP()
solver.setup(
    P=P_sparse, q=q,
    A=C_sparse, l=l_bound, u=u_bound,
    verbose=False,
    warm_start=True,
    max_iter=4000,
    eps_abs=1e-5,
    eps_rel=1e-5,
    polish=True,
    adaptive_rho=True,
)

result = solver.solve()

print(f"\n=== OSQP 결과 ===")
print(f"Status: {result.info.status}")
print(f"Iterations: {result.info.iter}")
print(f"Solve time: {result.info.solve_time*1000:.3f} ms")

# ============================================================
# 8. 결과 해석
# ============================================================
if result.info.status == 'solved':
    F_opt = result.x
    leg_names = ['FR', 'FL', 'BR', 'BL']
    
    print(f"\n=== 최적 GRF ===")
    for i, leg_idx in enumerate(stance_idx):
        f = F_opt[3*i:3*i+3]
        fn = f[2]
        ft = np.sqrt(f[0]**2 + f[1]**2)
        ratio = ft / fn if fn > 0 else 0
        print(f"  {leg_names[leg_idx]}: fx={f[0]:7.2f}, fy={f[1]:7.2f}, fz={f[2]:7.2f} [N]  (|ft|/fn={ratio:.3f}, mu={mu})")
    
    achieved = A_dyn @ F_opt
    error = achieved - bd
    print(f"\n=== Wrench 추종 ===")
    print(f"  목표 Force:  {bd[0:3]}")
    print(f"  달성 Force:  {achieved[0:3]}")
    print(f"  에러 Force:  {error[0:3]}")
    print(f"  목표 Torque: {bd[3:6]}")
    print(f"  달성 Torque: {achieved[3:6]}")
    print(f"  에러 Torque: {error[3:6]}")

# ============================================================
# 9. Warm Start 반복 호출 (제어 루프 시뮬레이션)
# ============================================================
print(f"\n=== Warm Start 반복 호출 ===")

bd_new = bd.copy()
bd_new[0] += 10.0
q_new = -2.0 * A_dyn.T @ S @ bd_new

solver.update(q=q_new)
result2 = solver.solve()
print(f"  q만 업데이트 -> Status: {result2.info.status}, "
      f"Time: {result2.info.solve_time*1000:.3f} ms, "
      f"Iter: {result2.info.iter}")

print("\n완료!")