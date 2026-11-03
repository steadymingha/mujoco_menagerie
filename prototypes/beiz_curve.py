import os
# os.environ['DISPLAY'] = '127.0.0.1:0.0'
import numpy as np
import matplotlib.pyplot as plt
import math

def bezier_curve(points, n=100):
    t = np.linspace(0, 1, n)
    curve = np.zeros((n, 2))
    degree = len(points) - 1

    for i in range(degree + 1):
        binomial = math.comb(degree, i)
        curve += binomial * ((1 - t) ** (degree - i))[:, None] * (t ** i)[:, None] * points[i]

    return curve

def bezier_swing_trajectory(p_start, p_end, t, step_height=0.08):
    """
    Cubic Bézier 스윙 궤적 생성

    Args:
        p_start: 스윙 시작 발 위치 (3,)
        p_end: 착지 목표점, 수식(6) 결과 (3,)
        t: 0~1 정규화 시간 (0=시작, 1=착지)
        step_height: 원하는 발 clearance [m]

    Returns:
        pos: 위치 (3,)  → 수식(8) p_i,ref
        vel: dp/dt (3,)  → 수식(8) v_i,ref (시간 스케일링 필요)
        acc: d²p/dt² (3,) → 수식(7) a_i,ref (시간 스케일링 필요)
    """
    # Bézier 최고점 보정 (cubic Bézier는 ~75%까지만 올라감)
    h = step_height * 1.35

    # 제어점 정의
    P0 = p_start
    P1 = p_start.copy()
    P1[2] += h
    P2 = p_end.copy()
    P2[2] += h
    P3 = p_end

    # 위치: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    s = 1 - t
    pos = s**3 * P0 + 3 * s**2 * t * P1 + 3 * s * t**2 * P2 + t**3 * P3

    # 1차 미분: dB/dt
    vel = (3 * s**2 * (P1 - P0)
         + 6 * s * t * (P2 - P1)
         + 3 * t**2 * (P3 - P2))

    # 2차 미분: d²B/dt²
    acc = (6 * s * (P2 - 2 * P1 + P0)
         + 6 * t * (P3 - 2 * P2 + P1))

    return pos, vel, acc

if __name__ == "__main__":
#    control_points = np.array([
#         [0, 0],
#         [1, 2],
#         [3, 3],
#         [4, 0]
#     ])

#     curve = bezier_curve(control_points)

#     cp2 = np.array([ [0, 0], [6,2], [4, 0]])
#     c2 = bezier_curve(cp2)

#     plt.plot(curve[:, 0], curve[:, 1], label='Bézier Curve')
#     plt.plot(c2[:, 0], c2[:, 1])
#     plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control Points')
#     plt.legend()
#     plt.axis('equal')
#     plt.grid(True)
#     plt.title('Cubic Bézier Curve')
   
    # --- bezier_swing_trajectory 예시 ---
    p_start = np.array([0.15,  0.1, 0.0])   # 발 시작점 (world frame)
    p_end   = np.array([0.35, -0.1, 0.0])   # 발 착지점
    step_height = 0.08

    N = 200
    ts = np.linspace(0, 1, N)

    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    accels = np.zeros((N, 3))

    for i, t in enumerate(ts):
        pos, vel, acc = bezier_swing_trajectory(p_start, p_end, t, step_height)
        positions[i]  = pos
        velocities[i] = vel
        accels[i]     = acc

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Plot 1: 3D 궤적 ---
    ax = fig.add_subplot(131, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], lw=2, label='trajectory')
    ax.scatter(*p_start, color='green', s=60, label='start')
    ax.scatter(*p_end,   color='red',   s=60, label='end')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Swing Trajectory (3D)')
    ax.legend()

    fig.delaxes(axes[0])   # subplot(1,3,1)은 3D로 대체됐으므로 제거

    # --- Plot 2: XZ 단면 (side view) ---
    ax2 = axes[1]
    ax2.plot(positions[:, 0], positions[:, 2], lw=2)
    ax2.scatter(p_start[0], p_start[2], color='green', s=60, zorder=5, label='start')
    ax2.scatter(p_end[0],   p_end[2],   color='red',   s=60, zorder=5, label='end')
    ax2.axhline(0, color='gray', linestyle='--', lw=0.8, label='ground')
    ax2.set_xlabel('X [m]'); ax2.set_ylabel('Z [m]')
    ax2.set_title('Side View (XZ)')
    ax2.legend(); ax2.grid(True)

    # --- Plot 3: 속도 / 가속도 시계열 ---
    ax3 = axes[2]
    ax3.plot(ts, velocities[:, 0], label='vel x')
    ax3.plot(ts, velocities[:, 2], label='vel z')
    ax3.plot(ts, accels[:, 0],     label='acc x', linestyle='--')
    ax3.plot(ts, accels[:, 2],     label='acc z', linestyle='--')
    ax3.set_xlabel('t (normalized)'); ax3.set_ylabel('value')
    ax3.set_title('Velocity & Acceleration')
    ax3.legend(); ax3.grid(True)

    plt.tight_layout()
    # plt.savefig('bezier_swing.png', dpi=120)
    plt.show()
