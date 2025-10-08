# # Python 3.x
# # Required: numpy, matplotlib

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# from matplotlib.widgets import Slider

# # ---------- Kinematics ----------
# def fk_2link(theta1, theta2, l1=1.0, l2=0.7):
#     """Forward kinematics of a 2-link planar arm. Returns joint and end-effector points."""
#     p0 = np.array([0.0, 0.0])
#     p1 = np.array([l1*np.cos(theta1), l1*np.sin(theta1)])
#     p2 = p1 + np.array([l2*np.cos(theta1+theta2), l2*np.sin(theta1+theta2)])
#     return p0, p1, p2

# def jacobian_2link(theta1, theta2, l1=1.0, l2=0.7):
#     """Geometric Jacobian mapping joint velocities to end-effector linear velocity in 2D."""
#     s1, c1 = np.sin(theta1), np.cos(theta1)
#     s12, c12 = np.sin(theta1+theta2), np.cos(theta1+theta2)
#     J = np.array([
#         [-l1*s1 - l2*s12, -l2*s12],
#         [ l1*c1 + l2*c12,  l2*c12]
#     ])
#     return J

# # ---------- Manipulability Ellipse ----------
# def ellipse_from_J(J, center, scale=1.0):
#     """
#     Build a velocity manipulability ellipse for ||dq|| = 1.
#     The ellipse in task space satisfies dx^T (J J^T)^-1 dx = 1.
#     We plot an ellipse whose axes are sqrt(eigenvalues) of (J J^T).
#     """
#     JJt = J @ J.T
#     # Regularize for near-singular cases
#     JJt = JJt + 1e-12*np.eye(2)
#     vals, vecs = np.linalg.eigh(JJt)  # vals >= 0
#     # Semi-axis lengths are sqrt of eigenvalues (for unit joint-velocity ball)
#     a, b = np.sqrt(vals[1]), np.sqrt(vals[0])  # ensure a>=b by sorting next
#     # Sort so that a is the largest
#     order = np.argsort([a, b])[::-1]
#     axes = np.array([a, b])[order] * scale
#     vecs = vecs[:, order]
#     angle_deg = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
#     e = Ellipse(xy=center, width=2*axes[0], height=2*axes[1],
#                 angle=angle_deg, fill=False, linewidth=2, linestyle='--')
#     return e, axes

# # ---------- Plot Setup ----------
# def draw_scene(ax, th1, th2, l1=1.0, l2=0.7):
#     """Render arm, Jacobian columns, and manipulability ellipse."""
#     ax.clear()
#     p0, p1, p2 = fk_2link(th1, th2, l1, l2)
#     J = jacobian_2link(th1, th2, l1, l2)

#     # Robot links
#     ax.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], 'o-', linewidth=3)

#     # Jacobian columns as velocity basis at end-effector
#     v1, v2 = J[:, 0], J[:, 1]
#     ax.quiver(p2[0], p2[1], v1[0], v1[1], angles='xy', scale_units='xy', scale=1, label='J col 1 (joint1)')
#     ax.quiver(p2[0], p2[1], v2[0], v2[1], angles='xy', scale_units='xy', scale=1, label='J col 2 (joint2)')

#     # Manipulability ellipse
#     ell, axes = ellipse_from_J(J, p2, scale=1.0)
#     ax.add_patch(ell)

#     # Singularity indicator (rank drop when det(J) ~ 0)
#     detJJt = np.linalg.det(J @ J.T)
#     if detJJt < 1e-6:
#         ax.text(0.02, 0.95, 'Near singularity', transform=ax.transAxes, color='red', fontsize=11, ha='left', va='top')

#     # Formatting
#     ax.set_aspect('equal', adjustable='box')
#     r = l1 + l2 + 0.2
#     ax.set_xlim(-r, r)
#     ax.set_ylim(-r, r)
#     ax.grid(True)
#     ax.set_title('Geometric Role of the Jacobian (2-Link Planar Arm)')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.legend(loc='upper left', fontsize=9)

# def main():
#     # Initial joint angles (radians)
#     th1_0 = np.deg2rad(40)
#     th2_0 = np.deg2rad(30)

#     fig, ax = plt.subplots(figsize=(7, 7))
#     plt.subplots_adjust(bottom=0.22)
#     draw_scene(ax, th1_0, th2_0)

#     # Sliders for interactive exploration
#     ax_th1 = plt.axes([0.15, 0.10, 0.70, 0.03])
#     ax_th2 = plt.axes([0.15, 0.05, 0.70, 0.03])
#     s_th1 = Slider(ax_th1, 'theta1 (deg)', -180, 180, valinit=np.degrees(th1_0))
#     s_th2 = Slider(ax_th2, 'theta2 (deg)', -180, 180, valinit=np.degrees(th2_0))

#     def update(_):
#         th1 = np.deg2rad(s_th1.val)
#         th2 = np.deg2rad(s_th2.val)
#         draw_scene(ax, th1, th2)
#         fig.canvas.draw_idle()

#     s_th1.on_changed(update)
#     s_th2.on_changed(update)

#     plt.show()

# if __name__ == '__main__':
#     main()


import numpy as np
import matplotlib.pyplot as plt

# 로봇팔의 링크 길이 정의
L1 = 1.0
L2 = 1.0

# 순방향 기구학 (Forward Kinematics)
def forward_kinematics(q):
    """
    관절 각도(q1, q2)로부터 로봇팔 끝의 (x, y) 좌표를 계산합니다.
    """
    q1, q2 = q[0], q[1]
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, y])

# 자코비안 계산
def get_jacobian(q):
    """
    주어진 관절 각도(q1, q2)에서 자코비안 행렬을 계산합니다.
    """
    q1, q2 = q[0], q[1]
    # partial derivatives
    # dx/dq1, dx/dq2
    # dy/dq1, dy/dq2
    j11 = -L1 * np.sin(q1) - L2 * np.sin(q1 + q2)
    j12 = -L2 * np.sin(q1 + q2)
    j21 = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    j22 = L2 * np.cos(q1 + q2)
    return np.array([[j11, j12], [j21, j22]])

# --- 시각화 ---

# 1. 기준 관절 각도 설정 (로봇팔의 현재 자세)
q_center = np.array([np.pi / 4, np.pi / 4])

# 2. 관절 공간(Joint Space)에서 작은 사각형 정의
dq = 0.2  # 작은 변화량
joint_space_square = np.array([
    q_center + np.array([-dq, -dq]),
    q_center + np.array([dq, -dq]),
    q_center + np.array([dq, dq]),
    q_center + np.array([-dq, dq]),
    q_center + np.array([-dq, -dq])
])

# 3. 실제 비선형 변환 (Forward Kinematics 적용)
cartesian_true_shape = np.array([forward_kinematics(q) for q in joint_space_square])

# 4. 자코비안을 이용한 선형 근사 변환
p_center = forward_kinematics(q_center)
J = get_jacobian(q_center)
joint_space_offsets = joint_space_square - q_center
cartesian_linear_offsets = np.array([(J @ dq_vec) for dq_vec in joint_space_offsets])
cartesian_linear_shape = cartesian_linear_offsets + p_center


# 5. 플롯 생성
plt.figure(figsize=(10, 10))
ax = plt.gca()

# 로봇팔 그리기
p0 = np.array([0, 0])
p1 = np.array([L1 * np.cos(q_center[0]), L1 * np.sin(q_center[0])])
p2 = forward_kinematics(q_center)
plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', lw=3, label='Link 1')
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=3, label='Link 2')
plt.plot(p2[0], p2[1], 'ko', markersize=10, label='End-Effector')

# 변환된 도형들 그리기
plt.plot(cartesian_true_shape[:, 0], cartesian_true_shape[:, 1], 'r-', lw=2, label='True Transformation (Non-linear)')
plt.plot(cartesian_linear_shape[:, 0], cartesian_linear_shape[:, 1], 'b--', lw=2, label='Jacobian Approximation (Linear)')

# 평행사변형의 각 꼭짓점 표시
plt.scatter(cartesian_linear_shape[:-1, 0], cartesian_linear_shape[:-1, 1], c='blue', s=50, zorder=5)
plt.scatter(cartesian_true_shape[:-1, 0], cartesian_true_shape[:-1, 1], c='red', s=50, zorder=5)


plt.title('Jacobian Geometric Interpretation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Save the figure
plt.savefig('jacobian_visualization.png')

print("Visualization saved as jacobian_visualization.png")