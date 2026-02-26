import numpy as np

@staticmethod
def euler123_to_R(roll, pitch, yaw):
    """Intrinsic 1-2-3 (XYZ): Rx → Ry → Rz"""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1,  0,   0],
                    [0,  cr, -sr],
                    [0,  sr,  cr]])
    Ry = np.array([[ cp, 0, sp],
                    [  0, 1,  0],
                    [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                    [sy,  cy, 0],
                    [ 0,   0, 1]])
    return Rx @ Ry @ Rz

@staticmethod
def SO3_logmap(R):
    """SO(3) log map: R → rotation vector (R³)"""
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        # θ ≈ 0: R ≈ I, 미소 회전
        return np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / 2.0

    if np.pi - theta < 1e-6:
        # θ ≈ π: sin(θ) ≈ 0이라 일반 공식 불안정
        # R = I + 2*sin(θ)*[w]× + 2*sin²(θ/2)*[w]×² 에서
        # θ ≈ π일 때 (R + I)/2 의 대각성분으로 축 추출
        S = (R + np.eye(3)) / 2.0
        # 가장 큰 대각 성분 선택
        idx = np.argmax(np.diag(S))
        w = S[:, idx] / np.sqrt(S[idx, idx])
        return w * theta

    # 일반 케이스
    omega_skew = (theta / (2 * np.sin(theta))) * (R - R.T)
    return np.array([omega_skew[2,1], omega_skew[0,2], omega_skew[1,0]])

def quaternion_to_R(q):

    w, x, y, z = q[0], q[1], q[2], q[3]

    # 쿼터니언 → 회전행렬 (world → body)
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),       2*(x*z + w*y)],
        [    2*(x*y + w*z),    1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),        2*(y*z + w*x),    1 - 2*(x**2 + y**2)],
    ])          

    return R

@staticmethod
def euler123_jacobian(pitch, yaw):
    """Intrinsic 1-2-3 (XYZ): Euler 각속도 → body frame 각속도 변환 자코비안
    ω_body = B @ [roll_dot, pitch_dot, yaw_dot]
    singular: pitch = ±90° (gimbal lock)
    """
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    B = np.array([[ cp*cy,  sy, 0],
                    [-cp*sy,  cy, 0],
                    [    sp,   0, 1]])
    return B