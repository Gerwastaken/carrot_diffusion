import numpy as np
from scipy.spatial.transform import Rotation
# https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
# https://arxiv.org/pdf/2103.15980
# llm完全不行

# 实现SE(3)和se(3)之间的转换


def hat(omega):
    """将3维向量转换为3x3反对称矩阵"""
    wx, wy, wz = omega
    return np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])


def se3_log(T):
    """SE(3) -> se(3)  6-vector [w; v]"""
    R = T[:3, :3]
    t = T[:3, 3]
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta < 1e-10:
        w = np.zeros(3)
        v = t
    else:
        lnR = theta / (2 * np.sin(theta)) * (R - R.T)
        w = np.array([lnR[2, 1], lnR[0, 2], lnR[1, 0]])

        K = hat(w)

        V = (
            np.eye(3) +
            (1 - np.cos(theta)) / (theta**2) * K +
            (theta - np.sin(theta)) / (theta**3) * (K @ K)
        )
        v = np.linalg.inv(V) @ t
    return np.concat([w, v])


def se3_exp(xi):
    """se(3) 6-vector [w; v] -> SE(3)"""
    w = xi[:3]
    t = xi[3:]
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        R = np.eye(3)
        p = t
    else:
        R = Rotation.from_rotvec(w).as_matrix()
        K = hat(w)

        V = (
            np.eye(3) +
            (1 - np.cos(theta)) / (theta**2) * K +
            (theta - np.sin(theta)) / (theta**3) * (K @ K)
        )
        p = V @ t
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


if __name__ == '__main__':
    # 测试代码
    import scipy.linalg

    # 随机生成一个SE(3)矩阵
    R = scipy.linalg.expm(hat(np.array([0.3, 0.2, 0.1])))
    t = np.array([1.0, 2.0, 3.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    print("T:", T)

    # SE(3) -> se(3)
    xi = se3_log(T)
    print("se(3) xi:", xi)

    # se(3) -> SE(3)
    T_reconstructed = se3_exp(xi)
    print("Reconstructed T:\n", T_reconstructed)

    # 检查是否相等
    assert np.allclose(T, T_reconstructed), "Reconstruction failed!"
