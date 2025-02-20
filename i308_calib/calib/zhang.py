import numpy as np


def compute_homography(world_points, image_points):
    """
        Compute homography matrix H such that image_points ~ H * world_points.
    """
    A = []
    for wp, ip in zip(world_points, image_points):
        X, Y = wp[0], wp[1]
        u, v = ip[0], ip[1]
        A.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape((3, 3))

    return H / H[-1, -1]


def intrinsic_from_homographies(homographies):
    V = []
    for H in homographies:
        h1 = H[:, 0]
        h2 = H[:, 1]
        v12 = np.array([
            h1[0] * h2[0], h1[0] * h2[1] + h1[1] * h2[0], h1[1] * h2[1],
            h1[2] * h2[0] + h1[0] * h2[2], h1[2] * h2[1] + h1[1] * h2[2], h1[2] * h2[2]
        ])
        v11 = np.array([
            h1[0] * h1[0], h1[0] * h1[1] + h1[1] * h1[0], h1[1] * h1[1],
            h1[2] * h1[0] + h1[0] * h1[2], h1[2] * h1[1] + h1[1] * h1[2], h1[2] * h1[2]
        ])
        v22 = np.array([
            h2[0] * h2[0], h2[0] * h2[1] + h2[1] * h2[0], h2[1] * h2[1],
            h2[2] * h2[0] + h2[0] * h2[2], h2[2] * h2[1] + h2[1] * h2[2], h2[2] * h2[2]
        ])
        V.append(v12)
        V.append(v11 - v22)

    V = np.array(V)
    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1, :]

    B11, B12, B22, B13, B23, B33 = b
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12)
    lambda_ = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lambda_ / B11)
    beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12 * B12))
    gamma = -B12 * alpha * alpha * beta / lambda_
    u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda_

    intrinsic_matrix = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    return intrinsic_matrix


def extrinsics_from_homography(H, intrinsic_matrix):
    K_inv = np.linalg.inv(intrinsic_matrix)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    lambda_ = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = lambda_ * np.dot(K_inv, h1)
    r2 = lambda_ * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    t = lambda_ * np.dot(K_inv, h3)

    R = np.column_stack((r1, r2, r3))
    U, S, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)  # Ensure R is a proper rotation matrix

    return R, t
