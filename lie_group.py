import numpy as np
import scipy.linalg as la
from math import cos, sin, pi, acos


def Hat(v):
    return np.array([[0.,     -v[2],  v[1]],
                     [v[2],  0.,      -v[0]],
                     [-v[1], v[0],   0.]])


def Vee(H):
    return np.array([H[2, 1], H[0, 2], H[1, 0]])


def LieExp(v):
    t = la.norm(v)
    if t < 1e-5:
        return np.eye(3)
    a = np.array(v) / t
    c = cos(t)
    s = sin(t)
    return c * np.eye(3) + (1 - c) * a * a.reshape((3, 1)) + s * Hat(a)


def LieLog(V):
    a = la.null_space(V - np.eye(3))[:, 0]
    c = (np.trace(V) - 1) / 2
    # s = Vee(V - c * np.eye(3) - (1 - c) * a * a.reshape((3, 1))) @ a

    # s =  (V[2, 1] - (1 - c) * a[2] * a[1]) * a[0]
    # s += (V[0, 2] - (1 - c) * a[0] * a[2]) * a[1] 
    # s += (V[1, 0] - (1 - c) * a[1] * a[0]) * a[2]

    s = -3 * (1 - c) * a[0] * a[1] * a[2]
    s += V[2, 1] * a[0]
    s += V[0, 2] * a[1]
    s += V[1, 0] * a[2]
    return np.arctan2(s, c) * a


def RoundToRotation(C):
    U, _, Vh = la.svd(C)
    return U @ Vh
