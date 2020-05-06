import numpy as np
import scipy.linalg as la
from math import cos, sin, pi, acos


def Hat(v):
    return np.array([[0.,     -v[2],  v[1]],
                     [v[2],  0.,      -v[0]],
                     [-v[1], v[0],   0.]])


def LieExp(v):
    t = la.norm(v)
    if t < 1e-5:
        return np.eye(3)
    a = np.array(v) / t
    c = cos(t)
    s = sin(t)
    return c * np.eye(3) + (1 - c) * a * a.reshape((3, 1)) + s * Hat(a)


def LieLog(V):
    t = acos((np.trace(V) - 1) / 2)
    a = la.null_space(V - np.eye(3))[:, 0]
    return t * a


def RoundToRotation(C):
    U, _, Vh = la.svd(C)
    return U @ Vh
