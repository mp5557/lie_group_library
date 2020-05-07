from lie_group import *

import unittest


class LieGroupTest(unittest.TestCase):
    def test_Hat(self):
        v = np.array([1, 2, 3])
        h = Hat(v)
        np.testing.assert_almost_equal(h + h.T, np.zeros((3, 3)))
        np.testing.assert_almost_equal(h @ v, np.zeros(3,))

    def test_Vee(self):
        v = np.array([1,2,3])
        np.testing.assert_almost_equal(Vee(Hat(v)), v)

    def test_LieExp(self):
        v = np.array([1, 2, 3])
        v = v / la.norm(v)
        C = LieExp(v * pi / 6)
        np.testing.assert_almost_equal(C @ C.T, np.eye(
            3), err_msg="should be orthogonal")
        np.testing.assert_almost_equal(C @ v, v)
        np.testing.assert_almost_equal(LieExp([0, 0, 0]), np.eye(3))

    def test_LieLog(self):
        v = np.array([.1, .2, .3])
        np.testing.assert_almost_equal(LieLog(LieExp(v)), v)
        np.testing.assert_almost_equal(LieLog(LieExp(-1. * v)), -1. * v)
        np.testing.assert_almost_equal(LieLog(np.eye(3)), np.zeros((3,)))

    def test_RoundToRotation(self):
        v = np.array([.1, .2, .3])
        C = Hat(v) + np.eye(3)
        M = RoundToRotation(C)
        V = LieExp(v)
        np.testing.assert_almost_equal(
            M @ M.T, np.eye(3), err_msg="should be orthogonal")
        np.testing.assert_almost_equal(M, V, 2)


if __name__ == '__main__':
    unittest.main()
