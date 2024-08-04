import numpy as np
from numpy.testing import assert_array_equal

from ppmpy.eigen import eigen


class TestGrid:

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """

    def setup_method(self):
        """ this is run before each test """

    def teardown_method(self):
        """ this is run after each test """

    def test_eigenvalues(self):

        rho = 1.0
        u = 0.5
        p = 2.5

        gamma = 1.4

        ev, lvec, rvec = eigen(rho, u, p, gamma)

        cs = np.sqrt(gamma * p / rho)

        assert_array_equal(ev, np.array([u - cs, u, u + cs]))

        D = lvec @ rvec.T - np.eye(3)

        assert np.abs(D).max() < 2.e-16
