import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp

from ppmpy.grid import FVGrid
from ppmpy.reconstruction import PPMInterpolant


class TestPPM:

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """

    def setup_method(self):
        """ this is run before each test """

        g = FVGrid(4, ng=3)
        self.ppm = PPMInterpolant(g, np.array([1, 1, 1, 1, 0.5, 0.25, 0, 0, 0, 0]))
        self.ppm.construct_parabola()

    def teardown_method(self):
        """ this is run after each test """

    def test_parabola(self):
        am = np.array([1.0, 1.0, 1.0, 1.0, 0.7916666666666666, 0.3541666666666667,
                       0.0, 0.0, 0.0, 0.0])

        ap = np.array([1.0, 1.0, 1.0, 1.0, 0.3541666666666667, 0.08333333333333334,
                       0.0, 0.0, 0.0, 0.0])

        a6 = np.array([0.0, 0.0, 0.0, 0.0, -0.4375, 0.1875, 0.,
                       0.0, 0.0, 0.0])

        assert_array_almost_equal_nulp(am, self.ppm.am)
        assert_array_almost_equal_nulp(ap, self.ppm.ap)
        assert_array_almost_equal_nulp(a6, self.ppm.a6)

    def test_ghostcells(self):

        # we should get the same results if we use 3 or 4 ghost cells
        g2 = FVGrid(4, ng=4)
        a2 = g2.scratch_array()
        a2[g2.lo:g2.hi+1] = self.ppm.a[self.ppm.grid.lo:self.ppm.grid.hi+1]
        g2.ghost_fill(a2)

        ppm2 = PPMInterpolant(g2, a2)
        ppm2.construct_parabola()

        assert_array_equal(ppm2.am[1:-1], self.ppm.am)
        assert_array_equal(ppm2.ap[1:-1], self.ppm.ap)
        assert_array_equal(ppm2.a6[1:-1], self.ppm.a6)
