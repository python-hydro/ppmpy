import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp
from pytest import approx

from ppmpy.grid import FVGrid


class TestGrid:

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """

    def setup_method(self):
        """ this is run before each test """

        self.g = FVGrid(5, ng=2, xmin=-10, xmax=10)

    def teardown_method(self):
        """ this is run after each test """

    def test_properties(self):

        assert self.g.nx == 5
        assert self.g.ng == 2
        assert self.g.nq == 9

    def test_indices(self):

        assert self.g.lo == 2
        assert self.g.hi == 6

    def test_dx(self):
        assert self.g.dx == 4.0

    def test_coords(self):

        assert_array_equal(self.g.x, np.array([-16., -12., -8., -4., 0., 4., 8., 12., 16.]))
        assert_array_equal(self.g.xl, np.array([-18., -14., -10., -6., -2., 2., 6., 10., 14.]))
        assert_array_equal(self.g.xr, np.array([-14., -10., -6., -2., 2., 6., 10., 14., 18.]))

        assert_array_almost_equal_nulp(self.g.x, 0.5 * (self.g.xl + self.g.xr))

        assert self.g.xl[self.g.lo] == -10.0
        assert self.g.xr[self.g.hi] == 10.0

    def test_ghostfill_outflow(self):
        a = self.g.scratch_array()
        a[self.g.lo:self.g.hi+1] = np.arange(self.g.nx)
        self.g.ghost_fill(a)

        assert a[self.g.lo-2] == a[self.g.lo]
        assert a[self.g.lo-1] == a[self.g.lo]

        assert a[self.g.hi+1] == a[self.g.hi]
        assert a[self.g.hi+2] == a[self.g.hi]

    def test_ghostfill_periodic(self):
        a = self.g.scratch_array()
        a[self.g.lo:self.g.hi+1] = np.arange(self.g.nx)
        self.g.ghost_fill(a,
                          bc_left_type="periodic",
                          bc_right_type="periodic")

        assert a[self.g.lo-2] == a[self.g.hi-1]
        assert a[self.g.lo-1] == a[self.g.hi]

        assert a[self.g.hi+1] == a[self.g.lo]
        assert a[self.g.hi+2] == a[self.g.lo+1]

    def test_ghostfill_reflect(self):
        a = self.g.scratch_array()
        a[self.g.lo:self.g.hi+1] = np.arange(self.g.nx) + 1
        self.g.ghost_fill(a,
                          bc_left_type="reflect",
                          bc_right_type="reflect")

        assert_array_equal(a, np.array([2., 1., 1., 2., 3., 4., 5., 5., 4.]))

    def test_ghostfill_reflect_odd(self):
        a = self.g.scratch_array()
        a[self.g.lo:self.g.hi+1] = np.arange(self.g.nx) + 1
        self.g.ghost_fill(a,
                          bc_left_type="reflect-odd",
                          bc_right_type="reflect-odd")

        assert_array_equal(a, np.array([-2., -1., 1., 2., 3., 4., 5., -5., -4.]))


class TestGridUtils:

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """

    def setup_method(self):
        """ this is run before each test """

        self.g = FVGrid(10, ng=3)
        self.a = self.g.scratch_array()
        self.a[self.g.lo:self.g.hi+1] = np.arange(10) + 1

    def teardown_method(self):
        """ this is run after each test """

    def test_coarsen(self):
        cg, ca = self.g.coarsen(self.a)
        assert_array_equal(ca, np.array([0., 0., 0., 1.5, 3.5, 5.5, 7.5, 9.5, 0., 0., 0.]))

    def test_norm(self):
        assert self.g.norm(self.a) == approx(6.2048368229954285)
