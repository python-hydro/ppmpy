import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp

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

    def test_coords(self):

        assert_array_equal(self.g.x, np.array([-16., -12.,  -8.,  -4.,   0.,   4.,   8.,  12.,  16.]))
        assert_array_equal(self.g.xl, np.array([-18., -14., -10.,  -6.,  -2.,   2.,   6.,  10.,  14.]))
        assert_array_equal(self.g.xr, np.array([-14., -10.,  -6.,  -2.,   2.,   6.,  10.,  14., 18.]))

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
