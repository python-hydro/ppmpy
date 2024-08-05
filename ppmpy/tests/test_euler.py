import numpy as np
from numpy.testing import assert_array_almost_equal_nulp

from ppmpy.euler import Euler
from ppmpy.initial_conditions import sod


class TestGrid:

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """

    def setup_method(self):
        """ this is run before each test """

        self.euler = Euler(32, 0.5, init_cond=sod)

    def teardown_method(self):
        """ this is run after each test """

    def test_cons_to_prim(self):

        # we used the primitive variables for the sod problem, so
        # we can check to see if we recover them

        q = self.euler.cons_to_prim()
        v = self.euler.v

        rho_l = 1.0
        u_l = 0.0
        p_l = 1.0

        rho_r = 0.125
        u_r = 0.0
        p_r = 0.1

        assert q[0, v.qrho] == rho_l
        assert q[0, v.qu] == u_l
        assert q[0, v.qp] == p_l

        assert q[-1, v.qrho] == rho_r
        assert q[-1, v.qu] == u_r
        assert q[-1, v.qp] == p_r

    def test_properties(self):

        # for sod, there is only a left and right state.  No initial velocity
        # so we can just get the soundspeed

        q = self.euler.cons_to_prim()
        v = self.euler.v
        cs_l = np.sqrt(self.euler.gamma * q[0, v.qp] / q[0, v.qrho])
        cs_r = np.sqrt(self.euler.gamma * q[-1, v.qp] / q[-1, v.qrho])

        dt = self.euler.C * self.euler.grid.dx / max(cs_l, cs_r)

        self.euler.estimate_dt()

        assert dt == self.euler.dt

    def test_result(self):

        self.euler.evolve(0.2, verbose=False)

        answer = np.array([0.9999999999884858, 0.9999999996587968, 0.9999999926055335,
                           0.9999998591088528, 0.9999976235288265, 0.9999639783206077,
                           0.999498225421092, 0.9934864722315098, 0.9577722709259177,
                           0.8876458746394962, 0.8022992747762369, 0.7152743208442388,
                           0.6378990088575718, 0.557368168194832, 0.4989238371211581,
                           0.44627949643723397, 0.4128638393142262, 0.4239756578324169,
                           0.4249987123598105, 0.4277550223279699, 0.41807251757339403,
                           0.3608284261210408, 0.2969979679601523, 0.26795929193227236,
                           0.2645912393860567, 0.2617028465028553, 0.2469803347669788,
                           0.18824882577593602, 0.13350768081641531, 0.12510813949387228,
                           0.12500108398757634, 0.12500001063923663])

        assert_array_almost_equal_nulp(self.euler.U[self.euler.grid.lo:self.euler.grid.hi+1, 0], answer)
