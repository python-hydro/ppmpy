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

        answer = np.array([0.9999999999884881, 0.9999999996588286, 0.9999999926058906,
                           0.9999998591096452, 0.9999976234562524, 0.9999639758819311,
                           0.9994981599213116, 0.9934862384469025, 0.9577687586790802,
                           0.8876320715576469, 0.8022784628832944, 0.7151087222085558,
                           0.6382123744023379, 0.5565369207854404, 0.4992889798518584,
                           0.4466432050040894, 0.412291961266306, 0.4230073970280775,
                           0.4249293521592982, 0.4280102442524921, 0.4185554188618217,
                           0.3615829765172491, 0.29726803937498475, 0.26788891602110243,
                           0.26456252570588495, 0.2616850260087769, 0.24698111981707102,
                           0.18822753583187274, 0.13348535295747904, 0.1251076982939915,
                           0.125001080305388, 0.1250000106083176])

        assert_array_almost_equal_nulp(self.euler.U[self.euler.grid.lo:self.euler.grid.hi+1, 0], answer, nulp=9)
