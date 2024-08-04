import numpy as np

from ppmpy.euler import Euler


def sod(g, v, gamma, U):

    # setup initial conditions -- this is Sod's problem
    rho_l = 1.0
    u_l = 0.0
    p_l = 1.0
    rho_r = 0.125
    u_r = 0.0
    p_r = 0.1

    idx_l = g.x < 0.5
    idx_r = g.x >= 0.5

    U[idx_l, v.urho] = rho_l
    U[idx_l, v.umx] = rho_l * u_l
    U[idx_l, v.uener] = p_l/(gamma - 1.0) + 0.5 * rho_l * u_l**2

    U[idx_r, v.urho] = rho_r
    U[idx_r, v.umx] = rho_r * u_r
    U[idx_r, v.uener] = p_r/(gamma - 1.0) + 0.5 * rho_r * u_r**2


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

        # we used know the primitive variables for the sod problem, so
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
