import numpy as np
from numpy.testing import assert_array_almost_equal_nulp

from ppmpy.euler import Euler, FluidVars
from ppmpy.initial_conditions import sod, acoustic_pulse


class TestFluidVars:

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """

    def setup_method(self):
        """ this is run before each test """

        self.v = FluidVars()

    def teardown_method(self):
        """ this is run after each test """

    def test_indices(self):
        assert self.v.urho == self.v.qrho
        assert self.v.umx == self.v.qu
        assert self.v.uener == self.v.qp


class TestEuler:

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

    def test_estimate_dt(self):

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


class TestEulerReconstruction:

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """

    def setup_method(self):
        """ this is run before each test """

        self.euler = Euler(16, 0.5, init_cond=acoustic_pulse)
        self.euler.estimate_dt()

    def teardown_method(self):
        """ this is run after each test """

    def test_construct_parabola(self):

        self.euler.construct_parabola()

        ans = np.array([0.0000000000000000e+00, 0.0000000000000000e+00,
                        0.0000000000000000e+00, 0.0000000000000000e+00,
                        0.0000000000000000e+00, -1.4925725604797435e-05,
                        -5.5892314022898404e-04, -3.1886159900746947e-03,
                        -1.1509891420676155e-02, -1.5395301015125540e-02,
                        -7.3670928260867186e-03, 0.0000000000000000e+00,
                        0.0000000000000000e+00, -7.3670928260867186e-03,
                        -1.5395301015125540e-02, -1.1509891420676155e-02,
                        -3.1886159900746947e-03, -5.5892314022898404e-04,
                        -1.4925725604797435e-05, 0.0000000000000000e+00,
                        0.0000000000000000e+00, 0.0000000000000000e+00,
                        0.0000000000000000e+00, 0.0000000000000000e+00])

        assert_array_almost_equal_nulp(self.euler.q_parabola[0].a6, ans)

        ans2 = np.array([0.0000000000000000e+00, 0.0000000000000000e+00,
                         -8.8817841970012523e-16, 0.0000000000000000e+00,
                         -8.8817841970012523e-16, -1.4925738349269579e-05,
                         -5.5894233322995035e-04, -3.1567942491133039e-03,
                         -1.1552096619801056e-02, -1.5947978794340401e-02,
                         -8.1712935823112787e-03, 0.0000000000000000e+00,
                         0.0000000000000000e+00, -8.1712935823112787e-03,
                         -1.5947978794340401e-02, -1.1552096619801056e-02,
                         -3.1567942491133039e-03, -5.5894233322995035e-04,
                         -1.4925738349269579e-05, -8.8817841970012523e-16,
                         0.0000000000000000e+00, -8.8817841970012523e-16,
                         0.0000000000000000e+00, 0.0000000000000000e+00])

        assert_array_almost_equal_nulp(self.euler.q_parabola[2].a6, ans2)
