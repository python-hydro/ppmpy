from pytest import approx

from ppmpy import State, RiemannProblem


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

    def test_sod(self):

        q_l = State(rho=1.0, u=0.0, p=1.0)
        q_r = State(rho=0.125, u=0.0, p=0.1)

        rp = RiemannProblem(q_l, q_r, gamma=1.4)
        rp.find_star_state()

        assert rp.pstar == approx(0.30313017805064685, rel=1.e-12, abs=1.e-12)
        assert rp.ustar == approx(0.9274526200489498, rel=1.e-12, abs=1.e-12)

        qint = rp.sample_solution()

        assert qint.rho == approx(0.4263194281784952, rel=1.e-12, abs=1.e-12)
        assert qint.u == approx(0.9274526200489498, rel=1.e-12, abs=1.e-12)
        assert qint.p == approx(0.30313017805064685, rel=1.e-12, abs=1.e-12)
