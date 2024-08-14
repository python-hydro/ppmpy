import numpy as np
from numpy.testing import assert_array_almost_equal_nulp

from ppmpy.euler import Euler
from ppmpy.gravity import constant_gravity
from ppmpy.initial_conditions import hse


class TestHSE:

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """

    def setup_method(self):
        """ this is run before each test """

        params = {"base_density": 1.0, "base_pressure": 1.0, "g_const": -1.0}

        self.hse = Euler(16, 0.5, init_cond=hse,
                         grav_func=constant_gravity,
                         use_hse_reconstruction=True,
                         bc_left_type="reflect", bc_right_type="reflect", params=params)

        self.nohse = Euler(16, 0.5, init_cond=hse,
                         grav_func=constant_gravity,
                         use_hse_reconstruction=False,
                         bc_left_type="reflect", bc_right_type="reflect", params=params)

    def teardown_method(self):
        """ this is run after each test """

    def test_hse_pressure_reconstruction(self):

        self.hse.construct_parabola()

        # the key here is that the left parabola state for the first interior
        # zone (4, starting from 0) is high -- it is the dp/dr we expect

        ans = np.array([0.8034735033053615, 0.8553105035186105, 0.9104918263262627,
                        0.9692332344763441, 0.9995217730537299, 0.9389446958989583,
                        0.882038956753567, 0.8285820502836542, 0.7783649563270688,
                        0.7311913226102768, 0.6868766969975327, 0.6452478062704096,
                        0.6061418786176574, 0.5694060071862842, 0.5348965522052972,
                        0.5024785793443701, 0.4720253321113781, 0.4434177362258399,
                        0.4165439340303344, 0.3912988471194051, 0.379441306297605,
                        0.40392139057486975, 0.42998083512808716, 0.457721534168609])

        assert_array_almost_equal_nulp(self.hse.q_parabola[self.hse.v.qp].am, ans)

    def test_standard_pressure_reconstruction(self):

        self.nohse.construct_parabola()

        ans = np.array([0.8034735033053615, 0.8553105035186105, 0.8327632963498921,
                        0.9692332344763441, 0.9692332344763441, 0.9493560913144479,
                        0.8823257975943671, 0.8288515068310723, 0.7786180821746437,
                        0.7314291074973925, 0.6871000706793687, 0.6454576421533463,
                        0.6063389971743558, 0.5695911791637885, 0.5350705016387105,
                        0.5026419863878796, 0.472178835697705, 0.44356193656451076,
                        0.41667939495454037, 0.379441306297605, 0.379441306297605,
                        0.40392139057486975, 0.42998083512808716, 0.457721534168609])

        assert_array_almost_equal_nulp(self.nohse.q_parabola[self.nohse.v.qp].am, ans)
