from ppmpy.euler import FluidVars


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

    def test_vars(self):

        assert self.v.qrho == self.v.urho
        assert self.v.qu == self.v.umx
        assert self.v.qp == self.v.uener
