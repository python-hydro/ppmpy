"""
A 1D hydrodynamics code that implements the piecewise parabolic method.
This is intended to be used for testing and prototyping variations on
the core PPM algorithm.
"""

from ._version import version

__version__ = version


from .eigen import eigen
from .grid import FVGrid
from .reconstruction import PPMInterpolant, flattening_coefficient
from .riemann_exact import RiemannProblem, State

from .euler import Euler
