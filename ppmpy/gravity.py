"""
Functions that can provide the gravitational acceleration
"""


import numpy as np


def constant_gravity(grid, rho, params):
    g = grid.scratch_array()
    g[grid.lo:grid.hi+1] = params["g_const"]
    return g
