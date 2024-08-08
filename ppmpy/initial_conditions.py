"""Some standard initial conditions for testing the Euler solver"""


import numpy as np


def sod(g, v, gamma, U):
    """Initial conditions for the classic Sod shock tube problem"""

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


def acoustic_pulse(g, v, gamma, U):
    """The acoustic pulse problem from McCorquodale & Colella 2011"""

    xcenter = 0.5 * (g.xmin + g.xmax)

    rho0 = 1.4
    delta_rho = 0.14

    r = np.abs(g.x - xcenter)

    rho = np.where(r <= 0.5, rho0 + delta_rho * np.exp(-16.0 * r**2) * np.cos(np.pi * r)**6, rho0)
    p = (rho / rho0)**gamma
    u = 0.0

    U[:, v.urho] = rho
    U[:, v.umx] = rho * u
    U[:, v.uener] = p / (gamma - 1.0) + 0.5 * rho * u**2
