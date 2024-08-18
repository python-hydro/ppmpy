"""Some standard initial conditions for testing the Euler solver"""


import numpy as np


def sod(euler):
    """Initial conditions for the classic Sod shock tube problem

    Parameters
    ----------
    euler : Euler
        the Euler simulation object

    Returns
    -------
    None
    """

    gamma = euler.params["gamma"]
    g = euler.grid

    # setup initial conditions -- this is Sod's problem
    rho_l = 1.0
    u_l = 0.0
    p_l = 1.0
    rho_r = 0.125
    u_r = 0.0
    p_r = 0.1

    idx_l = g.x < 0.5
    idx_r = g.x >= 0.5

    euler.U[idx_l, euler.v.urho] = rho_l
    euler.U[idx_l, euler.v.umx] = rho_l * u_l
    euler.U[idx_l, euler.v.uener] = p_l/(gamma - 1.0) + 0.5 * rho_l * u_l**2

    euler.U[idx_r, euler.v.urho] = rho_r
    euler.U[idx_r, euler.v.umx] = rho_r * u_r
    euler.U[idx_r, euler.v.uener] = p_r/(gamma - 1.0) + 0.5 * rho_r * u_r**2


def acoustic_pulse(euler):
    """The acoustic pulse problem from McCorquodale & Colella 2011

    Parameters
    ----------
    euler : Euler
        the Euler simulation object

    Returns
    -------
    None
    """

    gamma = euler.params["gamma"]

    xcenter = 0.5 * (euler.grid.xmin + euler.grid.xmax)

    rho0 = 1.4
    delta_rho = 0.14

    r = np.abs(euler.grid.x - xcenter)

    rho = np.where(r <= 0.5, rho0 + delta_rho * np.exp(-16.0 * r**2) * np.cos(np.pi * r)**6, rho0)
    p = (rho / rho0)**gamma
    u = 0.0

    euler.U[:, euler.v.urho] = rho
    euler.U[:, euler.v.umx] = rho * u
    euler.U[:, euler.v.uener] = p / (gamma - 1.0) + 0.5 * rho * u**2


def hse(euler):
    """An isothermal hydrostatic atmosphere.

    The following parameters should be set in the Euler initialization:

    *  `base_density` : the density at the lower boundary
    * `base_pressure` : the pressure at the lower boundary
    * `g_const` : the gravitational acceleration
    * `verbose` : output the HSE error

    Parameters
    ----------
    euler : Euler
        the Euler simulation object

    Returns
    -------
    None
    """

    rho_base = euler.params["base_density"]
    pres_base = euler.params["base_pressure"]
    g = euler.params["g_const"]
    gamma = euler.params["gamma"]

    verbose = euler.params.get("verbose", False)

    # we will assume we are isothermal and constant composition.  In that case,
    # p/rho = constant
    A = pres_base / rho_base

    # we will discretize HSE as second-order
    # p_{i+1} = p_i + dx / 2 (rho_i + rho_{i+1} g
    # but we can write p_{i+1} = A rho_{i+1} and solve for rho_{i+1}

    p = euler.grid.scratch_array()
    rho = euler.grid.scratch_array()

    # we want the base conditions to be at the lower boundary.  We will
    # set the conditions in the first zone center from the analytic expression:
    # P = P_base e^{-z/H}

    H = pres_base / rho_base / np.abs(g)

    p[euler.grid.lo] = pres_base * np.exp(-euler.grid.x[euler.grid.lo] / H)
    rho[euler.grid.lo] = rho_base * np.exp(-euler.grid.x[euler.grid.lo] / H)

    for i in range(euler.grid.lo+1, euler.grid.hi+1):
        rho[i] = (p[i-1] + 0.5 * euler.grid.dx * rho[i-1] * g) / (A - 0.5 * euler.grid.dx * g)
        p[i] = A * rho[i]

    # now check HSE
    if verbose:
        max_err = 0.0
        for i in range(euler.grid.lo+1, euler.grid.hi+1):
            dpdr = (p[i] - p[i-1]) / euler.grid.dx
            rhog = 0.5 * (rho[i] + rho[i-1]) * g
            err = np.abs(dpdr - rhog) / np.abs(rhog)
            max_err = max(max_err, err)

        print(f"max err = {max_err}")

    # now fill the conserved variables
    euler.U[:, euler.v.urho] = rho[:]
    euler.U[:, euler.v.umx] = 0.0
    euler.U[:, euler.v.uener] = p[:] / (gamma - 1.0)
