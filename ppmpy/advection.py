"""a simple linear advection solver for testing"""


import numpy as np

from ppmpy import FVGrid, PPMInterpolant


def states(grid, a, u, dt):
    """Compute the left and right interface states via PPM
    reconstruction and tracing under the profile

    """

    a_ppm = PPMInterpolant(grid, a)
    sigma = u * dt / grid.dx

    # Im is the left side of the zone, Ip is the right side of the zone
    Im, Ip = a_ppm.integrate(sigma)

    # convert these to be left and right of an interface
    a_left = grid.scratch_array()
    a_right = grid.scratch_array()

    a_right[:] = Im[:]
    a_left[1:] = Ip[:-1]

    return a_left, a_right


def advection(nx, u, C, *, num_periods=1, init_cond=None):
    """Evolve the linear advection equation using nx zones,
    a velocity u, and a CFL number C.  You need to provide
    a function to compute the initial conditions, init_cond(g),
    where g is a FVGrid object"""

    g = FVGrid(nx, ng=3)

    t_period = (g.xmax - g.xmin) / np.abs(u)
    tmax = num_periods * t_period

    # setup the initial conditions
    a_init = init_cond(g)

    # compute the timestep
    dt = C * g.dx / np.abs(u)

    a = g.scratch_array()
    a[:] = a_init[:]

    t = 0.0
    while t < tmax:
        if t + dt > tmax:
            dt = tmax - t

        g.ghost_fill(a, bc_left_type="periodic", bc_right_type="periodic")

        # get the interface states
        a_left, a_right = states(g, a, u, dt)

        # solve the Riemann problem
        a_int = np.where(u > 0, a_left, a_right)

        # do the update
        a[g.lo:g.hi+1] += -dt * u * (a_int[g.lo+1:g.hi+2] - a_int[g.lo:g.hi+1]) / g.dx
        t += dt

    g.ghost_fill(a, bc_left_type="periodic", bc_right_type="periodic")

    return g, a


def tophat(g):
    """Simple tophat initial conditions"""

    a = g.scratch_array()
    a[:] = 0.0
    a[np.logical_and(g.x >= 1./3., g.x <= 2./3.)] = 1.0
    return a


def sine(g):
    """A sine wave initial condition (shifted so a > 0 everywhere)"""

    a = g.scratch_array()
    a[:] = 1.0 + 0.5 * np.sin(2.0 * np.pi * g.x)
    return a
