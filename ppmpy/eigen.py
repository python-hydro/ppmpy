"""
Construct the eigenvalues and eigenvectors of the primitive variable
form of the Euler equations.
"""

import numpy as np


def eigen(rho, u, p, gamma):
    """Compute the left and right eigenvectors and the eigenvalues for
    the Euler equations.

    Parameters
    ----------
    rho : ndarray
        density
    u : ndarray
        velocity
    p : ndarray
        pressure
    gamma : float
        ratio of specific heats

    Returns
    -------
    ev : ndarray
        array of eigenvalues
    lvec : ndarray
        matrix of left eigenvectors, `lvec(iwave, :)` is
        the eigenvector for wave iwave
    rvec : ndarray
        matrix of right eigenvectors, `rvec(iwave, :)` is
        the eigenvector for wave iwave
    """

    # The Jacobian matrix for the primitive variable formulation of the
    # Euler equations is
    #
    #       / u   r   0   \
    #   A = | 0   u   1/r |
    #       \ 0  rc^2 u   /
    #
    # With the rows corresponding to rho, u, and p
    #
    # The eigenvalues are u - c, u, u + c

    cs = np.sqrt(gamma * p / rho)

    ev = np.array([u - cs, u, u + cs])

    # The left eigenvectors are
    #
    #   l1 =     ( 0,  -r/(2c),  1/(2c^2) )
    #   l2 =     ( 1,     0,     -1/c^2,  )
    #   l3 =     ( 0,   r/(2c),  1/(2c^2) )
    #

    lvec = np.array([[0.0, -0.5 * rho / cs, 0.5 / cs**2],  # u - c
                     [1.0, 0.0, -1.0 / cs**2],  # u
                     [0.0, 0.5 * rho / cs, 0.5 / cs**2]])  # u + c

    # The right eigenvectors are
    #
    #       /  1  \        / 1 \        /  1  \
    # r1 =  |-c/r |   r2 = | 0 |   r3 = | c/r |
    #       \ c^2 /        \ 0 /        \ c^2 /
    #

    rvec = np.array([[1.0, -cs / rho, cs**2],  # u - c
                     [1.0, 0.0, 0.0],  # u
                     [1.0, cs / rho, cs**2]])  # u + c

    return ev, lvec, rvec
