"""
A 1D finite-volume grid class
"""


import numpy as np
import matplotlib.pyplot as plt


class FVGrid:
    """The main finite-volume grid class for holding our fluid state."""

    def __init__(self, nx, ng, *,
                 xmin=0.0, xmax=1.0):

        self.xmin = xmin
        self.xmax = xmax
        self.ng = ng
        self.nx = nx

        self.nq = self.nx + 2*self.ng

        self.lo = ng
        self.hi = ng+nx-1

        # physical coords -- cell-centered
        self.dx = (xmax - xmin)/(nx)
        self.xl = xmin + (np.arange(nx+2*ng)-ng)*self.dx
        self.xr = xmin + (np.arange(nx+2*ng)-ng+1.0)*self.dx
        self.x = xmin + (np.arange(nx+2*ng)-ng+0.5)*self.dx

    def scratch_array(self, nc=1):
        """ return a scratch array dimensioned for our grid """
        return np.squeeze(np.zeros((self.nq, nc), dtype=np.float64))

    def ghost_fill(self, atmp, *,
                   bc_left_type="outflow", bc_right_type="outflow"):
        """fill all ghost cells with zero-gradient boundary
        conditions"""

        # left

        if bc_left_type == "outflow":
            atmp[0:self.lo] = atmp[self.lo]

        elif bc_left_type == "periodic":
            atmp[0:self.lo] = atmp[self.hi-self.ng+1:self.hi+1]

        elif bc_left_type == "reflect":
            atmp[0:self.lo] = np.flip(atmp[self.lo:self.lo+self.ng])

        elif bc_left_type == "reflect-odd":
            atmp[0:self.lo] = -np.flip(atmp[self.lo:self.lo+self.ng])
        else:
            raise ValueError("invalid boundary condition")

        # right

        if bc_right_type == "outflow":
            atmp[self.hi+1:] = atmp[self.hi]

        elif bc_right_type == "periodic":
            atmp[self.hi+1:] = atmp[self.lo:self.lo+self.ng]

        elif bc_right_type == "reflect":
            atmp[self.hi+1:] = np.flip(atmp[self.hi-self.ng+1:self.hi+1])
        elif bc_right_type == "reflect-odd":
            atmp[self.hi+1:] = -np.flip(atmp[self.hi-self.ng+1:self.hi+1])
        else:
            raise ValueError("invalid boundary condition")

    def norm(self, e):
        """compute the L2 norm of array e defined on this grid"""

        assert len(e) == self.nq
        return np.sqrt(self.dx * np.sum(e[self.lo:self.hi+1]**2))

    def coarsen(self, fdata):
        """coarsen an array fine defined on this grid down to a grid
        with 1/2 the number of zones (but the same number of ghost
        cells

        """

        assert self.nx % 2 == 0

        cgrid = FVGrid(self.nx//2, self.ng)
        cdata = cgrid.scratch_array()
        cdata[cgrid.lo:cgrid.hi+1] = 0.5 * (fdata[self.lo:self.hi:2] +
                                            fdata[self.lo+1:self.hi+1:2])

        return cgrid, cdata

    def draw(self):
        """Draw a finite volume representation of the grid and return the
        figure and axis objects"""

        fig, ax = plt.subplots()

        nstart = self.lo - self.ng
        nstop = self.hi + self.ng

        ax.plot([self.xl[nstart], self.xr[nstop]], [0, 0], color="0.25", lw=2)

        # draw edges

        for n in range(nstart, nstop+1):
            if n == self.lo:
                ax.plot([self.xl[n], self.xl[n]], [0, 1.0], color="0.25", lw=4)
            else:
                ax.plot([self.xl[n], self.xl[n]], [0, 1.0], color="0.25", lw=2)

            if n == self.hi:
                ax.plot([self.xr[n], self.xr[n]], [0, 1.0], color="0.25", lw=4)

        ax.plot([self.xr[nstop], self.xr[nstop]], [0, 1.0], color="0.25", lw=2)

        fig.set_size_inches((self.nx + 2 * self.ng), 1)
        ax.axis("off")
        return fig, ax
