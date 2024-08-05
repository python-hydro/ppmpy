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

    def draw(self):
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
