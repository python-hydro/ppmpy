"""
A 1D finite-volume grid class
"""


import numpy as np
import matplotlib.pyplot as plt


class GridPlot:
    """a container to hold info about the grid figure

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
         the figure object.
    ax : matplotlib.pyplot.Axes
         the axis object.
    lo_index : int, optional
         the 0-based zone index for the left edge of the plot.
    hi_index : int, optional
         the 0-based zone index for the right edge of the plot.
    """

    def __init__(self, *, fig=None, ax=None, lo_index=None, hi_index=None):
        self.fig = fig
        self.ax = ax
        self.lo_index = lo_index
        self.hi_index = hi_index

    def show_fig(self):
        """return the Figure object

        Returns
        -------
        matplotlib.pyplot.Figure
        """
        return self.fig


class FVGrid:
    """The main finite-volume grid class for holding our fluid state.

    Parameters
    ----------
    nx : int
        number of zones on the grid.
    ng : int
        number of ghost cells on each end of the grid.
    xmin : float, optional
        minimum x-coordinate.
    xmax : float, optional
        maximum x-coordinate.
    """

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
        """ return a scratch array dimensioned for our grid

        Parameters
        ----------
        nc : int
            number of components.

        Returns
        -------
        ndarray
        """
        return np.squeeze(np.zeros((self.nq, nc), dtype=np.float64))

    def ghost_fill(self, atmp, *,
                   bc_left_type="outflow", bc_right_type="outflow"):
        """fill all ghost cells with zero-gradient boundary
        conditions.

        Parameters
        ----------
        atmp : ndarray
             the array of data defined on the `FVGrid` to fill ghost cells in
        bc_left_type : string
             boundary condition type on the left edge of the domain.
        bc_right_type : string
             boundary condition type on the left edge of the domain.
        """

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
        """compute the L2 norm of array e defined on this grid

        Parameters
        ----------
        e : ndarray
             array of data defined on the grid whose norm we want to
             compute

        Returns
        -------
        float
        """

        assert len(e) == self.nq
        return np.sqrt(self.dx * np.sum(e[self.lo:self.hi+1]**2))

    def coarsen(self, fdata):
        """coarsen an array fine defined on this grid down to a grid
        with 1/2 the number of zones (but the same number of ghost
        cells.

        Parameters
        ----------
        fdata : ndarray
            The data defined on this `FVGrid` object.

        Returns
        -------
        FVGrid
            The coarse grid object.
        ndarray
            The coarsened data on the coarse grid.

        """

        assert self.nx % 2 == 0

        cgrid = FVGrid(self.nx//2, self.ng)
        cdata = cgrid.scratch_array()
        cdata[cgrid.lo:cgrid.hi+1] = 0.5 * (fdata[self.lo:self.hi:2] +
                                            fdata[self.lo+1:self.hi+1:2])

        return cgrid, cdata

    def draw(self, *, lo_index=None, hi_index=None, stretch=1):
        """Draw a finite volume representation of the grid and return the
        figure and axis objects


        Parameters
        ----------
        lo_index : int
            0-based zone index to start the grid plot.
        hi_index : int
            0-based zone index to end the grid plot.
        stretch : float
            factor by which to stretch the vertical axis

        Returns
        -------
        GridPlot
        """

        fig, ax = plt.subplots()

        if lo_index is None:
            nstart = self.lo - self.ng
        else:
            nstart = lo_index

        if hi_index is None:
            nstop = self.hi + self.ng
        else:
            nstop = hi_index

        assert nstop > nstart

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

        nzones = nstop - nstart + 1

        fig.set_size_inches((nzones, stretch))
        ax.axis("off")

        return GridPlot(fig=fig, ax=ax, lo_index=nstart, hi_index=nstop)
