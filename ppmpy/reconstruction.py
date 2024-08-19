"""
Routines for doing the reconstruction of the interface states from
cell-average data.
"""


import numpy as np


def flattening_coefficient(grid, p, u):
    """Compute the flattening coefficient, chi, for a shock.  This works
    by looking for compression and a steep pressure profile.  chi = 1
    means no flattening.  This follows Saltzman (1994).

    Parameters
    ----------
    grid : FVGrid
         the grid object.
    p : ndarray
         the pressure defined on the grid.
    u : ndarray
         the velocity defined on the grid

    Returns
    -------
    ndarray
    """

    smallp = 1.e-10
    z0 = 0.75
    z1 = 0.85
    delta = 0.33

    # dp = p_{i+1} - p_{i-1}
    dp = grid.scratch_array()
    dp[grid.lo-2:grid.hi+3] = p[grid.lo-1:grid.hi+4] - p[grid.lo-3:grid.hi+2]

    # dp2 = p_{i+2} - p_{i-2}
    dp2 = grid.scratch_array()
    dp2[grid.lo-2:grid.hi+3] = p[grid.lo:grid.hi+5] - p[grid.lo-4:grid.hi+1]

    z = np.abs(dp) / np.clip(np.abs(dp2), smallp, None)

    chi = np.clip(1.0 - (z - z0) / (z1 - z0), 0.0, 1.0)

    # du = u_{i+1} - u_{i-1}
    du = grid.scratch_array()
    du[grid.lo-2:grid.hi+3] = u[grid.lo-1:grid.hi+4] - u[grid.lo-3:grid.hi+2]

    # construct |dp_i| / min(p_{i+1}, p_{i-1})
    test = grid.scratch_array()
    test[grid.lo-2:grid.hi+3] = np.abs(dp[grid.lo-2:grid.hi+3]) / \
        np.minimum(p[grid.lo-3:grid.hi+2],
                   p[grid.lo-1:grid.hi+4]) > delta

    chi = np.where(np.logical_and(test, du < 0), chi, 1.0)

    # combine chi with the neighbor, following the sign of the pressure jump
    chi[grid.lo-1:grid.hi+2] = np.where(dp[grid.lo-1:grid.hi+2] > 0,
                                        np.minimum(chi[grid.lo-1:grid.hi+2],
                                                   chi[grid.lo-2:grid.hi+1]),
                                        np.minimum(chi[grid.lo-1:grid.hi+2],
                                                   chi[grid.lo:grid.hi+3]))
    return chi


class PPMInterpolant:
    """Given a fluid variable a defined on the FVGrid grid, perform
    the PPM reconstruction

    Parameters
    ----------
    grid : FVGrid
        the grid object.
    a : ndarray
        the data for a single component defined on the grid.
    limit : bool, optional
        do we use limiting?
    chi_flat : ndarray, optional
        the flattening coefficient.
    """

    def __init__(self, grid, a, *, limit=True, chi_flat=None):
        self.grid = grid
        assert grid.ng >= 3

        self.a = a
        self.limit = limit
        self.chi_flat = chi_flat

        self.aint = grid.scratch_array()

        self.ap = grid.scratch_array()
        self.am = grid.scratch_array()
        self.a6 = grid.scratch_array()

        self.initialized = False

    def construct_parabola(self):
        """compute the coefficients of a parabolic interpolant for the
        data in each zone.  This will give am, the parabola value on
        the left edge of a zone, ap, the parabola value on the right
        edge of the zone, and a6, a measure of the curvature of the
        parabola.

        """

        # first do the cubic interpolation in zones in all but the last ghost cell
        # we will be getting a_{i+1/2}

        # the state will initially be defined on ib:ie+1
        ib = self.grid.lo-2
        ie = self.grid.hi+1

        da0 = self.grid.scratch_array()
        dap = self.grid.scratch_array()

        # 1/2 (a_{i+1} - a_{i-1})
        da0[ib:ie+1] = 0.5 * (self.a[ib+1:ie+2] - self.a[ib-1:ie])

        # 1/2 (a_{i+2} - a_{i})
        dap[ib:ie+1] = 0.5 * (self.a[ib+2:ie+3] - self.a[ib:ie+1])

        if self.limit:
            # van-Leer slopes
            dl = self.grid.scratch_array()
            dr = self.grid.scratch_array()
            dr[ib:ie+1] = self.a[ib+1:ie+2] - self.a[ib:ie+1]
            dl[ib:ie+1] = self.a[ib:ie+1] - self.a[ib-1:ie]

            da0 = np.where(dl * dr < 0, 0.0,
                           np.sign(da0) * np.minimum(np.abs(da0),
                                                     2.0 * np.minimum(np.abs(dl),
                                                                      np.abs(dr))))

            dl[:] = dr[:]
            dr[ib:ie+1] = self.a[ib+2:ie+3] - self.a[ib+1:ie+2]

            dap = np.where(dl * dr < 0, 0.0,
                           np.sign(dap) * np.minimum(np.abs(dap),
                                                     2.0 * np.minimum(np.abs(dl),
                                                                      np.abs(dr))))

        # cubic
        self.aint[ib:ie+1] = 0.5 * (self.a[ib:ie+1] + self.a[ib+1:ie+2]) - \
                             (1.0 / 6.0) * (dap[ib:ie+1] - da0[ib:ie+1])

        # now the parabola coefficients
        self.ap[:] = self.aint[:]
        self.am[1:] = self.ap[:-1]

        if self.limit:

            # now we work on each zone + 1 ghost cell and limit the parabola
            # coefficients as needed.  At the end, this will be valid on
            # lo-1:hi+2

            test = (self.ap - self.a) * (self.a - self.am) < 0

            da = self.ap - self.am
            testm = da * (self.a - 0.5 * (self.am + self.ap)) > da**2 / 6
            self.am[:] = np.where(test, self.a, np.where(testm, 3.0*self.a - 2.0*self.ap, self.am))

            testp = -da**2 / 6 > da * (self.a - 0.5 * (self.am + self.ap))
            self.ap[:] = np.where(test, self.a, np.where(testp, 3.0*self.a - 2.0*self.am, self.ap))

        if self.chi_flat is not None:
            self.am[:] = (1.0 - self.chi_flat[:]) * self.a[:] + self.chi_flat[:] * self.am[:]
            self.ap[:] = (1.0 - self.chi_flat[:]) * self.a[:] + self.chi_flat[:] * self.ap[:]

        self.a6 = 6.0 * self.a - 3.0 * (self.am + self.ap)

        self.initialized = True

    def integrate(self, sigma):
        """integrate under the parabola to the left edge (Im) and
        right edge (Ip) for a fraction sigma = lambda * dt / dx,
        where lambda is the characteristic speed.  If sigma is not
        moving toward the edge, then we us the limit of the parabola
        in that direction.

        Parameters
        ----------
        sigma : ndarray
            the dimensionless wavespeed (lambda dt / dx)

        Returns
        -------
        Im : ndarray
            the integral under the parabola from the left edge through a
            distance sigma.
        Ip : ndarray
            the integral under the parabola from the right edge through a
            distance sigma.
        """

        if not self.initialized:
            self.construct_parabola()

        Ip = self.grid.scratch_array()
        Ip[:] = np.where(sigma <= 0.0, self.ap,
                         self.ap - 0.5 * np.abs(sigma) *
                           (self.ap - self.am - (1.0 - (2.0/3.0) * np.abs(sigma)) * self.a6))

        Im = self.grid.scratch_array()
        Im[:] = np.where(sigma >= 0.0, self.am,
                         self.am + 0.5 * np.abs(sigma) *
                           (self.ap - self.am + (1.0 - (2.0/3.0) * np.abs(sigma)) * self.a6))

        return Im, Ip

    def draw_parabola(self, gp, *, scale=None):
        """Draw the parabolas in each zone on the axes ax.

        Parameters
        ----------

        gp : GridPlot
            the grid plot object
        scale : float, optional
            normalization factor (default is maximum data value)
        """

        ilo = max(gp.lo_index, self.grid.lo-1)
        ihi = min(gp.hi_index, self.grid.hi+1)

        if scale is None:
            scale = np.max(self.a[ilo:ihi+1])

        for n in range(ilo, ihi+1):
            x = np.linspace(self.grid.xl[n], self.grid.xr[n], 50)
            xi = (x - self.grid.xl[n]) / self.grid.dx
            a = self.am[n] + xi*(self.ap[n] - self.am[n] + self.a6[n] * (1.0-xi))
            gp.ax.plot(x, a/scale, color="C1", lw=2)

    def mark_cubic(self, gp, *, scale=None):
        """Mark the location of the initial interface states from the
        cubic interpolant on the axes ax.

        Parameters
        ----------

        gp : GridPlot
            the grid plot object
        scale : float, optional
            normalization factor (default is maximum data value)

        """

        ilo = max(gp.lo_index-1, self.grid.lo-2)
        ihi = min(gp.hi_index, self.grid.hi+1)

        if scale is None:
            # we need to scale with the same limits as the centers
            scale = np.max(self.a[ilo+1:ihi+1])

        gp.ax.scatter(self.grid.xr[ilo:ihi+1], self.aint[ilo:ihi+1] / scale,
                      marker="x", zorder=10)


class HSEPPMInterpolant(PPMInterpolant):
    """PPM interpolation for pressure that subtracts off HSE

    Parameters
    ----------
    grid : FVGrid
        the grid object.
    p : ndarray
        the pressure defined on the grid.
    rho : ndarray
        the density defined on the grid.
    g : ndarray
        the gravitational acceleration defined on the grid
    limit : bool, optional
        use limiting?
    chi_flat : ndarray, optional
        the flattening coefficient.
    """

    def __init__(self, grid, p, rho, g, *, limit=True, chi_flat=None, leave_as_perturbation=False):

        super().__init__(grid, p, limit=limit, chi_flat=chi_flat)

        self.rho = rho
        self.g = g

        self.p_hse_m = None
        self.p_hse_p = None

        self.leave_as_perturbation = leave_as_perturbation

    def construct_parabola(self):
        """compute the coefficients of a parabolic interpolant for the
        pressure in each zone by subtracting off HSE.  This will give
        am, the parabola value on the left edge of a zone, ap, the
        parabola value on the right edge of the zone, and a6, a
        measure of the curvature of the parabola.

        """

        # first do the cubic interpolation in zones in all but the last ghost cell
        # we will be getting a_{i+1/2}

        # for easy indexing
        im2 = 0
        im1 = 1
        i0 = 2
        ip1 = 3
        ip2 = 4

        # the state will initially be defined on ib:ie
        ib = self.grid.lo-2
        ie = self.grid.hi+1

        for i in range(ib, ie+1):

            p = np.array([self.a[i-2], self.a[i-1], self.a[i], self.a[i+1], self.a[i+2]])
            rho = np.array([self.rho[i-2], self.rho[i-1], self.rho[i], self.rho[i+1], self.rho[i+2]])
            src = np.array([self.g[i-2], self.g[i-1], self.g[i], self.g[i+1], self.g[i+2]])

            p_hse = np.zeros(5)

            p_hse[i0] = p[i0]
            p_hse[ip1] = p_hse[i0] + 0.25*self.grid.dx * (rho[i0] + rho[ip1]) * (src[i0] + src[ip1])
            p_hse[ip2] = p_hse[ip1] + 0.25*self.grid.dx * (rho[ip1] + rho[ip2]) * (src[ip1] + src[ip2])
            p_hse[im1] = p_hse[i0] - 0.25*self.grid.dx * (rho[i0] + rho[im1]) * (src[i0] + src[im1])
            p_hse[im2] = p_hse[im1] - 0.25*self.grid.dx * (rho[im1] + rho[im2]) * (src[im1] + src[im2])

            tp = p - p_hse

            # 1/2 (a_{i+1} - a_{i-1})
            da0 = 0.5 * (tp[ip1] - tp[im1])

            # 1/2 (a_{i+2} - a_{i})
            dap = 0.5 * (tp[ip2] - tp[i0])

            if self.limit:
                # van-Leer slopes
                dr = tp[ip1] - tp[i0]
                dl = tp[i0] - tp[im1]

                if dl * dr > 0.0:
                    da0 = np.sign(da0) * min(np.abs(da0),
                                             2.0 * min(np.abs(dl), np.abs(dr)))
                else:
                    da0 = 0.0

                dl = dr
                dr = tp[ip2] - tp[ip1]

                if dl * dr > 0.0:
                    dap = np.sign(dap) * min(np.abs(dap),
                                             2.0 * min(np.abs(dl), np.abs(dr)))
                else:
                    dap = 0.0

            # cubic
            self.aint[i] = 0.5 * (tp[i0] + tp[ip1]) - (1.0 / 6.0) * (dap - da0)

        # now the parabola coefficients
        self.ap[:] = self.aint[:]
        self.am[1:] = self.ap[:-1]

        p_hse = np.zeros_like(self.a)

        if self.limit:

            # now we work on each zone + 1 ghost cell and limit the parabola
            # coefficients as needed.  At the end, this will be valid on
            # lo-1:hi+2

            test = (self.ap - p_hse) * (p_hse - self.am) < 0

            da = self.ap - self.am
            testm = da * (p_hse - 0.5 * (self.am + self.ap)) > da**2 / 6
            self.am[:] = np.where(test, p_hse, np.where(testm, 3.0*p_hse - 2.0*self.ap, self.am))

            testp = -da**2 / 6 > da * (p_hse - 0.5 * (self.am + self.ap))
            self.ap[:] = np.where(test, p_hse, np.where(testp, 3.0*p_hse - 2.0*self.am, self.ap))

        if self.chi_flat is not None:
            self.am[:] = (1.0 - self.chi_flat[:]) * p_hse[:] + self.chi_flat[:] * self.am[:]
            self.ap[:] = (1.0 - self.chi_flat[:]) * p_hse[:] + self.chi_flat[:] * self.ap[:]

        self.p_hse_p = self.a[:] + 0.5 * self.grid.dx * self.rho[:] * self.g[:]
        self.p_hse_m = self.a[:] - 0.5 * self.grid.dx * self.rho[:] * self.g[:]

        if not self.leave_as_perturbation:

            # finally, add back in the HSE correction
            self.ap[:] += self.p_hse_p[:]
            self.am[:] += self.p_hse_m[:]

            self.a6 = 6.0 * self.a - 3.0 * (self.am + self.ap)

        else:
            # the cell-center state here is zero, since we subtract off pressur
            self.a6 = - 3.0 * (self.am + self.ap)

        self.initialized = True
