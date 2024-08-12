"""
Routines for doing the reconstruction of the interface states from
cell-average data.
"""


import numpy as np


def flattening_coefficient(grid, p, u):
    """Compute the flattening coefficient, chi, for a shock.  This works
    by looking for compression and a steep pressure profile.  chi = 1
    means no flattening"""

    # see Saltzman 1994 for an implementation

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
    the PPM reconstruction"""

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
        """Draw the parabolas in each zone on the axes ax."""

        ilo = max(gp.lo_index, self.grid.lo-1)
        ihi = min(gp.hi_index, self.grid.hi+1)

        if scale is None:
            scale = np.max(self.a[ilo:ihi+1])

        for n in range(ilo, ihi+1):
            x = np.linspace(self.grid.xl[n], self.grid.xr[n], 50)
            xi = (x - self.grid.xl[n]) / self.grid.dx
            a = self.am[n] + xi*(self.ap[n] - self.am[n] + self.a6[n] * (1.0-xi))
            gp.ax.plot(x, a/scale, color="C1")

    def mark_cubic(self, gp, *, scale=None):
        """Mark the location of the initial interface states from the
        cubic interpolant on the axes ax."""

        ilo = max(gp.lo_index-1, self.grid.lo-2)
        ihi = min(gp.hi_index, self.grid.hi+1)

        if scale is None:
            scale = np.max(self.a[ilo:ihi+1])

        gp.ax.scatter(self.grid.xr[ilo:ihi+1], self.aint[ilo:ihi+1] / scale,
                      marker="x", zorder=10)
