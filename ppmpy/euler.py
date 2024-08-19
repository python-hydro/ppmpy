"""
The main Euler solver classes.
"""


import matplotlib.pyplot as plt
import numpy as np

from ppmpy.eigen import eigen
from ppmpy.grid import FVGrid
from ppmpy.reconstruction import PPMInterpolant, HSEPPMInterpolant, flattening_coefficient
from ppmpy.riemann_exact import RiemannProblem, State


class FluidVars:
    """A simple container that holds the integer indices we will use to
    refer to the different fluid components"""

    def __init__(self):
        self.nvar = 3

        # conserved variables
        self.urho = 0
        self.umx = 1
        self.uener = 2

        # primitive variables
        self.qrho = 0
        self.qu = 1
        self.qp = 2

        self.prim_names = [""] * self.nvar
        self.prim_names[self.qrho] = "rho"
        self.prim_names[self.qu] = "u"
        self.prim_names[self.qp] = "p"


class Euler:
    """A 1D compressible Euler solver using the piecewise parabolic method
    (PPM), following the original Colella & Woodward ideas

    Parameters
    ----------
    nx : int
        the number of zones
    C : float
        the CFL number
    fixed_dt : float, optional
        a fixed timestep to use for every step.  In this case we
        do not estimate the timestep using the CFL criteria.
    bc_left_type : str
        boundary condition type at the left edge.  Allowed values
        are: "reflect", "outflow", "periodic"
    bc_left_type : str
        boundary condition type at the right edge.  Allowed values
        are: "reflect", "outflow", "periodic"
    init_cond : function
        the function to call to initialize the conserved state.
        This has the signature `init_cond(euler)`
        where `euler` is an `Euler` object.
    grav_func : function
        the function to call to compute the gravitational acceleration.
        This has the signature: `g = grav_func(grid, rho, params)`, where
        `grid` is a FVGrid`, `rho` is the density (array), and `params`
        is a `dict` of option parameters needed to interpret gravity.
    params : dict, optional
        a dictionary of parameters that is passed to the initial condition
        and gravity functions.  The ratio specific heats can be set
        here as "gamma".
    use_hse_reconstruction : bool, optional
        do we subtract off HSE from pressure before doing the parabolic
        reconstruction?
    hse_as_perturbation : bool, optional
        do we do the characteristic tracing on p' -- the pressure perturbation?
    use_limiting : bool, optional
        do we limit the parabola coefficients?
    use_flattening : bool, optional
        do we apply flattening to the shock to smear them out?
    """

    def __init__(self, nx, C, *,
                 fixed_dt=None,
                 bc_left_type="outflow", bc_right_type="outflow",
                 gamma=1.4,
                 init_cond=None, grav_func=None,
                 params=None,
                 use_hse_reconstruction=False, hse_as_perturbation=False,
                 use_limiting=True, use_flattening=True):

        self.grid = FVGrid(nx, ng=4)
        self.v = FluidVars()

        self.C = C
        self.fixed_dt = fixed_dt

        self.grav_func = grav_func

        # params can be passed into the initial condition and gravity
        # functions to provude any parameters needed to implement the
        # custom behavior
        if params is None:
            self.params = {}
        else:
            self.params = params

        self.gamma = self.params.get("gamma", 1.4)

        if "gamma" not in self.params:
            self.params["gamma"] = gamma

        # setup the BCs -- we need the flexibiility to have different
        # types for each state variable.  In particular, we want
        # odd reflection for velocity

        assert bc_left_type in ["outflow", "reflect", "periodic"]
        assert bc_right_type in ["outflow", "reflect", "periodic"]

        self.bcs_left = self.v.nvar * [bc_left_type]
        if bc_left_type == "reflect":
            self.bcs_left[self.v.umx] = "reflect-odd"

        self.bcs_right = self.v.nvar * [bc_right_type]
        if bc_right_type == "reflect":
            self.bcs_right[self.v.umx] = "reflect-odd"

        # storage for the current solution
        self.U = self.grid.scratch_array(nc=self.v.nvar)

        self.q_parabola = None
        self.g_parabola = None

        # initialize
        init_cond(self)
        for n in range(self.v.nvar):
            self.grid.ghost_fill(self.U[:, n],
                                 bc_left_type=self.bcs_left[n],
                                 bc_right_type=self.bcs_right[n])

        # save ICs for later diagnostics
        self.U_init = self.U.copy()

        self.use_hse_reconstruction = use_hse_reconstruction
        self.hse_as_perturbation = hse_as_perturbation

        self.use_flattening = use_flattening
        self.use_limiting = use_limiting

        self.t = 0
        self.dt = np.nan
        self.nstep = 0

    def estimate_dt(self):
        """compute the Courant-limited timestep from the current
        fluid state"""

        if self.fixed_dt:
            self.dt = self.fixed_dt
            return

        q = self.cons_to_prim()
        cs = np.sqrt(self.gamma * q[:, self.v.qp] / q[:, self.v.qrho])

        self.dt = self.C * self.grid.dx / np.max(np.abs(q[:, self.v.qu]) + cs)

    def cons_to_prim(self):
        """Convert the conserved variable state to primitive variables

        Returns
        -------
        q : ndarray
            the primitive variable array.
        """

        q = self.grid.scratch_array(nc=self.v.nvar)

        q[:, self.v.qrho] = self.U[:, self.v.urho]
        q[:, self.v.qu] = self.U[:, self.v.umx] / self.U[:, self.v.urho]

        rhoe = self.U[:, self.v.uener] - \
            0.5 * q[:, self.v.qrho] * q[:, self.v.qu]**2
        q[:, self.v.qp] = rhoe * (self.gamma - 1.0)

        return q

    def construct_parabola(self):
        """Create the parabola reconstruction of the primitive variables"""

        # convert to primitive variables
        q = self.cons_to_prim()

        # compute flattening
        chi = None
        if self.use_flattening:
            chi = flattening_coefficient(self.grid, q[:, self.v.qp], q[:, self.v.qu])

        g = None
        if self.grav_func is not None:
            g = self.grav_func(self.grid, q[:, self.v.qrho], self.params)

        # construct parabola
        self.q_parabola = []
        for ivar in range(self.v.nvar):
            if ivar == self.v.qp and self.use_hse_reconstruction:
                self.q_parabola.append(HSEPPMInterpolant(self.grid, q[:, ivar], q[:, self.v.qrho], g,
                                                      limit=self.use_limiting, chi_flat=chi, leave_as_perturbation=self.hse_as_perturbation))
            else:
                self.q_parabola.append(PPMInterpolant(self.grid, q[:, ivar],
                                                      limit=self.use_limiting, chi_flat=chi))
            self.q_parabola[-1].construct_parabola()

        # now deal with gravity
        if self.grav_func is not None:
            self.grid.ghost_fill(g,
                                 bc_left_type=self.bcs_left[self.v.umx],
                                 bc_right_type=self.bcs_right[self.v.umx])

            self.g_parabola = PPMInterpolant(self.grid, g,
                                             limit=self.use_limiting, chi_flat=chi)
            self.g_parabola.construct_parabola()

    def interface_states(self):
        """Trace the primitive variables to the interfaces by integrating
        under the parabola and doing a characteristic projection

        Returns
        -------
        q_left : ndarray
            the left primitive variable state on the interface.
        q_right : ndarray
            the right primitive variable state on the interface.
        """

        # convert to primitive variables
        q = self.cons_to_prim()
        cs = np.sqrt(self.gamma * q[:, self.v.qp] / q[:, self.v.qrho])

        # integrate over the 3 waves
        Ip = self.grid.scratch_array(nc=3*self.v.nvar).reshape(self.grid.nq, 3, self.v.nvar)
        Im = self.grid.scratch_array(nc=3*self.v.nvar).reshape(self.grid.nq, 3, self.v.nvar)

        for iwave, sgn in enumerate([-1, 0, 1]):
            sigma = (q[:, self.v.qu] + sgn * cs) * self.dt / self.grid.dx

            for ivar in range(self.v.nvar):
                Im[:, iwave, ivar], Ip[:, iwave, ivar] = self.q_parabola[ivar].integrate(sigma)

        # now deal with gravity
        if self.grav_func is not None:
            Ip_g = self.grid.scratch_array(nc=3)
            Im_g = self.grid.scratch_array(nc=3)

            for iwave, sgn in enumerate([-1, 0, 1]):
                sigma = (q[:, self.v.qu] + sgn * cs) * self.dt / self.grid.dx
                Im_g[:, iwave], Ip_g[:, iwave] = self.g_parabola.integrate(sigma)

        # loop over zones -- we will construct the state on
        # the left and right sides of this zone, these are
        # q_{i,r} and q_{i+1,l}

        # from the perspective of the zone, the subscript
        # p means toward the right edge of the zone and m
        # means toward the left edge of the zone

        # from the perspective of an interface, the subscript
        # left means to the left of the interface and right
        # means to the right of the interface

        q_left = self.grid.scratch_array(nc=self.v.nvar)
        q_right = self.grid.scratch_array(nc=self.v.nvar)

        for i in range(self.grid.lo-1, self.grid.hi+2):

            # right state on interface i -- this uses the "m" reconstructuion

            # reference state -- fastest wave moving to the left
            q_ref_m = Im[i, 0, :]

            # build eigensystem
            if self.use_hse_reconstruction and self.hse_as_perturbation:
                # we need to make the reference state a pressure again
                ev, lvec, rvec = eigen(q_ref_m[self.v.qrho],
                                       q_ref_m[self.v.qu],
                                       self.q_parabola[self.v.qp].p_hse_m[i] + q_ref_m[self.v.qp],
                                       self.gamma)
            else:
                ev, lvec, rvec = eigen(q_ref_m[self.v.qrho],
                                       q_ref_m[self.v.qu],
                                       q_ref_m[self.v.qp],
                                       self.gamma)

            # loop over waves and compute l . (qref - I) for each wave
            beta_xm = np.zeros(3)
            for iwave in range(3):
                dq = q_ref_m - Im[i, iwave, :]
                if self.grav_func is not None and not self.hse_as_perturbation:
                    dq[self.v.qu] -= 0.5 * self.dt * Im_g[i, iwave]
                beta_xm[iwave] = lvec[iwave, :] @ dq

            # finally sum up the waves moving toward the interface,
            # accumulating (l . (q_ref - I)) r
            q_right[i, :] = q_ref_m[:]
            for iwave in range(3):
                if ev[iwave] <= 0:
                    q_right[i, :] -= beta_xm[iwave] * rvec[iwave, :]

            if self.use_hse_reconstruction and self.hse_as_perturbation:
                q_right[i, self.v.qp] += self.q_parabola[self.v.qp].p_hse_m[i]

            # left state on interface i+1 -- this uses the "p" reconstructuion

            # reference state -- fastest wave moving to the right
            q_ref_p = Ip[i, 2, :]

            # build eigensystem
            if self.use_hse_reconstruction and self.hse_as_perturbation:
                ev, lvec, rvec = eigen(q_ref_p[self.v.qrho],
                                       q_ref_p[self.v.qu],
                                       self.q_parabola[self.v.qp].p_hse_p[i] + q_ref_p[self.v.qp],
                                       self.gamma)
            else:
                ev, lvec, rvec = eigen(q_ref_p[self.v.qrho],
                                       q_ref_p[self.v.qu],
                                       q_ref_p[self.v.qp],
                                       self.gamma)

            # loop over waves and compute l . (qref - I) for each wave
            beta_xp = np.zeros(3)
            for iwave in range(3):
                dq = q_ref_p - Ip[i, iwave, :]
                if self.grav_func is not None and not self.hse_as_perturbation:
                    dq[self.v.qu] -= 0.5 * self.dt * Ip_g[i, iwave]
                beta_xp[iwave] = lvec[iwave, :] @ dq

            # finally sum up the waves moving toward the interface,
            # accumulating (l . (q_ref - I)) r
            q_left[i+1, :] = q_ref_p[:]
            for iwave in range(3):
                if ev[iwave] >= 0:
                    q_left[i+1, :] -= beta_xp[iwave] * rvec[iwave, :]

            if self.use_hse_reconstruction and self.hse_as_perturbation:
                q_left[i+1, self.v.qp] += self.q_parabola[self.v.qp].p_hse_p[i]

        # enforce reflection
        for n in range(self.v.nvar):
            if self.bcs_left[n] == "reflect":
                q_left[self.grid.lo, n] = q_right[self.grid.lo, n]
            elif self.bcs_left[n] == "reflect-odd":
                q_left[self.grid.lo, n] = -q_right[self.grid.lo, n]

            if self.bcs_right[n] == "reflect":
                q_right[self.grid.hi+1, n] = q_left[self.grid.hi+1, n]
            elif self.bcs_right[n] == "reflect-odd":
                q_right[self.grid.hi+1, n] = -q_left[self.grid.hi+1, n]

        return q_left, q_right

    def cons_flux(self, state):
        """ given an interface state, return the conservative flux

        Parameters
        ----------
        state : RiemannState
            the interface state from the Riemann solver.

        Returns
        -------
        flux : ndarray
            the conserved flux through the interface for the input state.
        """

        flux = np.zeros((self.v.nvar), dtype=np.float64)

        flux[self.v.urho] = state.rho * state.u
        flux[self.v.umx] = flux[self.v.urho] * state.u + state.p
        flux[self.v.uener] = (0.5 * state.rho * state.u**2 +
                        state.p/(self.gamma - 1.0) + state.p) * state.u
        return flux

    def compute_fluxes(self, q_left, q_right):
        """given the left and right states, solve the Riemann
        problem to get the interface state and return the fluxes

        Parameters
        ----------
        q_left : ndarray
            the left primitive variable state on the interface.
        q_right : ndarray
            the right primitive variable state on the interface.

        Returns
        -------
        flux : ndarray
            the conserved flux through each interface.
        """

        flux = self.grid.scratch_array(nc=self.v.nvar)

        # this is a loop over interfaces
        for i in range(self.grid.lo, self.grid.hi+2):
            sl = State(rho=q_left[i, self.v.qrho],
                       u=q_left[i, self.v.qu],
                       p=q_left[i, self.v.qp])
            sr = State(rho=q_right[i, self.v.qrho],
                       u=q_right[i, self.v.qu],
                       p=q_right[i, self.v.qp])
            rp = RiemannProblem(sl, sr, gamma=self.gamma)
            rp.find_star_state()
            s_int = rp.sample_solution()
            flux[i, :] = self.cons_flux(s_int)

        return flux

    def advance_step(self):
        """Advance the conserved state through a single timestep"""

        # construct interface states
        self.construct_parabola()
        q_left, q_right = self.interface_states()

        # solve Riemann problem and compute fluxes
        flux = self.compute_fluxes(q_left, q_right)

        # conservative update
        # this is a loop over zones

        U_old = self.U.copy()

        for i in range(self.grid.lo, self.grid.hi+1):
            self.U[i, :] += self.dt * (flux[i, :] - flux[i+1, :]) / self.grid.dx

        # time-centered gravitational source terms
        if self.grav_func:
            g_old = self.grav_func(self.grid, U_old[:, self.v.urho], self.params)
            g_new = self.grav_func(self.grid, self.U[:, self.v.urho], self.params)

            self.U[:, self.v.umx] += 0.5 * self.dt * (U_old[:, self.v.urho] * g_old +
                                                      self.U[:, self.v.urho] * g_new)

            self.U[:, self.v.uener] += 0.5 * self.dt * (U_old[:, self.v.umx] * g_old +
                                                        self.U[:, self.v.umx] * g_new)

    def evolve(self, tmax, *, verbose=True):
        """The main evolution driver to advance the state to time tmax

        Parameters
        ----------
        tmax : float
            maximum simulation time to evolve to
        verbose : bool, optional
            enable / disable verbosity
        """

        while self.t < tmax:

            # fill ghost cells
            for n in range(self.v.nvar):
                self.grid.ghost_fill(self.U[:, n],
                                     bc_left_type=self.bcs_left[n],
                                     bc_right_type=self.bcs_right[n])

            # get the timestep
            self.estimate_dt()

            if self.t + self.dt > tmax:
                self.dt = tmax - self.t

            # advance
            self.advance_step()
            self.t += self.dt
            self.nstep += 1

            if verbose:
                print(f"step: {self.nstep:4d}, t = {self.t:#8.4g}, dt = {self.dt:#8.4g}")

        # we're done, but make sure that the ghost cells are
        # consistent with the solution, in case we plot them

        for n in range(self.v.nvar):
            self.grid.ghost_fill(self.U[:, n],
                                 bc_left_type=self.bcs_left[n],
                                 bc_right_type=self.bcs_right[n])

    def draw_prim(self, gp, ivar):
        """Draw the parabola for a primitive variable (ivar) on a GridPlot object

        Parameters
        ----------
        gp : GridPlot
            the grid plot object for the figure
        ivar : int
            the index of the primitive variable to plot
        """

        self.construct_parabola()
        self.q_parabola[ivar].draw_parabola(gp)
        if not isinstance(self.q_parabola[ivar], HSEPPMInterpolant):
            self.q_parabola[ivar].mark_cubic(gp)

    def draw_waves(self, gp):
        """Draw the domains seen by each of the 3 characteristic waves
        and the interface they interact with

        Parameters
        ----------
        gp : GridPlot
            the grid plot object for the figure
        """

        ilo = max(gp.lo_index, self.grid.lo-1)
        ihi = min(gp.hi_index, self.grid.hi+1)

        q = self.cons_to_prim()
        cs = np.sqrt(self.gamma * q[:, self.v.qp] / q[:, self.v.qrho])

        for n in range(ilo, ihi+1):
            u = q[n, self.v.qu]
            evals = np.array([u - cs[n], u, u + cs[n]])
            colors = ["C4", "C9", "C8"]
            for iwave, ev in enumerate(evals):
                if ev > 0:
                    # reaches the right side of the zone
                    gp.ax.fill([self.grid.xr[n] - ev * self.dt,
                                self.grid.xr[n],
                                self.grid.xr[n],
                                self.grid.xr[n] - ev * self.dt],
                                [0.0, 1.0, 0.0, 0.0],
                               color=colors[iwave], alpha=0.33)
                elif ev < 0:
                    # reaches the left side of the zone
                    gp.ax.fill([self.grid.xl[n],
                                self.grid.xl[n] + np.abs(ev) * self.dt,
                                self.grid.xl[n],
                                self.grid.xl[n]],
                                [1.0, 0.0, 0.0, 1.0],
                               color=colors[iwave], alpha=0.25)

    def plot_prim(self, *, ivar=None):
        """Plot the primitive variable(s) for the current solution.

        Parameters
        ----------
        ivar : int
            the index of the variable to plot (None plots rho, u, p)

        Returns
        -------
        matplotlib.pyplot.Figure
        """

        assert ivar is None or (0 <= ivar < self.v.nvar)

        q = self.cons_to_prim()

        if ivar is not None:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(self.grid.x[self.grid.lo:self.grid.hi+1],
                    q[self.grid.lo:self.grid.hi+1, ivar], lw=2)
            ax.grid(color="0.5", linestyle=":")
            ax.set_xlabel("x")
            ax.set_ylabel(self.v.prim_names[ivar])

        else:
            fig, ax = plt.subplots(self.v.nvar, 1, sharex=True)
            for idx in range(self.v.nvar):
                ax[idx].plot(self.grid.x[self.grid.lo:self.grid.hi+1],
                        q[self.grid.lo:self.grid.hi+1, idx], lw=2)
                ax[idx].grid(color="0.5", linestyle=":")
                if idx == self.v.nvar-1:
                    ax[idx].set_xlabel("x")
                ax[idx].set_ylabel(self.v.prim_names[idx])

            size = fig.get_size_inches()
            fig.set_size_inches(size[0], size[0]*self.v.nvar/2.5)

        fig.tight_layout()
        return fig
