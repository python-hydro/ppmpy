"""An exact Riemann solver for the Euler equations with a gamma-law
gas.  The left and right states are stored as State objects.  We then
create a RiemannProblem object with the left and right state:

`rp = RiemannProblem(left_state, right_state)`

Next we solve for the star state:

`rp.find_star_state()`

Finally, we sample the solution to find the interface state, which
is returned as a State object:

`q_int = rp.sample_solution()`
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


class State:
    """ a simple object to hold a primitive variable state

    Parameters
    ----------
    p : float
        pressure
    u : float
        velocity
    rho : float
        density
    """

    def __init__(self, *, p=1.0, u=0.0, rho=1.0):
        self.p = p
        self.u = u
        self.rho = rho

    def __str__(self):
        return f"rho: {self.rho}; u: {self.u}; p: {self.p}"


class RiemannProblem:
    """ a class to define a Riemann problem.  It takes a left
        and right state.  Note: we assume a constant gamma.

    Parameters
    ----------
    left_state : State
        primitive variable state to the left of the interface.
    right_state : State
        primitive variable state to the right of the interface.
    gamma : float
        ratio of specific heats.
    """

    def __init__(self, left_state, right_state, *, gamma=1.4):
        self.left = left_state
        self.right = right_state
        self.gamma = gamma

        self.ustar = None
        self.pstar = None

    def __str__(self):
        return f"pstar = {self.pstar}, ustar = {self.ustar}"

    def u_hugoniot(self, p, side):
        """define the Hugoniot curve, u(p).

        Parameters
        ----------
        p : float
            pressure
        side : str
            "left" or "right" to indicate which state to use.

        Returns
        -------
        float
            the velocity on the Hugoniot curve for the input pressure
        """

        if side == "left":
            state = self.left
            s = 1.0
        elif side == "right":
            state = self.right
            s = -1.0
        else:
            raise ValueError("invalid side")

        c = np.sqrt(self.gamma * state.p / state.rho)

        if p < state.p:
            # rarefaction
            u = state.u + s * (2.0 * c / (self.gamma - 1.0)) * \
                (1.0 - (p/state.p)**((self.gamma - 1.0) / (2.0 * self.gamma)))
        else:
            # shock
            beta = (self.gamma + 1.0) / (self.gamma - 1.0)
            u = state.u + s * (2.0 * c / np.sqrt(2.0 * self.gamma * (self.gamma-1.0))) * \
                (1.0 - p/state.p) / np.sqrt(1.0 + beta * p / state.p)

        return u

    def find_star_state(self, p_min=0.001, p_max=1000.0):
        """ root find the Hugoniot curve to find ustar, pstar.

        Parameters
        ----------
        p_min : float, optional
            minimum possible pressure.
        p_max : float, optional
            maximum possible pressure.
        """

        # we need to root-find on
        try:
            self.pstar = optimize.brentq(
                lambda p: self.u_hugoniot(p, "left") - self.u_hugoniot(p, "right"),
                p_min, p_max)
        except ValueError:
            print("unable to solve for the star region")
            print(f"left state = {self.left}")
            print(f"right state = {self.right}")
            raise

        self.ustar = self.u_hugoniot(self.pstar, "left")

    def shock_solution(self, sgn, state):
        """return the interface solution considering a shock.

        Parameters
        ----------
        sgn : float
            a sign, -1 or +1, indicating whether it is "+" or "-" in the
            shock expression (this depends on left or right jump).
        state : State
            the Riemann state on the non-star side of the shock.

        Returns
        -------
        State
            the star state across the shock.
        """

        p_ratio = self.pstar/state.p
        c = np.sqrt(self.gamma*state.p/state.rho)

        # Toro, eq. 4.52 / 4.59
        S = state.u + sgn*c*np.sqrt(0.5*(self.gamma + 1.0)/self.gamma*p_ratio +
                                    0.5*(self.gamma - 1.0)/self.gamma)

        # are we to the left or right of the shock?
        if (self.ustar < 0 and S < 0) or (self.ustar > 0 and S > 0):
            # R/L region
            solution = state
        else:
            # * region -- get rhostar from Toro, eq. 4.50 / 4.57
            gam_fac = (self.gamma - 1.0)/(self.gamma + 1.0)
            rhostar = state.rho * (p_ratio + gam_fac)/(gam_fac * p_ratio + 1.0)
            solution = State(rho=rhostar, u=self.ustar, p=self.pstar)

        return solution

    def rarefaction_solution(self, sgn, state):
        """return the interface solution considering a rarefaction wave.

        Parameters
        ----------
        sgn : float
            a sign, -1 or +1, indicating whether it is "+" or "-" in the
            rarefaction expression (this depends on left or right jump).
        state : State
            the Riemann state on the non-star side of the rarefaction.

        Returns
        -------
        State
            the star state across the rarefaction.
        """

        # find the speed of the head and tail of the rarefaction fan

        # isentropic (Toro eq. 4.54 / 4.61)
        p_ratio = self.pstar / state.p
        c = np.sqrt(self.gamma * state.p / state.rho)
        cstar = c*p_ratio**((self.gamma - 1.0) / (2 * self.gamma))

        lambda_head = state.u + sgn*c
        lambda_tail = self.ustar + sgn*cstar

        gam_fac = (self.gamma - 1.0) / (self.gamma + 1.0)

        if sgn * lambda_head < 0:
            # R/L region
            solution = state

        elif sgn * lambda_tail > 0:
            # * region, we use the isentropic density (Toro 4.53 / 4.60)
            solution = State(rho=state.rho*p_ratio**(1.0/self.gamma),
                             u=self.ustar,
                             p=self.pstar)

        else:
            # we are in the fan -- Toro 4.56 / 4.63
            rho = state.rho * (2 / (self.gamma + 1.0) -
                               sgn * gam_fac * state.u / c)**(2.0 / (self.gamma-1.0))
            u = 2.0 / (self.gamma + 1.0) * (-sgn * c + 0.5 * (self.gamma - 1.0) * state.u)
            p = state.p * (2 / (self.gamma + 1.0) -
                           sgn * gam_fac * state.u / c)**(2.0 * self.gamma / (self.gamma - 1.0))
            solution = State(rho=rho, u=u, p=p)

        return solution

    def sample_solution(self):
        """given the star state (ustar, pstar), find the state on the interface"""

        if self.ustar < 0:
            # we are in the R* or R region
            state = self.right
            sgn = 1.0
        else:
            # we are in the L* or L region
            state = self.left
            sgn = -1.0

        # is the non-contact wave a shock or rarefaction?
        if self.pstar > state.p:
            # compression! we are a shock
            solution = self.shock_solution(sgn, state)

        else:
            # rarefaction
            solution = self.rarefaction_solution(sgn, state)

        return solution


def plot_hugoniot(riemann_problem, p_min=0.0, p_max=1.5, N=500):
    """ plot the Hugoniot curves.

    Parameters
    ----------
    riemann_problem : RiemannProblem
        the Riemann problem object.
    p_min : float
        the minimum pressure to plot.
    p_max : float
        the maximum pressure to plot.
    N : int
        number of points to use in the plot.

    Returns
    -------
    matplotlib.pyplot.Figure
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    p = np.linspace(p_min, p_max, num=N)
    u_left = np.zeros_like(p)
    u_right = np.zeros_like(p)

    for n in range(N):
        u_left[n] = riemann_problem.u_hugoniot(p[n], "left")

    # shock for p > p_s; rarefaction otherwise
    ish = np.where(p > riemann_problem.left.p)
    ir = np.where(p < riemann_problem.left.p)

    ax.plot(p[ish], u_left[ish], c="C0", ls="-", lw=2)
    ax.plot(p[ir], u_left[ir], c="C0", ls=":", lw=2)
    ax.scatter([riemann_problem.left.p], [riemann_problem.left.u],
               marker="x", c="C0", s=40)

    du = 0.025*(max(np.max(u_left), np.max(u_right)) -
                min(np.min(u_left), np.min(u_right)))

    ax.text(riemann_problem.left.p, riemann_problem.left.u+du, "left",
            horizontalalignment="center", color="C0")

    for n in range(N):
        u_right[n] = riemann_problem.u_hugoniot(p[n], "right")

    ish = np.where(p > riemann_problem.right.p)
    ir = np.where(p < riemann_problem.right.p)

    ax.plot(p[ish], u_right[ish], c="C1", ls="-", lw=2)
    ax.plot(p[ir], u_right[ir], c="C1", ls=":", lw=2)
    ax.scatter([riemann_problem.right.p], [riemann_problem.right.u],
               marker="x", c="C1", s=40)

    ax.text(riemann_problem.right.p, riemann_problem.right.u+du, "right",
            horizontalalignment="center", color="C1")

    ax.set_xlim(p_min, p_max)

    ax.set_xlabel(r"$p$", fontsize="large")
    ax.set_ylabel(r"$u$", fontsize="large")

    return fig


if __name__ == "__main__":

    q_l = State(rho=1.0, u=0.0, p=1.0)
    q_r = State(rho=0.125, u=0.0, p=0.1)

    rp = RiemannProblem(q_l, q_r, gamma=1.4)

    rp.find_star_state()
    q_int = rp.sample_solution()
    print(q_int)
