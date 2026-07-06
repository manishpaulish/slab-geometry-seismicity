"""
rate_state_slider.py
--------------------
Numerically stable single-block rate-and-state friction slider, the
building block of the Burridge-Knopoff earthquake model.

This is a computational-methods reference implementation: it demonstrates
how to integrate the stiff, quasi-dynamic rate-and-state system without the
velocity blow-up that naive formulations suffer during coseismic slip.

Key numerical choices (this is where the difficulty lives):
  * Regularized friction  f = a*asinh( V/(2 V0) * exp(psi/a) ), valid at V=0,
    removing the ln(0) singularity of the classical form.
  * Quasi-dynamic force balance with radiation damping (eta*V) replacing raw
    inertia, which removes the microsecond-scale stiffness of full dynamics.
  * Velocity recovered each step by a monotone bracketed root find (Brent),
    not Newton, guaranteeing convergence through the stick->slip transition.
  * State integrated in the psi form of the aging law.

Reference physics:
  Erickson, Birnir & Lavallee (2011), Geophys. J. Int. 187, 178-198.
  Regularized RSF form: Rice, Lapusta & Ranjith (2001); Lapusta et al. (2000).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class RateStateSlider:
    """
    Single-degree-of-freedom spring-block slider with regularized
    rate-and-state friction under the quasi-dynamic approximation.
    """

    def __init__(self, a=0.015, b=0.020, Dc=1.0, sigma=1.0, V0=1.0,
                 f0=0.6, v_p=1e-3, eta=0.1, k_ratio=0.5):
        self.a, self.b, self.Dc = a, b, Dc
        self.sigma, self.V0, self.f0 = sigma, V0, f0
        self.v_p, self.eta = v_p, eta
        self.k_crit = (b - a) * sigma / Dc
        self.k = k_ratio * self.k_crit          # < k_crit => unstable => stick-slip

    def friction(self, V, psi):
        return self.a * np.arcsinh(V / (2 * self.V0) * np.exp(psi / self.a))

    def solve_velocity(self, u, psi, t):
        """Recover slip velocity from the quasi-dynamic force balance."""
        load = self.k * (self.v_p * t - u)
        g = lambda V: self.sigma * self.friction(V, psi) + self.eta * V - load
        if g(1e-20) > 0:          # load below static friction -> locked
            return 0.0
        return brentq(g, 1e-20, 1e8, xtol=1e-20, rtol=1e-12, maxiter=300)

    def _rhs(self, t, y):
        u, psi = y
        V = self.solve_velocity(u, psi, t)
        dpsi = (self.b * self.V0 / self.Dc) * (
            np.exp((self.f0 - psi) / self.b) - V / self.V0)
        return [V, dpsi]

    def simulate(self, n_cycles=1500, n_out=200000):
        psi_ss = self.f0 - self.b * np.log(self.v_p / self.V0)
        y0 = [-(self.sigma * self.f0) / self.k * 0.98, psi_ss * 0.9]
        t_max = n_cycles * self.Dc / self.v_p
        sol = solve_ivp(self._rhs, (0, t_max), y0, method="LSODA",
                        rtol=1e-7, atol=1e-10, dense_output=True,
                        max_step=t_max / 1.5e5)
        t = np.linspace(0, t_max, n_out)
        Y = sol.sol(t)
        V = np.array([self.solve_velocity(Y[0, i], Y[1, i], t[i])
                      for i in range(len(t))])
        return dict(t=t, u=Y[0], psi=Y[1], V=V, success=sol.success)


def classify_events(t, V, v_p, threshold=5.0):
    """
    Detect slip events and classify the sequence as periodic or chaotic
    by the coefficient of variation (CV) of recurrence intervals.
    """
    pk = (V[1:-1] > V[:-2]) & (V[1:-1] > V[2:]) & (V[1:-1] > threshold * v_p)
    ev = t[1:-1][pk]
    iv = np.diff(ev)
    if len(iv) < 3:
        return dict(n_events=len(ev), cv=np.nan, classification="insufficient")
    cv = float(iv.std() / iv.mean())
    return dict(n_events=len(ev), cv=cv,
                classification="periodic" if cv < 0.15 else "chaotic",
                mean_recurrence=float(iv.mean()))


if __name__ == "__main__":
    slider = RateStateSlider(k_ratio=0.5)
    print(f"k_crit = {slider.k_crit:.5f},  k = {slider.k:.5f}")
    res = slider.simulate(n_cycles=1500)
    cls = classify_events(res["t"], res["V"], slider.v_p)
    print(f"integration success: {res['success']}")
    print(f"peak V / loading rate: {res['V'].max() / slider.v_p:.1f}")
    print(f"events: {cls['n_events']},  CV: {cls['cv']:.4f},  "
          f"class: {cls['classification']}")
