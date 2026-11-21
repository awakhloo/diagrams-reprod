"""Continuous-time single-site DMFT (eq:mft-one, eq:dmft-selfcons) for phi = tanh.

    dx/dt = -x + eta + xi,   <eta(t) eta(t')> = sigma^2 <phi(x(t)) phi(x(t'))>,   <xi xi> = G

Solved by self-consistent iteration on a periodic time grid of M points with spacing dt_D
(period T_D = M dt_D). Fourier conventions follow the paper:
    C(w_k) = dt_D * sum_n C(tau_n) e^{-i w_k tau_n} = dt_D * fft(C_tau),   w_k = 2 pi k / T_D
    C(tau_n) = ifft(C_w) / dt_D
The frequency-domain relation is  C0x(w) = |R0x(w)|^2 (sigma^2 C0phi(w) + G(w)),  R0x = 1/(1 + i w).
Only the Gaussian statistics of x enter, so C0phi(tau) = E[phi(x1) phi(x2)] for a bivariate
normal with variance C0x(0) and covariance C0x(tau), evaluated by Gauss-Hermite quadrature.
"""
import numpy as np
from dataclasses import dataclass
from .fourier import forward_ft


def gauss_hermite(n):
    """Nodes/weights for E[f(z)], z ~ N(0,1)."""
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return z, w / w.sum()


def gaussian_pair_expectation(f, v, c, n_gh=64, odd=True, rho_lin=1e-5):
    """E[f(x1) f(x2)] for (x1, x2) jointly Gaussian, Var = v, Cov = c (array over lags).
    Work is done only on the unique |c| values (for odd f the result is odd in c); for
    |rho| = |c/v| < rho_lin the exact odd expansion is linear to O(rho^3) < 1e-15 and
    E = E[f'(x)]^2 c (Stein's identity) is used."""
    z, w = gauss_hermite(n_gh)
    c = np.asarray(c, dtype=float)
    s = np.sqrt(v)
    f1 = f(s * z)
    a1 = np.sum(f1 * z * w) / s  # E[f'(x)] = E[x f(x)] / v  (Stein)
    rho = np.clip(c / v, -1.0, 1.0)
    out = np.empty_like(rho)
    small = np.abs(rho) < rho_lin
    out[small] = a1 ** 2 * c[small]
    key = np.abs(rho[~small]) if odd else rho[~small]
    uniq, inv = np.unique(key, return_inverse=True)
    x2 = s * (uniq[:, None, None] * z[:, None] + np.sqrt(1 - uniq[:, None, None] ** 2) * z[None, :])
    inner = f(x2) @ w  # (n_uniq, n)
    e_uniq = (inner * f1) @ w
    vals = e_uniq[inv]
    if odd:
        vals = vals * np.sign(rho[~small])
    out[~small] = vals
    return out


def gaussian_expectation(f, v, n_gh=64):
    z, w = gauss_hermite(n_gh)
    return np.sum(f(np.sqrt(v) * z) * w)


@dataclass
class DMFTSolution:
    sigma2: float
    dt: float
    tau: np.ndarray      # (M,) periodic grid, tau_n = n dt, n = 0..M-1 (tau > T_D/2 means negative lag)
    omega: np.ndarray    # (M,) fft frequencies
    Cx_tau: np.ndarray
    Cphi_tau: np.ndarray
    gain: float
    G_omega: np.ndarray
    residual: float
    n_iter: int

    @property
    def T_D(self):
        return len(self.tau) * self.dt

    def centered(self, arr):
        """Return (tau_centered, arr_centered) with tau in [-T_D/2, T_D/2)."""
        return np.fft.fftshift(self.tau - np.where(self.tau >= self.T_D / 2, self.T_D, 0.0)), np.fft.fftshift(arr)

    def spectrum(self, which, omegas):
        """C0x or C0phi at arbitrary frequencies by direct trapezoidal transform (real, even)."""
        arr = self.Cx_tau if which == "x" else self.Cphi_tau
        tau_c, a_c = self.centered(arr)
        # extend by one point to close the period so the trapezoid covers [-T_D/2, T_D/2]
        tau_c = np.append(tau_c, tau_c[0] + self.T_D)
        a_c = np.append(a_c, a_c[0])
        return forward_ft(tau_c, a_c, omegas).real

    def R0x(self, omegas):
        return 1.0 / (1.0 + 1j * np.asarray(omegas))

    def R0phi(self, omegas):
        return self.gain * self.R0x(omegas)

    @property
    def Delta0(self):
        return float(self.Cx_tau[0])

    def tau_c(self):
        """Integral timescale of C0phi: int_0^inf C0phi(tau) dtau / C0phi(0)."""
        tau_c, a_c = self.centered(self.Cphi_tau)
        pos = tau_c >= 0
        return float(np.trapezoid(a_c[pos], tau_c[pos]) / a_c[pos][0])


def solve(sigma2, dt=0.05, T_D=200.0, G=0.0, n_gh=64, tol=1e-12, max_iter=5000, damping=1.0,
          phi=np.tanh, Cx0_init=None, verbose=False):
    """Iterate the DMFT to a fixed point. G is a constant (white noise, G(w) = G) or a callable G(w)."""
    M = int(round(T_D / dt))
    tau = np.arange(M) * dt
    omega = 2 * np.pi * np.fft.fftfreq(M, d=dt)
    G_w = G(omega) if callable(G) else np.full(M, float(G))
    R2 = 1.0 / (1.0 + omega ** 2)
    if Cx0_init is None:
        Cx = 0.5 * np.exp(-np.minimum(tau, T_D - tau))  # any decaying, even initial guess
    else:
        Cx = np.array(Cx0_init, dtype=float)
    residual = np.inf
    for it in range(1, max_iter + 1):
        v = Cx[0]
        Cphi = gaussian_pair_expectation(phi, v, Cx, n_gh)
        Cphi_w = (dt * np.fft.fft(Cphi)).real
        Cx_w_new = R2 * (sigma2 * Cphi_w + G_w)
        Cx_new = (np.fft.ifft(Cx_w_new) / dt).real
        residual = np.max(np.abs(Cx_new - Cx))
        Cx = (1 - damping) * Cx + damping * Cx_new
        if not np.isfinite(residual) or Cx[0] <= 0:
            raise RuntimeError(f"DMFT iteration diverged or collapsed at iter {it} (Cx[0] = {Cx[0]})")
        if verbose and it % 50 == 0:
            print(f"iter {it}: residual {residual:.3e}, Delta0 {Cx[0]:.8f}")
        if residual < tol:
            break
    if residual >= tol:
        raise RuntimeError(f"DMFT did not converge: residual {residual:.2e} after {max_iter} iterations")
    v = Cx[0]
    Cphi = gaussian_pair_expectation(phi, v, Cx, n_gh)
    # The map also has a frozen fixed point (Cx constant in tau, q = sigma^2 <phi^2>_q) and the trivial one.
    # The chaotic solution decays over the period; flag anything else.
    if Cx[M // 2] > 1e-6 * v:
        import warnings
        warnings.warn(f"DMFT fixed point does not decay over the period (Cx(T_D/2)/Cx(0) = {Cx[M // 2] / v:.2e}); "
                      "frozen branch or T_D too short")
    z, w = gauss_hermite(n_gh)
    gain = float(np.sum(phi(np.sqrt(v) * z) * z * w) / np.sqrt(v))  # E[phi'(x)] by Stein's identity
    return DMFTSolution(sigma2, dt, tau, omega, Cx, Cphi, gain, G_w, residual, it)


# ---- independent check for the deterministic continuous case (Sompolinsky-Crisanti-Sommers) ----
def energy_condition(Delta0, sigma2, n_gh=96, phi=np.tanh, n_int=400):
    """V(Delta0) - V(0) with V(D) = -D^2/2 + sigma^2 int_0^D f_phi(c; Delta0) dc, f_phi = E[phi phi]; V(0) = 0.
    The chaotic DMFT solution has Delta(tau -> inf) = 0 with zero velocity, so this must vanish."""
    c = np.linspace(0.0, Delta0, n_int)
    f = gaussian_pair_expectation(phi, Delta0, c, n_gh)
    return -0.5 * Delta0 ** 2 + sigma2 * np.trapezoid(f, c)


def energy_condition_closed_form(Delta0, sigma2, n_gh=96):
    """Same quantity using int_0^D E[phi phi] dc = Var[Phi(x)], Phi = log cosh (SCS 1988)."""
    Phi = lambda x: np.logaddexp(x, -x) - np.log(2.0)
    var_Phi = gaussian_expectation(lambda x: Phi(x) ** 2, Delta0, n_gh) - gaussian_expectation(Phi, Delta0, n_gh) ** 2
    return -0.5 * Delta0 ** 2 + sigma2 * var_Phi


def Delta0_from_energy(sigma2, n_gh=96):
    from scipy.optimize import brentq
    return brentq(lambda D: energy_condition_closed_form(D, sigma2, n_gh), 1e-6, 50.0, xtol=1e-13)
