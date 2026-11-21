"""Fourier helpers with the paper's convention:
    C(omega) = int dtau e^{-i omega tau} C(tau),   C(tau) = (1/2pi) int domega e^{i omega tau} C(omega).
"""
import numpy as np


def inverse_ft(omega, C_omega, taus):
    """Trapezoidal inverse transform on a uniform omega grid (any length, any sign range).
    C_omega has the frequency axis first; returns array with shape (len(taus),) + C_omega.shape[1:]."""
    omega = np.asarray(omega)
    dw = omega[1] - omega[0]
    w = np.ones(len(omega), dtype=float)
    w[0] = w[-1] = 0.5
    taus = np.atleast_1d(taus)
    ph = np.exp(1j * np.outer(taus, omega)) * w  # (n_tau, n_w)
    out = np.tensordot(ph, C_omega, axes=(1, 0)) * dw / (2 * np.pi)
    return out


def forward_ft(tau, C_tau, omegas):
    """Trapezoidal forward transform on a uniform tau grid."""
    tau = np.asarray(tau)
    dt = tau[1] - tau[0]
    w = np.ones(len(tau), dtype=float)
    w[0] = w[-1] = 0.5
    omegas = np.atleast_1d(omegas)
    ph = np.exp(-1j * np.outer(omegas, tau)) * w
    return np.tensordot(ph, C_tau, axes=(1, 0)) * dt
