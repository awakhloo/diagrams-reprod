"""Fourier convention checks: C(omega) = int dtau e^{-i omega tau} C(tau)."""
import numpy as np
from scs.fourier import inverse_ft, forward_ft


def test_gaussian_pair_high_precision():
    # C(tau) = exp(-tau^2/2)  <->  C(omega) = sqrt(2 pi) exp(-omega^2/2)
    w = np.arange(-40, 40 + 1e-9, 0.05)
    Cw = np.sqrt(2 * np.pi) * np.exp(-w**2 / 2)
    taus = np.array([-2.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    Ct = inverse_ft(w, Cw, taus)
    assert np.max(np.abs(Ct - np.exp(-taus**2 / 2))) < 1e-10
    # forward direction on a tau grid
    tau = np.arange(-40, 40 + 1e-9, 0.05)
    Cw2 = forward_ft(tau, np.exp(-tau**2 / 2), np.array([0.0, 1.0, 2.5]))
    assert np.max(np.abs(Cw2 - np.sqrt(2 * np.pi) * np.exp(-np.array([0.0, 1.0, 2.5])**2 / 2))) < 1e-10


def test_causal_response_sign_convention():
    # R(tau) = e^{-tau} theta(tau)  <->  R(omega) = 1/(1 + i omega)   (paper: R0x)
    w = np.arange(-4000, 4000 + 1e-9, 0.02)
    Rw = 1.0 / (1.0 + 1j * w)
    taus = np.array([-1.0, 0.5, 1.0, 2.0])
    Rt = inverse_ft(w, Rw, taus)
    expected = np.where(taus > 0, np.exp(-taus), 0.0)
    assert np.max(np.abs(Rt - expected)) < 2e-3


def test_hermitian_matrix_spectrum_gives_real_lags():
    # C(omega) = A(omega) A(omega)^dagger with A(-omega) = conj(A(omega)) -> C(tau) real, C(-tau) = C(tau)^T
    rng = np.random.default_rng(0)
    n = 4
    B = rng.normal(size=(n, n))
    w = np.arange(-30, 30 + 1e-9, 0.02)
    A = np.stack([np.linalg.inv(np.eye(n) * (1 + 1j * wi) - 0.3 * B) for wi in w])  # (n_w, n, n)
    Cw = np.einsum("wik,wjk->wij", A, A.conj())
    taus = np.array([-1.0, 0.0, 1.0])
    Ct = inverse_ft(w, Cw, taus)
    assert np.max(np.abs(Ct.imag)) < 1e-8
    assert np.max(np.abs(Ct[0].real - Ct[2].real.T)) < 1e-8
