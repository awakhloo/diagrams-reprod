import numpy as np
from scs import dmft


def test_gaussian_pair_expectation_odd_polynomials():
    # The routine assumes odd f (as phi is). f(x) = x: E[x1 x2] = c ; f(x) = x^3: E[x1^3 x2^3] = 9 v^2 c + 6 c^3
    v = 1.7
    c = np.array([-1.2, -1e-6, 0.0, 1e-6, 0.5, 1.7])
    assert np.allclose(dmft.gaussian_pair_expectation(lambda x: x, v, c), c, atol=1e-12)
    assert np.allclose(dmft.gaussian_pair_expectation(lambda x: x ** 3, v, c), 9 * v ** 2 * c + 6 * c ** 3, rtol=1e-12, atol=1e-13)
    # tanh against a fine 2D grid at rho = -0.7 (odd branch, negative) and rho = 0.999 (near 1)
    z = np.linspace(-9, 9, 6001); dz = z[1] - z[0]
    g = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    for rho in (-0.7, 0.999):
        x1 = np.sqrt(v) * z[:, None]; x2 = np.sqrt(v) * (rho * z[:, None] + np.sqrt(1 - rho ** 2) * z[None, :])
        brute = np.sum(np.tanh(x1) * np.tanh(x2) * g[:, None] * g[None, :]) * dz * dz
        val64 = dmft.gaussian_pair_expectation(np.tanh, v, np.array([rho * v]), n_gh=64)[0]
        val128 = dmft.gaussian_pair_expectation(np.tanh, v, np.array([rho * v]), n_gh=128)[0]
        assert abs(val64 - brute) < 1e-6, (rho, val64, brute)   # 64 nodes: up to 3.4e-7 near rho -> 1
        assert abs(val128 - brute) < 1e-9, (rho, val128, brute)  # 128 nodes: <= 1.7e-10; 96 nodes (cached DMFT): <= 5.6e-9


def test_linear_dmft_matches_closed_form():
    # phi = x, sigma^2 = 0.25, white noise G = 1: C0x(w) = G / (1 + w^2 - sigma^2) -> C(tau) = G e^{-k|tau|}/(2k), k = sqrt(1 - sigma^2)
    sol = dmft.solve(0.25, dt=0.05, T_D=200.0, G=1.0, phi=lambda x: x, tol=1e-13)
    k = np.sqrt(0.75)
    tau_c, C = sol.centered(sol.Cx_tau)
    # With white noise C(tau) has a kink at 0, so the periodic-grid solution carries an O(dt)
    # Nyquist-truncation error at tau = 0: (1/pi) [pi/2 - arctan(w_N/k)] / k with w_N = pi/dt.
    trunc = (np.pi / 2 - np.arctan(np.pi / sol.dt / k)) / (np.pi * k)
    err0 = abs(C[np.argmin(np.abs(tau_c))] - 1 / (2 * k))
    assert abs(err0 - trunc) < 0.1 * trunc, (err0, trunc)
    far = np.abs(tau_c) > 1.0
    assert np.max(np.abs(C[far] - np.exp(-k * np.abs(tau_c[far])) / (2 * k))) < 1e-4
    assert abs(sol.spectrum("x", np.array([0.0]))[0] - 1.0 / 0.75) < 1e-4  # spectrum itself has the same O(dt) floor


def test_chaotic_solution_energy_condition():
    sol = dmft.solve(3.0, dt=0.05, T_D=200.0, tol=1e-10, damping=1.0)
    assert sol.residual < 1e-10
    D0_energy = dmft.Delta0_from_energy(3.0)
    assert abs(sol.Delta0 - D0_energy) < 2e-4, (sol.Delta0, D0_energy)
    assert abs(dmft.energy_condition(D0_energy, 3.0)) < 1e-6  # numeric-integral and closed-form versions agree
    assert 0 < sol.gain * np.sqrt(3.0) < 1


def test_gain_stein_equals_direct():
    v = 1.3
    z, w = dmft.gauss_hermite(64)
    direct = np.sum((1 - np.tanh(np.sqrt(v) * z) ** 2) * w)
    stein = np.sum(np.tanh(np.sqrt(v) * z) * z * w) / np.sqrt(v)
    assert abs(direct - stein) < 1e-7  # Gauss-Hermite error at 64 nodes is ~4e-8; 1e-12 needs ~140 nodes
    z, w = dmft.gauss_hermite(160)
    assert abs(np.sum((1 - np.tanh(np.sqrt(v) * z) ** 2) * w) - np.sum(np.tanh(np.sqrt(v) * z) * z * w) / np.sqrt(v)) < 1e-12
