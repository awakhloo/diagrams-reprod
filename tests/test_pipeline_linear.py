"""End-to-end check of simulate-loop + LagAccumulator on a linear noisy map with an exactly known
lagged covariance. Map: x_{t+1} = A x_t + s z_t, A = (1 - dt) I + dt J, z ~ N(0, I).
Stationary covariance S solves S = A S A^T + s^2 I (discrete Lyapunov); <x(t + l) x(t)^T> = A^l S.
The accumulator returns <x_i(t - l) x_j(t)> = [A^l S]^T_ij = [S (A^T)^l]_ij for i in rows."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scs.twopoint import LagAccumulator, metrics
from scs.backend import xp


def test_linear_map_lagged_covariance():
    N, B, dt, s = 40, 16, 0.05, 0.3
    rng = np.random.default_rng(7)
    J = rng.normal(0, 0.6 / np.sqrt(N), (N, N))
    A = (1 - dt) * np.eye(N) + dt * J
    S = solve_discrete_lyapunov(A, s**2 * np.eye(N))
    lags = [0, 10, 40]
    exact = {l: (np.linalg.matrix_power(A, l) @ S).T for l in lags}
    x = rng.normal(size=(N, B)) @ np.linalg.cholesky(S).T if False else rng.normal(0, 1, (N, B))
    for _ in range(4000):  # transient
        x = A @ x + s * rng.normal(size=(N, B))
    rows = np.array([0, 5, 11, 30])
    acc = LagAccumulator(N, B, rows, lags, every=3, fields=("x",))
    n_steps = 150000
    for t in range(n_steps):
        x = A @ x + s * rng.normal(size=(N, B))
        acc.update(t, xp.asarray(x), None)
    est = acc.estimates("x")
    for l in lags:
        c1, c2, c = est[l]
        m = metrics(c, exact[l][rows], c1, c2, rows, N)
        # corrected MSE must be consistent with zero: |corrected| < 3 * (floor uncertainty ~ floor * sqrt(2/n))
        n_off = m["n_off"]
        assert abs(m["mse_off_corr"]) < 4 * m["floor_off"] * np.sqrt(2.0 / n_off) + 1e-12, (l, m)
        # a transposition error would show up as a bias of order the asymmetry of A^l S for l > 0
        if l > 0:
            asym = np.mean((exact[l][rows] - exact[l].T[rows]) ** 2)
            assert asym > 20 * m["floor_off"], "test not discriminating: asymmetry below noise"
            wrong = metrics(c, exact[l].T[rows], c1, c2, rows, N)
            assert wrong["mse_off_corr"] > 10 * m["floor_off"]
        assert abs(m["slope_off"] - 1) < 0.05, (l, m["slope_off"])
