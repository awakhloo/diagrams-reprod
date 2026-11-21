"""Exact check of theory.py on the continuous linear network dx/dt = -x + J x + xi, <xi xi> = G delta.
Stationary covariance solves (J - I) S + S (J - I)^T = -G I and C(tau) = e^{(J - I) tau} S for tau >= 0,
C(-tau) = C(tau)^T. The theory with gain = 1, C0phi = C0x (so h_x = 0) is exact for every N.
The J is rescaled so that max Re(lambda) = 0.97: a 0.03-wide analyticity strip, like the tanh network."""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm
from scs.theory import theory_lags, strip_width
from scs.backend import xp


def make_case(N=60, seed=3, rmax=0.97):
    rng = np.random.default_rng(seed)
    J = rng.normal(0, 1 / np.sqrt(N), (N, N))
    lam = np.linalg.eigvals(J)
    J *= rmax / lam.real.max()
    return J


def exact_lags(J, G, taus):
    N = J.shape[0]
    S = solve_continuous_lyapunov(J - np.eye(N), -G * np.eye(N))
    out = []
    for tau in taus:
        if tau >= 0:
            out.append(expm((J - np.eye(N)) * tau) @ S)
        else:
            out.append((expm((J - np.eye(N)) * (-tau)) @ S).T)
    return np.array(out)


def test_linear_theory_matches_lyapunov_and_converges_in_dw():
    J = make_case()
    N = J.shape[0]
    G = 2.0
    taus = np.array([-2.0, -0.5, 0.0, 1.0])
    exact = exact_lags(J, G, taus)
    width, _ = strip_width(xp.asarray(J), 1.0)
    assert abs(width - 0.03) < 1e-9
    one = lambda w: np.ones_like(w)
    errs = {}
    for dw in (0.04, 0.02, 0.01, 0.005):
        th = theory_lags(xp.asarray(J), 1.0, one, one, sigma2=0.0, rows=np.arange(N), taus=taus,
                         dw=dw, w_max=400.0, G=G, fields=("x",))
        errs[dw] = np.max(np.abs(th["x"] - exact))
    # exponential convergence in dw: each halving of dw shrinks the error by a large factor until the
    # w_max truncation (~G/(pi w_max) ~ 1.6e-3 relative at |tau|~0... handled by large w_max) floor
    assert errs[0.005] < 1e-6, errs
    assert errs[0.02] / errs[0.01] > 20, errs
    # equal-time covariance must be symmetric and the lags related by transposition
    th = theory_lags(xp.asarray(J), 1.0, one, one, 0.0, np.arange(N), np.array([0.0, 1.0, -1.0]), 0.005, 400.0, G=G, fields=("x",))
    assert np.max(np.abs(th["x"][0] - th["x"][0].T)) < 1e-6
    assert np.max(np.abs(th["x"][1] - th["x"][2].T)) < 1e-6


def test_row_subset_matches_full():
    J = make_case(N=40, seed=5)
    one = lambda w: np.ones_like(w)
    rows = np.array([3, 17, 22])
    full = theory_lags(xp.asarray(J), 1.0, one, one, 0.0, np.arange(40), [0.0, -1.0], 0.02, 100.0, G=1.0, fields=("x",))["x"]
    sub = theory_lags(xp.asarray(J), 1.0, one, one, 0.0, rows, [0.0, -1.0], 0.02, 100.0, G=1.0, fields=("x",))["x"]
    assert np.max(np.abs(sub - full[:, rows, :])) < 1e-12
