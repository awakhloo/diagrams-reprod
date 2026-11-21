"""two_sided: the C_ij(tau) = C_ji(-tau) assembly, on an exact function and on the AR(1) delayed-copy process."""
import numpy as np
from scs.twopoint import two_sided
from tests.test_estimators import run_ar1


def test_exact_function():
    # f_ij(tau) = exp(-|tau - d_ij|) with d antisymmetric satisfies f_ij(-tau) = f_ji(tau)
    N, rows = 6, np.array([1, 3, 4])
    rng = np.random.default_rng(0)
    d = rng.normal(size=(N, N)); d = d - d.T
    taus_neg = -np.arange(0.0, 5.0 + 1e-9, 0.5)          # 0, -0.5, ..., -5
    f = lambda i, j, tau: np.exp(-np.abs(tau - d[i, j]))
    dense = np.array([[[f(i, j, t) for j in range(N)] for i in rows] for t in taus_neg])
    out, asym = two_sided(dense, rows)
    taus = np.concatenate([taus_neg[::-1], -taus_neg[1:]])   # -5 .. 5
    assert out.shape == (3, 3, 21) and np.allclose(taus, np.arange(-5, 5.01, 0.5))
    for a, i in enumerate(rows):
        for b, j in enumerate(rows):
            assert np.allclose(out[a, b], f(i, j, taus))
    assert asym < 1e-15


def test_ar1_delayed_copy():
    # x_1(t) = x_0(t - d): C_01(tau) = <x_0(t + tau) x_0(t - d)> = a^{|tau + d|}, peaked at tau = -d
    N, B, dt, d = 4, 8, 0.05, 10
    lags = list(range(0, 31))
    acc, a = run_ar1(N, B, 60000, dt, seed=3, lag_steps=lags, delay_pair=(0, 1, d))
    dense = np.stack([acc.estimates("x")[l][2] for l in lags])          # (31, N, N), rows = all
    out, asym = two_sided(dense, np.arange(N))
    l = np.arange(-30, 31)
    assert np.max(np.abs(out[0, 1] - a ** np.abs(l + d))) < 0.06
    assert np.max(np.abs(out[1, 0] - a ** np.abs(l - d))) < 0.06
    assert np.max(np.abs(out[2, 2] - a ** np.abs(l))) < 0.06
    assert np.max(np.abs(out[0, 1] - out[1, 0][::-1])) < 1e-12          # mirror identity is exact by construction
    assert asym < 1e-12
