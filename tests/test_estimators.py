"""LagAccumulator on exact AR(1) processes: lag convention, unbiasedness, noise floor."""
import numpy as np
from scs.twopoint import LagAccumulator, noise_floor, offdiag_mask, metrics
from scs.network import STEPPERS
from scs.backend import xp


def run_ar1(N, B, n_steps, dt, seed, lag_steps, every=1, delay_pair=None):
    """Independent AR(1) processes x_{t+1} = a x_t + sqrt(1-a^2) z, a = e^{-dt}: C_ij(tau) = delta_ij e^{-|tau|}.
    If delay_pair = (i, j, d), process j is replaced by a copy of process i delayed by d steps."""
    rng = np.random.default_rng(seed)
    a = np.exp(-dt)
    x = rng.normal(size=(N, B))
    acc = LagAccumulator(N, B, rows=np.arange(N), lag_steps=lag_steps, every=every, fields=("x",))
    hist = []
    for t in range(n_steps):
        x = a * x + np.sqrt(1 - a**2) * rng.normal(size=(N, B))
        if delay_pair is not None:
            i, j, d = delay_pair
            hist.append(x[i].copy())
            if t >= d:
                x[j] = hist[t - d]
        acc.update(t, xp.asarray(x), None)
    return acc, a


def test_lag_convention_with_delayed_copy():
    N, B, dt, d = 6, 4, 0.05, 10
    lags = [0, 5, 10, 15, 20]
    acc, a = run_ar1(N, B, 60000, dt, seed=1, lag_steps=lags, delay_pair=(0, 1, d))
    est = acc.estimates("x")
    # estimate at lag l is < x_i(t - l) x_j(t) >: with x_j(t) = x_i(t - d) this is a^{|d - l|}
    for l in lags:
        c = est[l][2]
        assert abs(c[0, 1] - a ** abs(d - l)) < 0.05, (l, c[0, 1], a ** abs(d - l))
        # reversed pair: < x_j(t - l) x_i(t) > = < x_i(t - l - d) x_i(t) > = a^{l + d}
        assert abs(c[1, 0] - a ** (l + d)) < 0.05, (l, c[1, 0], a ** (l + d))


def test_unbiased_and_noise_floor():
    N, B, dt, every = 200, 8, 0.05, 2
    n_steps = 20000
    lags = [0, 20]
    acc, a = run_ar1(N, B, n_steps, dt, seed=2, lag_steps=lags, every=every)
    est = acc.estimates("x")
    n = acc.n_samples
    rows = np.arange(N)
    off = offdiag_mask(rows, N)
    for l in lags:
        c1, c2, c = est[l]
        # unbiased diagonal: mean over sites within 5 sigma of e^{-l dt}
        truth = a ** l
        var_diag = (2.0 / (n * B)) * (1 + a ** (2 * every)) / (1 - a ** (2 * every))  # approx
        sig = np.sqrt(var_diag / N)
        assert abs(np.mean(np.diag(c)) - truth) < 5 * sig, (l, np.mean(np.diag(c)), truth, sig)
        # off-diagonal: truth is 0; empirical variance over pairs matches the analytic prediction
        var_pred = (1.0 / (n * B)) * (1 + a ** (2 * every)) / (1 - a ** (2 * every))
        var_emp = np.mean(c[off] ** 2)
        assert abs(var_emp / var_pred - 1) < 0.05, (l, var_emp, var_pred)
        # half-split floor matches the empirical variance
        floor = noise_floor(c1, c2, off)
        assert abs(floor / var_emp - 1) < 0.05, (l, floor, var_emp)
    # metrics() on truth = identity at lag 0: corrected MSE consistent with zero
    c1, c2, c = est[0]
    m = metrics(c, np.eye(N), c1, c2, rows, N)
    assert abs(m["mse_off_corr"]) < 0.1 * m["floor_off"]
    assert abs(m["mse_diag_corr"]) < 0.5 * m["floor_diag"]


def test_stepper_orders_on_linear_decay():
    # J = 0: dx/dt = -x, exact e^{-dt}. Local error must scale as dt^(p+1), p = 1, 2, 4.
    J = xp.zeros((1, 1))
    x0 = xp.ones((1, 1))
    for name, p in (("euler", 1), ("heun", 2), ("rk4", 4)):
        errs = []
        for dt in (0.1, 0.05):
            x1 = STEPPERS[name](x0, J, dt)
            errs.append(abs(float(x1[0, 0]) - np.exp(-dt)))
        ratio = errs[0] / errs[1]
        assert abs(np.log2(ratio) - (p + 1)) < 0.3, (name, ratio)
