"""Time-domain two-point accumulators and error metrics.

Convention (paper): C_ij(tau) = < f_i(t + tau) f_j(t) >.

LagAccumulator keeps a short history of a subset of rows, x_S(t - l dt), and at every
`every`-th step accumulates  x_S(t - l dt) x(t)^T  over trajectories. Its estimate is
therefore  C_ij(-l dt) = < x_i(t - l dt) x_j(t) >  for i in rows, all j, lags l >= 0.
Use C(tau)^T = C(-tau) to get positive lags. The B trajectories are split into two
halves with separate sums so that the estimation noise can be measured.
"""
import numpy as np
from .backend import xp, asnumpy


class LagAccumulator:
    def __init__(self, N, B, rows, lag_steps, every=1, fields=("x", "phi")):
        self.rows = xp.asarray(np.asarray(rows, dtype=np.int64))
        self.lag_steps = [int(l) for l in lag_steps]
        self.every = int(every)
        self.fields = tuple(fields)
        self.B = B
        self.halves = [slice(0, B // 2), slice(B // 2, B)]
        n_rows = len(rows)
        L = max(self.lag_steps) + 1
        # history of the row subset: hist[f][k] holds field f at step t with k = t % L
        self.hist = {f: xp.zeros((L, n_rows, B)) for f in self.fields}
        self.L = L
        # sums[f][(lag, half)] -> (n_rows, N)
        self.sums = {f: {(l, h): xp.zeros((n_rows, N)) for l in self.lag_steps for h in (0, 1)}
                     for f in self.fields}
        self.n_samples = 0
        self.t_seen = -1

    def update(self, t, x, phi):
        vals = {"x": x, "phi": phi}
        k = t % self.L
        for f in self.fields:
            self.hist[f][k] = vals[f][self.rows]
        self.t_seen = t
        if t < self.L - 1 or t % self.every != 0:
            return
        for f in self.fields:
            v = vals[f]
            for l in self.lag_steps:
                past = self.hist[f][(t - l) % self.L]
                for h, sl in enumerate(self.halves):
                    self.sums[f][(l, h)] += past[:, sl] @ v[:, sl].T
        self.n_samples += 1

    def estimates(self, field):
        """Return dict lag -> (C_half1, C_half2, C_mean), each (n_rows, N) numpy arrays."""
        out = {}
        for l in self.lag_steps:
            c1 = asnumpy(self.sums[field][(l, 0)]) / (self.n_samples * (self.B // 2))
            c2 = asnumpy(self.sums[field][(l, 1)]) / (self.n_samples * (self.B - self.B // 2))
            out[l] = (c1, c2, 0.5 * (c1 + c2))
        return out


def noise_floor(c1, c2, mask=None):
    """Per-entry noise variance of the mean 0.5 (c1 + c2), pooled over the entries in mask."""
    d = c1 - c2
    if mask is not None:
        d = d[mask]
    return 0.25 * np.mean(np.abs(d) ** 2)


def offdiag_mask(rows, N):
    """Boolean (n_rows, N) mask that is False on the diagonal entries (i == j)."""
    rows = np.asarray(rows)
    m = np.ones((len(rows), N), dtype=bool)
    m[np.arange(len(rows)), rows] = False
    return m


def metrics(c_emp, c_th, c1, c2, rows, N):
    """Plain and noise-corrected mean-square errors, off-diagonal and diagonal, plus the
    regression slope of the empirical on the theoretical off-diagonal entries."""
    rows = np.asarray(rows)
    off = offdiag_mask(rows, N)
    diag = ~off
    e = c_emp - c_th
    out = {}
    for name, m in (("off", off), ("diag", diag)):
        mse_plain = np.mean(np.abs(e[m]) ** 2)
        floor = noise_floor(c1, c2, m)
        out[f"mse_{name}_plain"] = mse_plain
        out[f"floor_{name}"] = floor
        out[f"mse_{name}_corr"] = mse_plain - floor
        out[f"n_{name}"] = int(m.sum())
    out["signal_off"] = np.mean(np.abs(c_th[off]) ** 2)
    xo, yo = c_th[off], c_emp[off]
    denom = np.sum(np.abs(xo) ** 2)
    out["slope_off"] = float(np.sum(np.conj(xo) * yo).real / denom) if denom > 0 else np.nan
    out["mean_diag_emp"] = float(np.mean(c_emp[diag]).real)
    out["mean_diag_th"] = float(np.mean(c_th[diag]).real)
    return out


def two_sided(dense, rows):
    """Two-sided lag functions for every ordered pair inside the row subset.

    `dense` is an (n_lag, n_rows, N) array of C_ij(-tau_l) for i in `rows`, all j, at lags
    tau_l = 0, dtau, ..., tau_max (the accumulator's and the theory's dense output). Because
    C_ij(tau) = C_ji(-tau), the restriction to columns j in `rows` gives both signs:
        tau <= 0:  C_{rows[a], rows[b]}(-tau_l) = dense[l, a, rows[b]]
        tau >= 0:  C_{rows[a], rows[b]}(+tau_l) = C_{rows[b], rows[a]}(-tau_l) = dense[l, b, rows[a]].
    Returns (out, asym): out has shape (n_rows, n_rows, 2 n_lag - 1) with the lag axis running
    from -tau_max to +tau_max; asym = max_ab |dense[0, a, rows[b]] - dense[0, b, rows[a]]| is the
    disagreement of the two tau = 0 estimates of the same quantity (round-off for the accumulator,
    Hermiticity of the transform for the theory)."""
    rows = np.asarray(rows)
    sub = np.asarray(dense)[:, :, rows]            # (n_lag, a, b) = C_{rows[a], rows[b]}(-tau_l)
    asym = float(np.max(np.abs(sub[0] - sub[0].T)))
    neg = sub[::-1]                                # lag index 0 -> tau = -tau_max, ..., n_lag - 1 -> tau = 0
    pos = sub.transpose(0, 2, 1)[1:]               # tau = +tau_l, l >= 1
    out = np.concatenate([neg, pos], axis=0).transpose(1, 2, 0)
    return out, asym


class DenseAccumulator:
    """Two-point functions among a site subset S at dense lags, all lags in one matrix product.

    Same estimator and convention as LagAccumulator with rows = S and the columns restricted to S:
        sums[f][h][l] += x_S(t - l dt)[:, half_h] @ x_S(t)[:, half_h]^T,   estimate C_ij(-l dt) for i, j in S,
    but the history slices of all lags are stacked into one (n_lag n_S, B/2) matrix so that every sample costs one
    matrix product per field and half instead of one per lag (a per-lag loop costs 74 ms per sample
    at 401 lags on a GPU, launch bound). Both signs of tau follow from C_ij(tau) = C_ji(-tau) inside S (two_sided).
    The ring buffer holds the last L = max lag + 1 steps of the subset (L x n_S x B per field)."""

    def __init__(self, N, B, sites, lag_steps, every=1, fields=("x", "phi")):
        sites = np.asarray(sites, dtype=np.int64)
        assert len(np.unique(sites)) == len(sites) and sites.min() >= 0 and sites.max() < N
        self.sites = sites
        self.sites_d = xp.asarray(sites)
        self.n_S, self.B = len(sites), int(B)
        self.lag_steps = [int(l) for l in lag_steps]
        assert all(l >= 0 for l in self.lag_steps)
        self.lag_d = xp.asarray(np.asarray(self.lag_steps, dtype=np.int64))
        self.n_lag = len(self.lag_steps)
        self.every = int(every)
        self.fields = tuple(fields)
        self.halves = [slice(0, B // 2), slice(B // 2, B)]
        self.L = max(self.lag_steps) + 1
        self.hist = {f: xp.zeros((self.L, self.n_S, B)) for f in self.fields}
        self.sums = {f: [xp.zeros((self.n_lag, self.n_S, self.n_S)) for _ in (0, 1)] for f in self.fields}
        self.n_samples = 0

    def update(self, t, x, phi):
        vals = {"x": x, "phi": phi}
        k = t % self.L
        for f in self.fields:
            self.hist[f][k] = vals[f][self.sites_d]
        if t < self.L - 1 or t % self.every != 0:
            return
        idx = (t - self.lag_d) % self.L                       # buffer slot of x_S(t - l dt) for every lag
        for f in self.fields:
            now = self.hist[f][k]                             # (n_S, B)
            past = self.hist[f][idx].reshape(self.n_lag * self.n_S, self.B)   # stacked history (n_lag n_S, B)
            for h, sl in enumerate(self.halves):
                self.sums[f][h] += (past[:, sl] @ now[:, sl].T).reshape(self.n_lag, self.n_S, self.n_S)
        self.n_samples += 1

    def estimates(self, field):
        """(h1, h2): the two half-batch estimates of C_ij(-l dt), each (n_lag, n_S, n_S) numpy, i = row, j = column."""
        n1, n2 = self.n_samples * (self.B // 2), self.n_samples * (self.B - self.B // 2)
        return asnumpy(self.sums[field][0]) / n1, asnumpy(self.sums[field][1]) / n2


def lag_window(taus, tau_max, kind="hann"):
    """Lag window k(tau) on |tau| <= tau_max, zero outside: 'hann' = cos^2(pi tau / 2 tau_max), 'rect' = 1."""
    taus = np.asarray(taus, dtype=float)
    inside = np.abs(taus) <= tau_max * (1 + 1e-12)
    if kind == "hann":
        k = np.cos(np.pi * taus / (2.0 * tau_max)) ** 2
    elif kind == "rect":
        k = np.ones_like(taus)
    else:
        raise ValueError(kind)
    return np.where(inside, k, 0.0)


def windowed_transform(C, taus, omegas, tau_max, kind="hann"):
    """Windowed discrete Fourier transform of two-sided lag functions (paper convention):
        F(omega) = dtau * sum_l k(tau_l) C(tau_l) exp(-i omega tau_l),   k = lag_window(taus, tau_max, kind).
    C has the lag axis last, e.g. (n_S, n_S, n_tau) from two_sided, on the uniform signed grid `taus`.
    Returns complex (n_omega,) + C.shape[:-1]. Applied identically to the simulation and the theory it compares
    the spectrum smoothed with the window's kernel; the sum is exact in expectation for the unbiased lag estimates."""
    taus = np.asarray(taus, dtype=float)
    d = np.diff(taus)
    assert np.allclose(d, d[0]), "uniform lag grid required"
    dtau = float(d[0])
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    ph = lag_window(taus, tau_max, kind) * np.exp(-1j * np.outer(omegas, taus)) * dtau   # (n_omega, n_tau)
    return np.tensordot(ph, np.asarray(C), axes=(1, -1))
