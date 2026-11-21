"""Four-point (pair-product) accumulators and the Wick-residual metrics for eq:simplemoment, m = 4.

Sites: a subset S of n_S sites. Pairs: all a < b inside S, P = n_S (n_S - 1)/2 of them, indexed by
(ia, ib) into S in numpy's triu order. At every `every`-th step the pair products
    Y_ab(t) = f_a(t) f_b(t),   f = x or phi,   shape (P, B),
are formed and their Gram matrices accumulated per trajectory group k (K contiguous groups of B/K
trajectories, independent by chaos):
    G[f][l][k] += Y(t - l)[:, grp_k] @ Y(t)[:, grp_k]^T
so that G / n estimates  m^l_{ab,cd} = < f_a(t-l) f_b(t-l) f_c(t) f_d(t) >  (l = 0: equal time; the
matrix is then symmetric and every quadruple of distinct sites appears at its three pairings).
Alongside, the two-point functions inside S per group and lag,
    D[f][l][k][i, j] = < f_i(t-l) f_j(t) >          (the LagAccumulator convention, columns restricted to S),
from which the Wick prediction of the same quantity is
    W^l_{ab,cd} = D_0[a,b] D_0[c,d] + D_l[a,c] D_l[b,d] + D_l[a,d] D_l[b,c].
Entries whose four sites are not distinct are masked out (eq:simplemoment needs distinct indices).

Residual and noise (reduce_block): per group R_k = m_k - W_k with W_k from the group's own two-point
functions (so the noise of the Wick input and its correlation with the noise of m are part of R_k), the
estimate is R = mean_k R_k, the noise variance of R per entry is the between-group variance / K, pooled
over entries (the floor), the corrected MSE is the raw MSE minus the floor, and the uncertainty of raw,
floor and corrected MSE comes from the jackknife over groups (the groups are the independent units, so
correlations between quadruples are automatically accounted for). With a noise-free Wick input (theory)
W is the same for every group.
"""
import numpy as np
from .backend import xp, asnumpy


def pair_arrays(n_S):
    """(ia, ib): first and second site (indices into S) of every pair a < b, numpy triu order."""
    ia, ib = np.triu_indices(n_S, k=1)
    return ia.astype(np.int64), ib.astype(np.int64)


def pair_index(a, b, n_S):
    """Position of the pair (a, b), a < b, in the triu order of pair_arrays."""
    a = np.asarray(a); b = np.asarray(b)
    assert np.all(a < b)
    return a * n_S - a * (a + 1) // 2 + (b - a - 1)


class PairProductAccumulator:
    def __init__(self, N, B, sites, K, lag_steps, every=1, fields=("x", "phi")):
        sites = np.sort(np.asarray(sites, dtype=np.int64))
        assert len(np.unique(sites)) == len(sites) and sites.min() >= 0 and sites.max() < N
        assert B % K == 0, "B must be a multiple of the number of groups K"
        self.sites = sites
        self.n_S = len(sites)
        self.B, self.K, self.Bk = B, int(K), B // int(K)
        self.ia, self.ib = pair_arrays(self.n_S)
        self.P = len(self.ia)
        self.sites_d, self.ia_d, self.ib_d = xp.asarray(sites), xp.asarray(self.ia), xp.asarray(self.ib)
        self.lag_steps = [int(l) for l in lag_steps]
        assert self.lag_steps[0] == 0 and all(l >= 0 for l in self.lag_steps), "lag 0 first, lags >= 0"
        self.every = int(every)
        self.fields = tuple(fields)
        self.L = max(self.lag_steps) + 1
        self.hist = {f: xp.zeros((self.L, self.n_S, B)) for f in self.fields}
        self.gram = {f: {l: xp.zeros((self.K, self.P, self.P)) for l in self.lag_steps} for f in self.fields}
        self.two = {f: {l: xp.zeros((self.K, self.n_S, self.n_S)) for l in self.lag_steps} for f in self.fields}
        self.n_samples = 0

    def _groups(self, a):
        """(n, B) -> (K, n, Bk): the K contiguous column blocks."""
        return a.reshape(a.shape[0], self.K, self.Bk).transpose(1, 0, 2)

    def update(self, t, x, phi):
        vals = {"x": x, "phi": phi}
        k = t % self.L
        for f in self.fields:
            self.hist[f][k] = vals[f][self.sites_d]
        if t < self.L - 1 or t % self.every != 0:
            return
        for f in self.fields:
            now = self.hist[f][k]                                        # (n_S, B), field f at time t on S
            Ynow = self._groups(now[self.ia_d] * now[self.ib_d])          # (K, P, Bk)
            Gnow = self._groups(now)                                     # (K, n_S, Bk)
            for l in self.lag_steps:
                if l == 0:
                    Ypast, Gpast = Ynow, Gnow
                else:
                    past = self.hist[f][(t - l) % self.L]                # field f at time t - l on S
                    Ypast = self._groups(past[self.ia_d] * past[self.ib_d])
                    Gpast = self._groups(past)
                self.gram[f][l] += Ypast @ Ynow.transpose(0, 2, 1)
                self.two[f][l] += Gpast @ Gnow.transpose(0, 2, 1)
        self.n_samples += 1

    @property
    def norm(self):
        return self.n_samples * self.Bk

    def two_point(self, field, lag):
        """(K, n_S, n_S) device array D_l[k][i, j] = < f_i(t - l) f_j(t) > per group."""
        return self.two[field][lag] / self.norm

    def moment(self, field, lag):
        """(K, P, P) device array m^l per group (a view of the sums divided by the norm; large)."""
        return self.gram[field][lag] / self.norm


def wick_rows(D0, Dl, ia, ib, r0, r1):
    """Wick prediction for the pair rows r0:r1 against all pair columns.
    D0, Dl: (K, n_S, n_S) (or (n_S, n_S) for a group-independent input); ia, ib: device pair arrays.
    Returns (K, r1 - r0, P) (or (r1 - r0, P)):  D0[a,b] D0[c,d] + Dl[a,c] Dl[b,d] + Dl[a,d] Dl[b,c]."""
    ar, br = ia[r0:r1], ib[r0:r1]
    C0 = D0[..., ia, ib]                                                   # (..., P): pair values at lag 0
    W = C0[..., r0:r1, None] * C0[..., None, :]
    Dl_a = xp.take(Dl, ar, axis=-2)                                        # (..., nr, n_S)
    Dl_b = xp.take(Dl, br, axis=-2)
    W += xp.take(Dl_a, ia, axis=-1) * xp.take(Dl_b, ib, axis=-1)
    W += xp.take(Dl_a, ib, axis=-1) * xp.take(Dl_b, ia, axis=-1)
    return W


def distinct_mask_rows(ia, ib, r0, r1):
    """(r1 - r0, P) boolean: the four sites of the entry (row pair, column pair) are distinct."""
    ar, br = ia[r0:r1, None], ib[r0:r1, None]
    ca, cb = ia[None, :], ib[None, :]
    return (ar != ca) & (ar != cb) & (br != ca) & (br != cb)


def reduce_block(gram_sum, norm, D0, Dl, ia, ib, D0_th=None, Dl_th=None, chunk=256, sample=0, seed=0):
    """Metrics of one block (field, lag). gram_sum: (K, P, P) device sums; norm: samples x Bk; D0, Dl: (K, n_S, n_S)
    device two-point functions per group; optional theory two-point functions D0_th, Dl_th: (n_S, n_S).
    Returns dict with, for each Wick version v in {'emp', 'th'}:
        mse_raw_v, floor_v, mse_corr_v (+ '_se' jackknife standard errors), slope_v (regression of m on W over the
        masked entries), rms_wick_v, and the sample arrays (indices and per-group values of `sample` random entries).
    Also n (masked entries), mean_m, rms_m, mean_R_v (should be ~0 by symmetry), and the common-mode diagnostic:
    eps_v = per-group multiplicative deviation eps_k = sum_q (R_kq - R_q) R_q / sum_q R_q^2 (a network-wide slow fluctuation
    of the correlation amplitude shows up as a scalar factor on every group's residual), and floor_idio_v = the floor after
    removing that scalar mode. Both floors are unbiased noise estimates of R; the split says how much of the
    between-group scatter is coherent across quadruples."""
    K, P, _ = gram_sum.shape
    versions = {"emp": None}
    if D0_th is not None:
        versions["th"] = (xp.asarray(D0_th), xp.asarray(Dl_th))
    S = {v: {k: 0.0 for k in ("n", "R2", "V", "W2", "WR", "Wm", "R", "m", "m2")} for v in versions}
    JR2 = {v: xp.zeros(K) for v in versions}     # leave-one-group-out sums of R_(-k)^2
    JV = {v: xp.zeros(K) for v in versions}      # leave-one-group-out sums of the per-entry variance estimate
    DR = {v: xp.zeros(K) for v in versions}      # sum_q d_kq R_q (common-mode diagnostic)
    D2 = {v: 0.0 for v in versions}              # sum_q sum_k d_kq^2
    rng = np.random.default_rng(seed)
    n_rows_total = P
    samp = {v: {"rows": [], "cols": [], "m_k": [], "W_k": []} for v in versions}
    per_row = max(1, int(np.ceil(sample / P))) if sample else 0
    for r0 in range(0, n_rows_total, chunk):
        r1 = min(P, r0 + chunk)
        mask = distinct_mask_rows(ia, ib, r0, r1)                          # (nr, P)
        m_k = gram_sum[:, r0:r1, :] / norm                                 # (K, nr, P)
        for v, th in versions.items():
            if th is None:
                W_k = wick_rows(D0, Dl, ia, ib, r0, r1)                    # (K, nr, P)
            else:
                W_k = wick_rows(th[0], th[1], ia, ib, r0, r1)[None]        # (1, nr, P), same for every group
            R_k = m_k - W_k                                                # (K, nr, P)
            R = R_k.mean(axis=0)
            d = R_k - R
            T = (d ** 2).sum(axis=0)                                       # sum_j (R_j - R)^2 per entry
            V = T / (K * (K - 1))                                          # unbiased variance of the mean R per entry
            Wbar = W_k.mean(axis=0)
            mbar = m_k.mean(axis=0)
            s = S[v]
            s["n"] += float(mask.sum())
            s["R2"] += float((R[mask] ** 2).sum())
            s["V"] += float(V[mask].sum())
            s["W2"] += float((Wbar[mask] ** 2).sum())
            s["WR"] += float((Wbar[mask] * R[mask]).sum())
            s["Wm"] += float((Wbar[mask] * mbar[mask]).sum())
            s["R"] += float(R[mask].sum())
            s["m"] += float(mbar[mask].sum())
            s["m2"] += float((mbar[mask] ** 2).sum())
            # jackknife over groups: R_(-k) = R - d_k/(K-1);  sum_{j != k} (R_j - R_(-k))^2 = T - d_k^2 K/(K-1)
            Rj = R[None] - d / (K - 1)                                     # (K, nr, P)
            Vj = (T[None] - d ** 2 * (K / (K - 1))) / ((K - 1) * (K - 2))
            JR2[v] += ((Rj ** 2) * mask[None]).sum(axis=(1, 2))
            JV[v] += (Vj * mask[None]).sum(axis=(1, 2))
            DR[v] += ((d * R[None]) * mask[None]).sum(axis=(1, 2))
            D2[v] += float((T * mask).sum())
            if per_row:
                # a few random masked entries per row block, for figures (indices and per-group values)
                rows_np, cols_np = np.nonzero(asnumpy(mask))
                if len(rows_np):
                    pick = rng.choice(len(rows_np), min(per_row * (r1 - r0), len(rows_np)), replace=False)
                    rr, cc = rows_np[pick], cols_np[pick]
                    samp[v]["rows"].append(rr + r0); samp[v]["cols"].append(cc)
                    samp[v]["m_k"].append(asnumpy(m_k[:, rr, cc]).T)      # (n_pick, K)
                    Wp = W_k[:, rr, cc]
                    samp[v]["W_k"].append(asnumpy(Wp).T)                  # (n_pick, K), or (n_pick, 1) for the theory input
    out = {"n": int(S["emp"]["n"]), "K": K}
    ia_np, ib_np = asnumpy(ia), asnumpy(ib)
    for v in versions:
        s = S[v]; n = s["n"]
        raw, floor = s["R2"] / n, s["V"] / n
        out[f"mse_raw_{v}"] = raw
        out[f"floor_{v}"] = floor
        out[f"mse_corr_{v}"] = raw - floor
        out[f"slope_{v}"] = s["Wm"] / s["W2"]
        out[f"mse_orth_{v}"] = (s["m2"] - s["Wm"] ** 2 / s["W2"]) / n     # raw MSE of m - slope W: the part of R not proportional to W
        out[f"rms_wick_{v}"] = np.sqrt(s["W2"] / n)
        out[f"mean_R_{v}"] = s["R"] / n
        eps = asnumpy(DR[v]) / s["R2"]
        out[f"eps_{v}"] = [float(e) for e in eps]
        # the scalar mode removes one degree of freedom per group out of n, not one per entry: same denominator as the floor
        out[f"floor_idio_{v}"] = (D2[v] - float(np.sum(eps ** 2)) * s["R2"]) / (K * (K - 1)) / n
        # jackknife standard errors (leave one group out; the K pseudo-values are the K leave-one-out statistics)
        jr, jv = asnumpy(JR2[v]) / n, asnumpy(JV[v]) / n
        for name, vals in (("mse_raw", jr), ("floor", jv), ("mse_corr", jr - jv)):
            out[f"{name}_{v}_se"] = float(np.sqrt((K - 1) / K * np.sum((vals - vals.mean()) ** 2)))
        if sample and samp[v]["rows"]:
            rows = np.concatenate(samp[v]["rows"]); cols = np.concatenate(samp[v]["cols"])
            out[f"sample_{v}"] = {"a": ia_np[rows], "b": ib_np[rows], "c": ia_np[cols], "d": ib_np[cols],
                                  "m_k": np.concatenate(samp[v]["m_k"]), "W_k": np.concatenate(samp[v]["W_k"])}
    out["mean_m"] = S["emp"]["m"] / S["emp"]["n"]
    out["rms_m"] = np.sqrt(S["emp"]["m2"] / S["emp"]["n"])
    return out


def pairing_consistency(gram_sum, n_S, n_check=2000, seed=0):
    """Equal-time, same-field block: the three pairings (ab|cd), (ac|bd), (ad|bc) of a quadruple a<b<c<d are the
    same time average and must agree to rounding. Returns the max over n_check random quadruples of
    max|difference| / max|value| (also checks the symmetry of the matrix)."""
    rng = np.random.default_rng(seed)
    quads = np.sort(np.array([rng.choice(n_S, 4, replace=False) for _ in range(n_check)]), axis=1)
    a, b, c, d = quads.T
    g = asnumpy(gram_sum.sum(axis=0))
    v1 = g[pair_index(a, b, n_S), pair_index(c, d, n_S)]
    v2 = g[pair_index(a, c, n_S), pair_index(b, d, n_S)]
    v3 = g[pair_index(a, d, n_S), pair_index(b, c, n_S)]
    v1t = g[pair_index(c, d, n_S), pair_index(a, b, n_S)]
    vals = np.stack([v1, v2, v3, v1t])
    return float(np.max(np.abs(vals - vals[0]) / np.maximum(np.abs(vals[0]), 1e-300)))


def two_point_offdiag(D_k, D_th=None):
    """Off-diagonal two-point check inside S from the per-group (K, n_S, n_S) estimates: pooled noise floor of the
    mean, and, if D_th is given, raw and corrected MSE against it plus the regression slope."""
    D_k = asnumpy(D_k); K, n_S, _ = D_k.shape
    off = ~np.eye(n_S, dtype=bool)
    D = D_k.mean(axis=0)
    V = ((D_k - D) ** 2).sum(axis=0) / (K * (K - 1))
    out = {"floor": float(V[off].mean()), "rms": float(np.sqrt((D[off] ** 2).mean())), "mean_diag": float(np.mean(np.diag(D)))}
    if D_th is not None:
        D_th = np.asarray(D_th)
        E = D - D_th
        out["mse_raw"] = float((E[off] ** 2).mean())
        out["mse_corr"] = out["mse_raw"] - out["floor"]
        out["rms_th"] = float(np.sqrt((D_th[off] ** 2).mean()))
        out["slope"] = float((D_th[off] * D[off]).sum() / (D_th[off] ** 2).sum())
        out["mean_diag_th"] = float(np.mean(np.diag(D_th)))
    return out
