"""Four-point accumulator and Wick residual: exact indexing, Wick assembly, jackknife algebra, and a Gaussian
process (linear noisy map) where eq:simplemoment holds exactly at every lag."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scs.fourpoint import (PairProductAccumulator, pair_arrays, pair_index, wick_rows, distinct_mask_rows,
                           reduce_block, pairing_consistency, two_point_offdiag)
from scs.backend import xp, asnumpy


def test_pair_index_matches_triu():
    n_S = 9
    ia, ib = pair_arrays(n_S)
    assert np.all(pair_index(ia, ib, n_S) == np.arange(len(ia)))


def test_accumulator_exact_on_deterministic_sequences():
    N, B, K, every, lags, n_steps = 9, 8, 4, 2, [0, 3], 41
    sites = np.array([0, 2, 3, 5, 7, 8])
    n_S = len(sites)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n_steps, N, B)) + 0.5 * np.sin(0.3 * np.arange(n_steps))[:, None, None]
    PHI = np.tanh(X) + 0.1 * rng.normal(size=X.shape)           # any second field
    acc = PairProductAccumulator(N, B, sites, K, lags, every=every)
    for t in range(n_steps):
        acc.update(t, xp.asarray(X[t]), xp.asarray(PHI[t]))
    ia, ib = pair_arrays(n_S)
    L = max(lags) + 1
    ts = [t for t in range(n_steps) if t >= L - 1 and t % every == 0]
    assert acc.n_samples == len(ts)
    Bk = B // K
    for f, F in (("x", X), ("phi", PHI)):
        FS = F[:, sites, :]
        for l in lags:
            g_ref = np.zeros((K, len(ia), len(ia))); d_ref = np.zeros((K, n_S, n_S))
            for t in ts:
                Ynow = FS[t][ia] * FS[t][ib]; Ypast = FS[t - l][ia] * FS[t - l][ib]
                for k in range(K):
                    sl = slice(k * Bk, (k + 1) * Bk)
                    g_ref[k] += Ypast[:, sl] @ Ynow[:, sl].T
                    d_ref[k] += FS[t - l][:, sl] @ FS[t][:, sl].T
            g = asnumpy(acc.gram[f][l]); d = asnumpy(acc.two[f][l])
            assert np.max(np.abs(g - g_ref)) < 1e-10 * np.max(np.abs(g_ref)), (f, l)
            assert np.max(np.abs(d - d_ref)) < 1e-10 * np.max(np.abs(d_ref)), (f, l)
            # explicit meaning of one entry: sum over sample times and trajectories of f_a(t-l) f_b(t-l) f_c(t) f_d(t)
            a, b, c, dd = 1, 4, 0, 3
            val = sum(FS[t - l][a, j] * FS[t - l][b, j] * FS[t][c, j] * FS[t][dd, j] for t in ts for j in range(Bk))
            assert abs(g[0, pair_index(a, b, n_S), pair_index(c, dd, n_S)] - val) < 1e-10 * abs(val)


def test_wick_rows_and_mask_against_brute_force():
    n_S, K = 7, 3
    rng = np.random.default_rng(5)
    D0 = rng.normal(size=(K, n_S, n_S)); Dl = rng.normal(size=(K, n_S, n_S))   # not symmetric: index roles matter
    ia, ib = pair_arrays(n_S)
    P = len(ia)
    W = asnumpy(wick_rows(xp.asarray(D0), xp.asarray(Dl), xp.asarray(ia), xp.asarray(ib), 0, P))
    Wg = asnumpy(wick_rows(xp.asarray(D0[1]), xp.asarray(Dl[1]), xp.asarray(ia), xp.asarray(ib), 2, 9))
    mask = asnumpy(distinct_mask_rows(xp.asarray(ia), xp.asarray(ib), 0, P))
    for p in range(P):
        for q in range(P):
            a, b, c, d = ia[p], ib[p], ia[q], ib[q]
            for k in range(K):
                ref = D0[k, a, b] * D0[k, c, d] + Dl[k, a, c] * Dl[k, b, d] + Dl[k, a, d] * Dl[k, b, c]
                assert abs(W[k, p, q] - ref) < 1e-12
            assert mask[p, q] == (len({a, b, c, d}) == 4)
            if 2 <= p < 9:
                ref = D0[1, a, b] * D0[1, c, d] + Dl[1, a, c] * Dl[1, b, d] + Dl[1, a, d] * Dl[1, b, c]
                assert abs(Wg[p - 2, q] - ref) < 1e-12


def _direct_stats(m_k, W_k, mask):
    """Raw MSE, floor and corrected MSE of R = mean_k (m_k - W_k) over masked entries, by the definitions."""
    K = m_k.shape[0]
    R_k = m_k - W_k
    R = R_k.mean(0)
    V = ((R_k - R) ** 2).sum(0) / (K * (K - 1))
    raw = (R[mask] ** 2).mean(); fl = V[mask].mean()
    return raw, fl, raw - fl


def test_reduce_block_matches_direct_definitions_and_loop_jackknife():
    n_S, K, norm = 8, 5, 7.0
    rng = np.random.default_rng(11)
    ia, ib = pair_arrays(n_S); P = len(ia)
    gram = rng.normal(size=(K, P, P))
    D0 = rng.normal(size=(K, n_S, n_S)); Dl = rng.normal(size=(K, n_S, n_S))
    D0t = rng.normal(size=(n_S, n_S)); Dlt = rng.normal(size=(n_S, n_S))
    out = reduce_block(xp.asarray(gram), norm, xp.asarray(D0), xp.asarray(Dl), xp.asarray(ia), xp.asarray(ib),
                       D0_th=D0t, Dl_th=Dlt, chunk=5, sample=50)
    mask = asnumpy(distinct_mask_rows(xp.asarray(ia), xp.asarray(ib), 0, P))
    m_k = gram / norm
    W_emp = asnumpy(wick_rows(xp.asarray(D0), xp.asarray(Dl), xp.asarray(ia), xp.asarray(ib), 0, P))
    W_th = asnumpy(wick_rows(xp.asarray(D0t), xp.asarray(Dlt), xp.asarray(ia), xp.asarray(ib), 0, P))[None]
    assert out["n"] == mask.sum()
    for v, W in (("emp", W_emp), ("th", np.repeat(W_th, K, axis=0))):
        raw, fl, corr = _direct_stats(m_k, W, mask)
        assert abs(out[f"mse_raw_{v}"] - raw) < 1e-12 and abs(out[f"floor_{v}"] - fl) < 1e-12 and abs(out[f"mse_corr_{v}"] - corr) < 1e-12
        Wbar = W.mean(0); mbar = m_k.mean(0)
        assert abs(out[f"slope_{v}"] - (Wbar[mask] * mbar[mask]).sum() / (Wbar[mask] ** 2).sum()) < 1e-12
        assert abs(out[f"rms_wick_{v}"] - np.sqrt((Wbar[mask] ** 2).mean())) < 1e-12
        sl = out[f"slope_{v}"]
        assert abs(out[f"mse_orth_{v}"] - ((mbar[mask] - sl * Wbar[mask]) ** 2).mean()) < 1e-12
        # jackknife by explicit leave-one-group-out loops
        loo = np.array([_direct_stats(np.delete(m_k, k, 0), np.delete(W, k, 0), mask) for k in range(K)])  # (K, 3)
        for j, name in enumerate(("mse_raw", "floor", "mse_corr")):
            se = np.sqrt((K - 1) / K * np.sum((loo[:, j] - loo[:, j].mean()) ** 2))
            assert abs(out[f"{name}_{v}_se"] - se) < 1e-12, (v, name, out[f"{name}_{v}_se"], se)
        # common-mode diagnostic: per-group multiplicative deviation and the floor with that scalar mode removed
        R_k = m_k - W; R = R_k.mean(0); dk = (R_k - R)[:, mask]; Rm = R[mask]
        eps = (dk * Rm).sum(1) / (Rm ** 2).sum()
        assert np.allclose(out[f"eps_{v}"], eps)
        fl_idio = ((dk - eps[:, None] * Rm[None]) ** 2).sum() / (K * (K - 1)) / mask.sum()
        # pure iid noise, no common mode: floor and floor_idio must agree (up to one dof in n per group)
        gn = W_emp + 0.1 * rng.normal(size=(K, P, P)); on = reduce_block(xp.asarray(gn), 1.0, xp.asarray(D0), xp.asarray(Dl), xp.asarray(ia), xp.asarray(ib), chunk=7)
        assert abs(on["floor_idio_emp"] / on["floor_emp"] - 1) < 3.0 / mask.sum() ** 0.5 + 8.0 / mask.sum()
        assert abs(out[f"floor_idio_{v}"] - fl_idio) < 1e-12 * max(1, fl_idio)
        s = out[f"sample_{v}"]
        p = pair_index(s["a"], s["b"], n_S); q = pair_index(s["c"], s["d"], n_S)
        assert np.all(mask[p, q])
        assert np.allclose(s["m_k"], m_k[:, p, q].T) and np.allclose(s["W_k"], W[:, p, q].T)


def _linear_map_run(N, B, K, dt, s, J, n_trans, n_steps, lags, seed):
    rng = np.random.default_rng(seed)
    A = (1 - dt) * np.eye(N) + dt * J
    x = rng.normal(0, 1, (N, B))
    for _ in range(n_trans):
        x = A @ x + s * rng.normal(size=(N, B))
    acc = PairProductAccumulator(N, B, np.arange(N), K, lags, every=2, fields=("x",))
    for t in range(n_steps):
        x = A @ x + s * rng.normal(size=(N, B))
        acc.update(t, xp.asarray(x), None)
    return acc


def test_gaussian_linear_map_wick_exact_at_all_lags():
    """x_{t+1} = A x_t + s z_t is Gaussian, so <x_a x_b x_c x_d> is exactly Wick with the exact covariance
    S (equal time) and A^l S (lagged); the accumulator convention gives D_l[i, j] = <x_i(t-l) x_j(t)> = (A^l S)^T_ij."""
    N, B, K, dt, s, lag = 14, 32, 4, 0.05, 0.3, 20
    rng = np.random.default_rng(7)
    J = rng.normal(0, 0.6 / np.sqrt(N), (N, N))
    A = (1 - dt) * np.eye(N) + dt * J
    S = solve_discrete_lyapunov(A, s ** 2 * np.eye(N))
    D0t = S; Dlt = (np.linalg.matrix_power(A, lag) @ S).T
    lags = [0, lag]
    ia, ib = pair_arrays(N)
    acc1 = _linear_map_run(N, B, K, dt, s, J, 3000, 120000, lags, seed=1)
    acc2 = _linear_map_run(N, B, K, dt, s, J, 3000, 120000, lags, seed=2)
    assert pairing_consistency(acc1.gram["x"][0], N) < 1e-10
    P = len(ia); mask = asnumpy(distinct_mask_rows(xp.asarray(ia), xp.asarray(ib), 0, P))
    for l in lags:
        outs = []
        for acc in (acc1, acc2):
            D0, Dl = acc.two_point("x", 0), acc.two_point("x", l)
            outs.append(reduce_block(acc.gram["x"][l], acc.norm, D0, Dl, xp.asarray(ia), xp.asarray(ib), D0_th=D0t, Dl_th=Dlt if l else D0t))
        o = outs[0]
        for v in ("emp", "th"):
            # the corrected MSE must be consistent with zero (Wick exact), the slope with one
            assert abs(o[f"mse_corr_{v}"]) < 4 * o[f"mse_corr_{v}_se"] + 1e-14, (l, v, o[f"mse_corr_{v}"], o[f"mse_corr_{v}_se"])
            assert abs(o[f"slope_{v}"] - 1) < 0.05, (l, v, o[f"slope_{v}"])
            # the floor must be the actual noise variance: two independent runs differ by 2 x floor per entry
            m1 = asnumpy(acc1.gram["x"][l]) / acc1.norm; m2 = asnumpy(acc2.gram["x"][l]) / acc2.norm
            if v == "th":
                Wt = asnumpy(wick_rows(xp.asarray(D0t), xp.asarray(Dlt if l else D0t), xp.asarray(ia), xp.asarray(ib), 0, P))
                R1 = m1.mean(0) - Wt; R2 = m2.mean(0) - Wt
                diff = ((R1 - R2)[mask] ** 2).mean()
                assert abs(diff / (o["floor_th"] + outs[1]["floor_th"]) - 1) < 0.25, (l, diff, o["floor_th"], outs[1]["floor_th"])
        if l > 0:
            # discrimination: the wrong lag direction (transposed lagged covariance) must be rejected
            wrong = reduce_block(acc1.gram["x"][l], acc1.norm, acc1.two_point("x", 0), acc1.two_point("x", l),
                                 xp.asarray(ia), xp.asarray(ib), D0_th=D0t, Dl_th=Dlt.T)
            assert wrong["mse_corr_th"] > 10 * wrong["mse_corr_th_se"] and wrong["mse_corr_th"] > 5 * wrong["floor_th"], wrong
            # also the two-point check inside S sees the direction
            tp = two_point_offdiag(acc1.two_point("x", l), Dlt); tpw = two_point_offdiag(acc1.two_point("x", l), Dlt.T)
            assert abs(tp["mse_corr"]) < 5 * tp["floor"] * np.sqrt(2.0 / (N * (N - 1))) * 4 + 1e-14 and tpw["mse_corr"] > 10 * tp["floor"]
