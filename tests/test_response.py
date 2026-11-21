"""Response-function estimator (perturbed copies) and theory (matrix exponential).

1. Layout: the batched accumulator reproduces a hand-written loop (one copy at a time, one trajectory at a time).
2. Finite differences converge to the tangent propagator as eps -> 0 (bias linear in eps), for x and for phi.
3. Linear network (state at the fixed point 0, tanh(eps u) = eps u to O(eps^3)): the estimate equals expm(-(I - J) tau)
   columns to the RK4 accuracy, both halves agree exactly, and the transposed theory is rejected.
4. scipy expm on our A = I - gain J agrees with a fine RK4 integration of du/dt = -A u.
5. The forward transform of expm(-A tau) reproduces the manuscript's R^x(omega) = R0x (I - R0phi J)^{-1}.
6. The dense-lag recursion equals the direct expm at the scaling lags.
"""
import numpy as np
from scipy.linalg import expm
from scs.backend import xp, asnumpy
from scs.network import make_J, initial_state, simulate, step_rk4, rhs
from scs.response import KickAccumulator, theory_columns, theory_dense_columns, kick_sites
from scs.twopoint import metrics, noise_floor

GAIN = 0.319421   # DMFT gain at sigma^2 = 9 (results/dmft_sigma2_9.npz)


def tangent_window(x, J, S, dt, n_steps, readout_steps):
    """Reference: tangent RK4 (exact derivative of the RK4 map) from state x (N, B), unit kicks on S.
    Returns dict k -> (u_x (N, B, n_col), u_phi (N, B, n_col))."""
    N, B = x.shape
    U = xp.zeros((N, B, len(S)))
    for a, j in enumerate(S):
        U[j, :, a] = 1.0

    def rhs_tan(x, U):
        W = (1.0 - xp.tanh(x) ** 2)[:, :, None] * U
        return -U + (J @ W.reshape(N, -1)).reshape(U.shape)

    out = {}
    for k in range(1, n_steps + 1):
        k1 = rhs(x, J);               l1 = rhs_tan(x, U)
        x2 = x + 0.5 * dt * k1;       U2 = U + 0.5 * dt * l1
        k2 = rhs(x2, J);              l2 = rhs_tan(x2, U2)
        x3 = x + 0.5 * dt * k2;       U3 = U + 0.5 * dt * l2
        k3 = rhs(x3, J);              l3 = rhs_tan(x3, U3)
        x4 = x + dt * k3;             U4 = U + dt * l3
        k4 = rhs(x4, J);              l4 = rhs_tan(x4, U4)
        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        U = U + (dt / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
        if k in readout_steps:
            out[k] = (asnumpy(U), asnumpy((1.0 - xp.tanh(x) ** 2)[:, :, None] * U))
    return out


def test_batched_accumulator_matches_manual_loop():
    N, B, dt, eps = 30, 3, 0.05, 1e-4            # B = 3: halves of 1 and 2 trajectories
    J = make_J(N, 9.0, 5)
    S = [2, 11, 25]
    x0 = simulate(initial_state(N, B, 3), J, dt, 400)
    readout = [7, 20]
    acc = KickAccumulator(N, B, S, readout)
    x_end = acc.run(x0, J, eps, dt, n_win=2, n_steps=20)
    assert acc.kicks_per_half() == [2, 4]
    # manual: one trajectory and one copy at a time, two windows back to back
    man = {f: {(k, h): np.zeros((N, len(S))) for k in readout for h in (0, 1)} for f in ("x", "phi")}
    x_ref_end = np.zeros((N, B))
    for b in range(B):
        h = 0 if b < B // 2 else 1
        xr = asnumpy(x0)[:, b:b + 1]
        for _ in range(2):
            for a, j in enumerate(S):
                xc = xr.copy(); xc[j, 0] += eps
                xr_w = xr.copy()
                for k in range(1, 21):
                    xc = asnumpy(step_rk4(xp.asarray(xc), J, dt)); xr_w = asnumpy(step_rk4(xp.asarray(xr_w), J, dt))
                    if k in readout:
                        man["x"][(k, h)][:, a] += (xc[:, 0] - xr_w[:, 0]) / eps
                        man["phi"][(k, h)][:, a] += (np.tanh(xc[:, 0]) - np.tanh(xr_w[:, 0])) / eps
            xr = xr_w
        x_ref_end[:, b] = xr[:, 0]
    n = acc.kicks_per_half()
    for f in ("x", "phi"):
        est = acc.estimates(f)
        for k in readout:
            for h in (0, 1):
                ref = man[f][(k, h)].T / n[h]
                assert np.max(np.abs(est[k][h] - ref)) < 1e-9, (f, k, h, np.max(np.abs(est[k][h] - ref)))
            assert np.max(np.abs(est[k][2] - 0.5 * (est[k][0] + est[k][1]))) == 0.0
    assert np.max(np.abs(asnumpy(x_end) - x_ref_end)) < 1e-12   # the reference is never perturbed


def test_finite_difference_converges_to_tangent():
    N, B, dt = 40, 2, 0.05
    J = make_J(N, 9.0, 1)
    S = kick_sites(N, 4, 1)
    x0 = simulate(initial_state(N, B, 1), J, dt, 1000)
    readout = [20, 40, 80]
    tan = tangent_window(x0, J, S, dt, 80, readout)
    errs = {}
    for eps in (1e-2, 1e-3, 1e-4):
        acc = KickAccumulator(N, B, S, readout)
        acc.run(x0, J, eps, dt, n_win=1, n_steps=80)
        for fi, f in enumerate(("x", "phi")):
            est = acc.estimates(f)
            for k in readout:
                truth = tan[k][fi].mean(axis=1).T                    # (n_col, N), mean over the B trajectories
                errs[(eps, f, k)] = np.sqrt(np.mean((est[k][2] - truth) ** 2) / np.mean(truth ** 2))
    for f in ("x", "phi"):
        for k in readout:
            r1 = errs[(1e-2, f, k)] / errs[(1e-3, f, k)]
            r2 = errs[(1e-3, f, k)] / errs[(1e-4, f, k)]
            assert 5 < r1 < 20 and 5 < r2 < 20, (f, k, r1, r2)      # bias linear in eps
            assert errs[(1e-4, f, k)] < 1e-4, (f, k, errs[(1e-4, f, k)])
            assert errs[(1e-2, f, k)] > 1e-3, (f, k, errs[(1e-2, f, k)])   # the test discriminates


def test_linear_network_equals_expm_and_rejects_transpose():
    N, B, dt, eps = 40, 4, 0.05, 1e-4
    rng = np.random.default_rng(11)
    J = xp.asarray(0.5 * rng.normal(0, 1 / np.sqrt(N), (N, N)))      # spectral radius ~0.5, non-symmetric
    S = [3, 17, 30]
    x0 = xp.zeros((N, B))                                          # fixed point: tanh(eps u) = eps u + O(eps^3)
    readout = [20, 40, 80]
    acc = KickAccumulator(N, B, S, readout)
    acc.run(x0, J, eps, dt, n_win=1, n_steps=80)
    th = theory_columns(J, 1.0, [1.0, 2.0, 4.0], S)                  # phi' = 1 at the fixed point: "gain" 1
    A = np.eye(N) - asnumpy(J)
    for k, tau in zip(readout, (1.0, 2.0, 4.0)):
        r1, r2, r = acc.estimates("x")[k]
        scale = np.max(np.abs(th[tau]))
        assert np.max(np.abs(r - th[tau])) / scale < 1e-6, (tau, np.max(np.abs(r - th[tau])) / scale)
        assert np.max(np.abs(r1 - r2)) == 0.0                        # identical trajectories: halves agree exactly
        wrong = expm(-A * tau)[S, :]                                 # rows S instead of columns S
        assert np.max(np.abs(r - wrong)) / scale > 1e-2
        m = metrics(r, th[tau], r1, r2, S, N)
        assert m["floor_off"] == 0.0 and abs(m["slope_off"] - 1) < 1e-5 and m["mse_off_plain"] < (1e-6 * scale) ** 2
        rp = acc.estimates("phi")[k][2]                              # phi = tanh: phi'(0) = 1, same response
        assert np.max(np.abs(rp - th[tau])) / scale < 1e-6


def test_expm_matches_fine_rk4_of_linear_ode():
    N = 60
    J = asnumpy(make_J(N, 9.0, 1))
    A = np.eye(N) - GAIN * J
    S = [7, 40]
    th = theory_columns(J, GAIN, [2.0], S)[2.0]
    dt, n = 1e-3, 2000
    u = np.zeros((N, len(S)))
    for a, j in enumerate(S):
        u[j, a] = 1.0
    f = lambda v: -A @ v
    for _ in range(n):
        k1 = f(u); k2 = f(u + 0.5 * dt * k1); k3 = f(u + 0.5 * dt * k2); k4 = f(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.max(np.abs(u.T - th)) < 1e-9, np.max(np.abs(u.T - th))


def test_time_domain_theory_matches_frequency_domain_formula():
    N = 30
    J = asnumpy(make_J(N, 9.0, 3))
    dtau, n_lags = 0.01, 20000                                       # tau up to 200; e^{-A tau} decays with rate >= strip
    dense = theory_dense_columns(J, GAIN, dtau, n_lags, np.arange(N))   # [k, a, i] = R_{i a}((k + 1) dtau)
    R_tau = np.concatenate([np.eye(N)[None], dense.transpose(0, 2, 1)])   # tau = 0 .. n_lags dtau
    taus = np.arange(n_lags + 1) * dtau
    assert np.max(np.abs(R_tau[-1])) < 1e-12
    for w in (0.0, 0.7, 2.0):
        ph = np.exp(-1j * w * taus); wts = np.full(len(taus), dtau); wts[0] = wts[-1] = 0.5 * dtau
        R_w = np.tensordot(wts * ph, R_tau, axes=(0, 0))
        R0x = 1.0 / (1.0 + 1j * w); R0phi = GAIN * R0x
        R_manuscript = R0x * np.linalg.inv(np.eye(N) - R0phi * J)
        assert np.max(np.abs(R_w - R_manuscript)) < 1e-4, (w, np.max(np.abs(R_w - R_manuscript)))


def test_dense_recursion_equals_direct_expm():
    N = 50
    J = asnumpy(make_J(N, 9.0, 2))
    S = kick_sites(N, 8, 2)
    dense = theory_dense_columns(J, GAIN, 0.25, 16, S)
    direct = theory_columns(J, GAIN, [1.0, 2.0, 4.0], S)
    for k, tau in ((4, 1.0), (8, 2.0), (16, 4.0)):
        assert np.max(np.abs(dense[k - 1] - direct[tau])) < 1e-12


def test_kick_sites_deterministic():
    S = kick_sites(100, 8, 3)
    assert np.array_equal(S, kick_sites(100, 8, 3)) and len(set(S.tolist())) == 8
    assert np.all(np.diff(S) > 0) and S.min() >= 0 and S.max() < 100
    assert not np.array_equal(S, kick_sites(100, 8, 4))


def test_chosen_sites_given_or_drawn():
    """--sites replaces the seed's draw in the run and the theory; sorted, distinct, in range; None falls back to kick_sites."""
    from scs.response import chosen_sites
    assert np.array_equal(chosen_sites(100, 8, 3), kick_sites(100, 8, 3))
    S = chosen_sites(100, 8, 3, sites=[71, 5, 33])
    assert S.dtype == np.int64 and S.tolist() == [5, 33, 71]              # n_col of the draw is ignored; sorted like kick_sites
    for bad in ([5, 5, 33], [5, 100], [-1, 3], []):
        try:
            chosen_sites(100, 8, 3, sites=bad)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_columns_on_rows_layout():
    """The dense response data keep R_{rows[b], S[a]}(tau) = theory[:, a, rows[b]]: the kicked site is the column j of R_ij,
    the read-out row is i. Checked entry by entry against the direct expm."""
    N = 50
    J = asnumpy(make_J(N, 9.0, 2))
    S = np.array([3, 20, 41]); rows = np.array([1, 3, 20, 33, 49])
    dense = theory_dense_columns(J, GAIN, 0.25, 8, S)                    # (8, 3, N)
    sub = dense[:, :, rows]                                              # (8, 3, 5): [k, a, b] = R_{rows[b], S[a]}
    E = expm(-(np.eye(N) - GAIN * J) * 2.0)                              # tau = 2 = lag index 7
    for a, j in enumerate(S):
        for b, i in enumerate(rows):
            assert abs(sub[7, a, b] - E[i, j]) < 1e-12
    assert abs(sub[7, 1, 2] - E[20, 20]) < 1e-12 and abs(sub[7, 0, 1] - E[3, 3]) < 1e-12   # diagonal entries where i == j
