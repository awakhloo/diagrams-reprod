"""Fourier-domain two-point test: DenseAccumulator, lag windows, windowed transform, theory column option,
and the end-to-end check on a linear noisy map with an exactly known spectrum."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scs.twopoint import LagAccumulator, DenseAccumulator, lag_window, windowed_transform, two_sided, metrics
from scs.theory import theory_lags
from scs.backend import xp, asnumpy


def _stream(N, B, n_steps, seed):
    """A correlated random stream (AR(1) with a random mixing) for exact accumulator comparisons."""
    rng = np.random.default_rng(seed)
    A = 0.9 * np.eye(N) + 0.05 * rng.normal(size=(N, N)) / np.sqrt(N)
    x = rng.normal(size=(N, B))
    for _ in range(n_steps):
        x = A @ x + rng.normal(size=(N, B))
        yield x


def test_dense_equals_lag_accumulator():
    N, B, n_steps = 30, 8, 400
    sites = np.array([2, 5, 11, 17, 23, 29])
    lags = [0, 3, 7, 12, 20]
    dense = DenseAccumulator(N, B, sites, lags, every=2, fields=("x", "phi"))
    ref = LagAccumulator(N, B, sites, lags, every=2, fields=("x", "phi"))
    for t, x in enumerate(_stream(N, B, n_steps, 0)):
        xd = xp.asarray(x); pd = xp.tanh(xd)
        dense.update(t, xd, pd); ref.update(t, xd, pd)
    assert dense.n_samples == ref.n_samples > 0
    for f in ("x", "phi"):
        h1, h2 = dense.estimates(f)
        est = ref.estimates(f)
        for il, l in enumerate(lags):
            c1, c2, _ = est[l]
            assert np.allclose(h1[il], c1[:, sites], rtol=1e-12, atol=1e-13)
            assert np.allclose(h2[il], c2[:, sites], rtol=1e-12, atol=1e-13)


def test_dense_two_sided_asymmetry_zero():
    N, B = 12, 6
    sites = np.arange(N)
    lags = list(range(0, 9))
    acc = DenseAccumulator(N, B, sites, lags, every=1, fields=("x",))
    for t, x in enumerate(_stream(N, B, 200, 1)):
        acc.update(t, xp.asarray(x), None)
    h1, h2 = acc.estimates("x")
    out, asym = two_sided(0.5 * (h1 + h2), sites)
    assert out.shape == (N, N, 2 * len(lags) - 1) and asym == 0.0


def test_lag_window_values():
    taus = np.arange(-10, 10.01, 0.5)
    k = lag_window(taus, 5.0, "hann")
    assert k[taus == 0] == 1.0 and np.all(k[np.abs(taus) > 5] == 0) and np.all(k[np.abs(taus) == 5] < 1e-30)
    assert np.allclose(k[np.abs(taus) == 2.5], 0.5)
    r = lag_window(taus, 5.0, "rect")
    assert np.all(r[np.abs(taus) <= 5] == 1.0) and np.all(r[np.abs(taus) > 5] == 0)


def test_windowed_transform_definition_and_fft():
    rng = np.random.default_rng(2)
    n_S, L, dtau = 4, 40, 0.25
    taus = np.arange(-L, L + 1) * dtau
    C = rng.normal(size=(n_S, n_S, len(taus)))
    omegas = np.array([0.0, 0.3, 1.0])
    for kind in ("hann", "rect"):
        F = windowed_transform(C, taus, omegas, 5.0, kind)
        k = lag_window(taus, 5.0, kind)
        loop = np.array([[[dtau * sum(k[l] * C[a, b, l] * np.exp(-1j * w * taus[l]) for l in range(len(taus)))
                           for b in range(n_S)] for a in range(n_S)] for w in omegas])
        assert F.shape == (3, n_S, n_S) and np.allclose(F, loop, atol=1e-12)
    # rectangular window at omega = 0 is the plain lag sum over |tau| <= tau_max
    F0 = windowed_transform(C, taus, [0.0], 5.0, "rect")[0]
    assert np.allclose(F0, dtau * C[:, :, np.abs(taus) <= 5].sum(-1))
    # against numpy's FFT: with omega_m = 2 pi m / (n dtau), sum_{l=-L..L} = e^{+i omega_m L dtau} fft(k C)[m]
    n = len(taus)
    om = 2 * np.pi * np.arange(n) / (n * dtau)
    F = windowed_transform(C, taus, om, 5.0, "hann")
    kc = lag_window(taus, 5.0, "hann") * C
    fft = np.fft.fft(kc, axis=-1) * dtau * np.exp(1j * om * L * dtau)
    assert np.allclose(F, np.moveaxis(fft, -1, 0), atol=1e-10)
    # Hermiticity: C_ab(tau) = C_ba(-tau) implies F_ab(omega) = conj(F_ba(omega))
    Cs = C + np.transpose(C, (1, 0, 2))[:, :, ::-1]
    F = windowed_transform(Cs, taus, omegas, 5.0, "hann")
    assert np.allclose(F, np.conj(np.transpose(F, (0, 2, 1))))


def test_theory_cols_option():
    N, rng = 40, np.random.default_rng(3)
    J = xp.asarray(rng.normal(0, 0.5 / np.sqrt(N), (N, N)))
    gain = 0.7
    C0x = lambda w: 2.0 / (1 + w ** 2); C0phi = lambda w: 1.0 / (1 + 4 * w ** 2)
    rows = np.array([1, 7, 20, 33])
    cols = np.array([33, 4, 20, 9, 1])                              # unsorted, containing some of the rows
    taus = np.array([0.0, -1.0, 2.5])
    for G in (0.0, 0.3):
        full = theory_lags(J, gain, C0x, C0phi, 1.0, rows, taus, 0.05, 6.0, G=G)
        sub = theory_lags(J, gain, C0x, C0phi, 1.0, rows, taus, 0.05, 6.0, G=G, cols=cols)
        for f in ("x", "phi"):
            assert sub[f].shape == (3, 4, 5)
            assert np.allclose(sub[f], full[f][:, :, cols], rtol=1e-13, atol=1e-14)


def test_linear_map_windowed_spectrum():
    """x_{t+1} = A x_t + s z_t: <x(t + l) x(t)^T> = A^l S exactly (S the discrete Lyapunov solution). The windowed
    transform of the exact lags is the reference; the accumulator's transform must agree within its noise floor
    (corrected MSE consistent with zero: over 40 independent runs of this configuration the corrected-to-floor ratio
    was +0.02 +- 0.02 with scatter 0.15 over the 552 off-diagonal entries, so 0.6 is a 4 sigma bound), and the
    transposed reference, C_ij(tau) mistaken for C_ij(-tau), must be rejected at omega > 0 (it flips the imaginary part)."""
    N, B, dt, s = 24, 16, 0.05, 0.3
    rng = np.random.default_rng(7)
    J = rng.normal(0, 0.6 / np.sqrt(N), (N, N))
    A = (1 - dt) * np.eye(N) + dt * J
    S = solve_discrete_lyapunov(A, s ** 2 * np.eye(N))
    sites = np.arange(N)
    lags = list(range(0, 81, 2))                                     # 0 .. 80 steps, spacing 0.1, horizon 4
    exact = np.stack([(np.linalg.matrix_power(A, l) @ S).T for l in lags])   # C_ij(-l dt)
    acc = DenseAccumulator(N, B, sites, lags, every=2, fields=("x",))
    x = rng.normal(0, 1, (N, B))
    for _ in range(3000):
        x = A @ x + s * rng.normal(size=(N, B))
    for t in range(120000):
        x = A @ x + s * rng.normal(size=(N, B))
        acc.update(t, xp.asarray(x), None)
    h1, h2 = acc.estimates("x")
    two_th, asym_th = two_sided(exact, sites)
    two_1, _ = two_sided(h1, sites); two_2, _ = two_sided(h2, sites)
    assert asym_th < 1e-14
    taus = np.arange(-80, 81, 2) * dt
    omegas = [0.0, 1.0, 2.0]
    off = ~np.eye(N, dtype=bool)
    for kind in ("hann", "rect"):
        F1 = windowed_transform(two_1, taus, omegas, 3.0, kind)
        F2 = windowed_transform(two_2, taus, omegas, 3.0, kind)
        Ft = windowed_transform(two_th, taus, omegas, 3.0, kind)
        Ft_T = np.transpose(Ft, (0, 2, 1))                           # = conj(Ft): the wrong lag-sign convention
        for iw, w in enumerate(omegas):
            m = metrics(0.5 * (F1[iw] + F2[iw]), Ft[iw], F1[iw], F2[iw], sites, N)
            assert abs(m["mse_off_corr"]) < 0.6 * m["floor_off"], (kind, w, m)
            assert abs(m["mse_diag_corr"]) < 1.5 * m["floor_diag"], (kind, w, m)
            if w > 0:
                wrong = metrics(0.5 * (F1[iw] + F2[iw]), Ft_T[iw], F1[iw], F2[iw], sites, N)
                gap = np.mean(np.abs(Ft[iw] - Ft_T[iw])[off] ** 2)
                assert gap > 50 * m["floor_off"], ("test not discriminating", kind, w, gap / m["floor_off"])
                assert wrong["mse_off_corr"] > 20 * m["floor_off"], (kind, w, wrong)
