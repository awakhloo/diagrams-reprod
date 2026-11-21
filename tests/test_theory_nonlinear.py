"""Checks of the phi branch and of h_x != 0 in theory.py, which the linear tests cannot exercise.
(1) Ginibre trace identity: (1/N) tr M M^dagger -> 1/(1 - sigma^2 |R0phi|^2), hence
    (1/N) tr C^phi(w) -> C0phi(w) and (1/N) tr C^x(w) -> C0x(w) (using C0x = |R0x|^2 sigma^2 C0phi at G = 0).
    At w = 1 the finite-N correction is < 1 % already at N = 500.
(2) The resolvent shortcut J M = (M - I)/R0phi against explicit matrix products of eq:shen-hu2.
(3) Realness and C(-tau) = C(tau)^T of the lag transform for both fields."""
import numpy as np
from scs.theory import theory_spectrum, theory_lags, strip_width
from scs.network import make_J
from scs.backend import xp, asnumpy
from scs import dmft

SIGMA2 = 9.0   # the study's coupling; the identities hold for any sigma^2


def _dmft():
    return dmft.solve(SIGMA2, dt=0.05, T_D=200.0, n_gh=64, tol=1e-10)


def test_trace_identity_both_fields():
    sol = _dmft()
    N = 500
    J = make_J(N, SIGMA2, 1)
    for w in (1.0, 2.0):
        C0x = sol.spectrum("x", np.array([w]))[0]; C0p = sol.spectrum("phi", np.array([w]))[0]
        out = theory_spectrum(J, sol.gain, C0x, C0p, SIGMA2, w)
        trx = float(xp.trace(out["x"]).real) / N; trp = float(xp.trace(out["phi"]).real) / N
        # The Ginibre trace identity holds only as N -> inf. Measured finite-N deviation of the trace ratio
        # at N = 500 over
        # 6 seeds, x field, w = 1: -1.8 % .. +1.1 % (this seed: +1.08 %), phi: -1.3 % .. +0.8 %; at N = 1000
        # 0.02-0.4 %. A sign or factor error in h_x, h_phi would give an O(1) deviation, so 5 % discriminates.
        assert abs(trp / C0p - 1) < 0.05, (w, trp, C0p)
        assert abs(trx / C0x - 1) < 0.05, (w, trx, C0x)
        # Hermitian
        assert float(xp.max(xp.abs(out["x"] - out["x"].conj().T))) < 1e-10
        assert float(xp.max(xp.abs(out["phi"] - out["phi"].conj().T))) < 1e-10


def test_resolvent_shortcut_matches_explicit_formula():
    sol = _dmft()
    N = 120
    J = make_J(N, SIGMA2, 2)
    w = 0.3
    C0x = sol.spectrum("x", np.array([w]))[0]; C0p = sol.spectrum("phi", np.array([w]))[0]
    G = 0.7
    out = theory_spectrum(J, sol.gain, C0x, C0p, SIGMA2, w, G=G)
    R0x = 1 / (1 + 1j * w); R0p = sol.gain * R0x
    Jn = asnumpy(J)
    M = np.linalg.inv(np.eye(N) - R0p * Jn)
    MMd = M @ M.conj().T
    Cphi = C0p * (1 - SIGMA2 * abs(R0p) ** 2) * MMd
    hx = (C0p - sol.gain ** 2 * C0x) / (1 + w ** 2)
    Cx = hx * (Jn @ MMd @ Jn.T) + G / (1 + w ** 2) * MMd
    assert np.max(np.abs(asnumpy(out["phi"]) - Cphi)) < 1e-10 * np.max(np.abs(Cphi))
    assert np.max(np.abs(asnumpy(out["x"]) - Cx)) < 1e-10 * np.max(np.abs(Cx))


def test_lag_transform_real_and_transposes_for_phi():
    sol = _dmft()
    N = 80
    J = make_J(N, SIGMA2, 3)
    width, _ = strip_width(J, sol.gain)
    assert width > 0.0, width
    Cx = lambda w: sol.spectrum("x", w); Cp = lambda w: sol.spectrum("phi", w)
    th = theory_lags(J, sol.gain, Cx, Cp, SIGMA2, np.arange(N), [0.0, 1.5, -1.5], 0.01, 10.0, strip=width)
    for f in ("x", "phi"):
        assert np.max(np.abs(th[f][0] - th[f][0].T)) < 1e-8 * np.max(np.abs(th[f][0]))
        assert np.max(np.abs(th[f][1] - th[f][2].T)) < 1e-8 * np.max(np.abs(th[f][1]))
    # refuse an unstable J
    try:
        theory_lags(J, sol.gain, Cx, Cp, SIGMA2, np.arange(N), [0.0], 0.01, 10.0, strip=-0.01)
        assert False, "should have raised"
    except ValueError:
        pass
