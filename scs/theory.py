"""Fixed-connectivity two-point theory (eq:introshenhu) and its transform to the time domain.

For a given J and the single-site DMFT quantities (gain, C0x(w), C0phi(w), G(w)), with
    R0x = 1/(1 + i w),   R0phi = gain R0x,   M(w) = (I - R0phi J)^{-1},
the theory reads (01_equations.md, Sec. 3a)
    C^phi(w) = h_phi M M^dagger,                     h_phi = C0phi (1 - sigma^2 |R0phi|^2)
    C^x(w)   = h_x (JM)(JM)^dagger + |R0x|^2 G M M^dagger,   h_x = |R0x|^2 (C0phi - gain^2 C0x)
with JM = (M - I)/R0phi (resolvent identity), so that (JM)(JM)^dagger = (M M^dagger - M - M^dagger + I)/|R0phi|^2 and only
one N^3 product per frequency is needed. Time domain, using C(-w) = conj(C(w)):
    C(tau) = (1/pi) Re int_0^inf dw e^{i w tau} C(w),
evaluated by the trapezoidal rule on a uniform grid w = 0, dw, ..., w_max.
Only the rows `rows` of the N x N matrices are returned.
"""
import numpy as np
from .backend import xp, asnumpy


def theory_spectrum(J, gain, C0x_w, C0phi_w, sigma2, w, G=0.0, rows=None, fields=("x", "phi")):
    """The theory matrices C^x(w), C^phi(w) at one frequency (rows `rows` of each, all rows if None).
    C0x_w, C0phi_w are the DMFT spectra at this frequency. Returns dict field -> complex (n_rows, N) array
    on the device, plus "A" (I - R0phi J) for diagnostics."""
    N = J.shape[0]
    R0x = 1.0 / (1.0 + 1j * w)
    R0phi = gain * R0x
    A = (-R0phi) * J
    idx = xp.arange(N)
    A[idx, idx] += 1.0                      # A = I - R0phi J
    M = xp.linalg.inv(A)
    MS = M if rows is None else M[rows]
    MMd = MS @ M.conj().T                   # rows of M M^dagger: the one large product
    out = {"A": A}
    if "phi" in fields:
        h_phi = C0phi_w * (1.0 - sigma2 * abs(R0phi) ** 2)
        out["phi"] = h_phi * MMd
    if "x" in fields:
        # (JM)(JM)^dagger with JM = (M - I)/R0phi (resolvent identity) equals (M M^dagger - M - M^dagger + I)/|R0phi|^2,
        # so the x field needs no second N^3 product; h_x/|R0phi|^2 = (C0phi - gain^2 C0x)/gain^2.
        MSd = (M if rows is None else M[:, rows]).conj().T   # rows of M^dagger
        E = MMd - MS - MSd
        r = idx if rows is None else rows
        E[xp.arange(len(r)), r] += 1.0
        out["x"] = ((C0phi_w - gain ** 2 * C0x_w) / gain ** 2) * E + (abs(R0x) ** 2 * G) * MMd
    return out


def theory_lags(J, gain, C0x_fn, C0phi_fn, sigma2, rows, taus, dw, w_max, G=0.0,
                fields=("x", "phi"), diagnostics=False, verbose=False, strip=None, cols=None):
    """Return dict field -> real array (n_tau, n_rows, N) of C^th_{ij}(tau) for i in rows,
    tau in taus (signed, in time units). C0x_fn, C0phi_fn map a frequency array to spectra.
    `cols`: keep only these columns j (array of site indices; default all N), so that the output is
    (n_tau, n_rows, n_cols) and the lag accumulation stays small for dense lags on a subset.
    G is a constant white-noise spectrum (0 for the undriven network).
    `strip` = min_k Re(1 - gain lambda_k) if known (see strip_width): the formula is meaningless when
    it is <= 0 (M has a pole in the lower half plane; the fixed-J linearization is unstable), so we refuse."""
    if strip is not None and strip <= 0:
        raise ValueError(f"strip width {strip:.4f} <= 0: the fixed-J theory is unstable for this J; refusing to evaluate")
    N = J.shape[0]
    rows_d = xp.asarray(np.asarray(rows, dtype=np.int64))
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    n_w = int(round(w_max / dw)) + 1
    omegas = np.linspace(0.0, w_max, n_w)
    wts = np.full(n_w, dw / np.pi); wts[0] *= 0.5; wts[-1] *= 0.5
    C0x = C0x_fn(omegas); C0phi = C0phi_fn(omegas)
    cols_np = None if cols is None else np.asarray(cols, dtype=np.int64)
    cols_d = None if cols is None else xp.asarray(cols_np)
    acc = {f: xp.zeros((len(taus), len(rows), N if cols is None else len(cols_np))) for f in fields}
    diag = {"omega": omegas, "smin": [], "cond": []} if diagnostics else None
    for k, w in enumerate(omegas):
        Cw = theory_spectrum(J, gain, C0x[k], C0phi[k], sigma2, w, G=G, rows=rows_d, fields=fields)
        ph = wts[k] * np.exp(1j * w * taus)  # (n_tau,)
        for f in fields:
            Cf = Cw[f] if cols is None else Cw[f][:, cols_d]
            for it in range(len(taus)):
                acc[f][it] += (complex(ph[it]) * Cf).real  # python scalar: safe for both numpy and cupy
        if diagnostics:
            sv = xp.linalg.svd(Cw["A"], compute_uv=False)
            diag["smin"].append(float(sv[-1])); diag["cond"].append(float(sv[0] / sv[-1]))
        if verbose and k % 100 == 0:
            print(f"  omega {w:.3f} ({k+1}/{n_w})", flush=True)
    out = {f: asnumpy(acc[f]) for f in fields}
    if "x" in fields and G != 0.0:
        # Analytic tail of the white-noise term beyond w_max: for w -> inf, M -> I, so
        # |R0x|^2 G M M^dagger -> G/(1 + w^2) I. Add (G/pi) int_{w_max}^inf cos(w tau)/(1 + w^2) dw
        # on the diagonal (rows). The neglected off-diagonal tail is O(G |J_ij| / w_max^2).
        rows_np = np.asarray(rows)
        if cols is None:
            ra, ca = np.arange(len(rows_np)), rows_np
        else:   # diagonal entries present in the kept columns: (row position, column position)
            pos = {int(c): p for p, c in enumerate(cols_np)}
            sel = [(a, pos[int(r)]) for a, r in enumerate(rows_np) if int(r) in pos]
            ra, ca = (np.array([a for a, _ in sel], dtype=int), np.array([c for _, c in sel], dtype=int)) if sel else (np.array([], int), np.array([], int))
        for it, tau in enumerate(taus):
            out["x"][it, ra, ca] += G * _tail_integral(tau, w_max)
    if diagnostics:
        diag["smin"] = np.array(diag["smin"]); diag["cond"] = np.array(diag["cond"])
        out["diagnostics"] = diag
    return out


def _tail_integral(tau, w_max):
    """(1/pi) int_{w_max}^inf cos(w tau)/(1 + w^2) dw."""
    from scipy.integrate import quad
    if tau == 0.0:
        return (np.pi / 2 - np.arctan(w_max)) / np.pi
    val, _ = quad(lambda w: 1.0 / (1.0 + w * w), w_max, np.inf, weight="cos", wvar=abs(tau))
    return val / np.pi


def strip_width(J, gain):
    """Half-width of the analyticity strip of C(w) = h M M^dagger: min_k Re(1 - gain lambda_k).
    M(w) has poles at w = +i(1 - gain lambda_k) (upper half plane, causal) and M^dagger at their
    mirror images; the strip is symmetric. Requires an eigendecomposition (N <= a few thousand)."""
    lam = np.linalg.eigvals(asnumpy(J))
    return float(np.min(1.0 - gain * lam.real)), lam
