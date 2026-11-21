"""Response functions by perturbed copies and their fixed-J theory (eq:responseintro).

Estimator. Next to the B reference trajectories x (N, B), n_col perturbed copies of every trajectory are advanced with
the same RK4 stepper; copy a is kicked on site S[a] by eps at the start of each window of n_steps steps,
    R^x_{i, S[a]}(k dt)   ~   < ( x^{eps, a}_i(t + k dt) - x_i(t + k dt) ) / eps >,
    R^phi_{i, S[a]}(k dt) ~   < ( tanh(x^{eps, a}_i) - tanh(x_i) ) / eps >,
averaged over windows and trajectories (the attractor ensemble). Forward difference: relative bias 1.8 eps (measured),
round-off ~1e-12/eps; eps = 1e-4. The state array is Z = [x | copy_0 | ... | copy_{n_col-1}]
of shape (N, B (1 + n_col)), so every RK4 stage is one matrix product. Sums are kept for two halves of the trajectories
(noise floor). Estimates come out in the (n_col, N) layout, row a, column i = R_{i, S[a]}: the layout of
twopoint.metrics with rows = S (the kicked sites are the "rows").

Theory (01_equations.md Sec. 3b): R^x(tau) = expm(-(I - gain J) tau), R^phi(tau) = gain R^x(tau), columns S only.
"""
import numpy as np
from scipy.linalg import expm
from .backend import xp, asnumpy
from .network import STEPPERS


def kick_sites(N, n_col, seed):
    """The n_col kicked sites of a draw, a function of the seed only (the simulation and the theory must agree)."""
    return np.sort(np.random.default_rng(seed + 4242).choice(N, n_col, replace=False))


def chosen_sites(N, n_col, seed, sites=None):
    """The kicked sites of a run: the given list (distinct indices in [0, N), returned sorted like kick_sites so that the
    row order of the estimates is the same rule in every script) or, when sites is None, kick_sites(N, n_col, seed)."""
    if sites is None:
        return kick_sites(N, n_col, seed)
    S = np.array(sorted(int(s) for s in sites), dtype=np.int64)
    assert len(S) > 0 and len(np.unique(S)) == len(S), "sites must be distinct"
    assert S.min() >= 0 and S.max() < N, "sites must be indices in [0, N)"
    return S


class KickAccumulator:
    def __init__(self, N, B, S, readout_steps, fields=("x", "phi")):
        self.N, self.B = int(N), int(B)
        self.S = np.asarray(S, dtype=np.int64)
        self.n_col = len(self.S)
        self.readout_steps = [int(k) for k in readout_steps]
        self.fields = tuple(fields)
        self.halves = [slice(0, B // 2), slice(B // 2, B)]
        self.sums = {f: {(k, h): xp.zeros((N, self.n_col)) for k in self.readout_steps for h in (0, 1)} for f in self.fields}
        self.n_win = 0
        self.gain_sum = xp.zeros(N)          # sum over readouts and reference trajectories of phi'(x_i) = 1 - tanh^2
        self.n_gain = 0

    def run(self, x, J, eps, dt, n_win, n_steps, stepper="rk4"):
        """Advance the reference x (N, B) through n_win windows of n_steps steps, kicked copies alongside; accumulate
        the finite differences at the readout steps. Returns the advanced reference (unperturbed)."""
        step = STEPPERS[stepper]
        B, n_col = self.B, self.n_col
        for _ in range(n_win):
            Z = xp.concatenate([x] * (1 + n_col), axis=1)          # block 0 = reference, block 1 + a = copy a
            for a in range(n_col):
                Z[self.S[a], B * (1 + a): B * (2 + a)] += eps
            for k in range(1, n_steps + 1):
                Z = step(Z, J, dt)
                if k in self.readout_steps:
                    self._readout(k, Z, eps)
            x = Z[:, :B]
            self.n_win += 1
        return x

    def _readout(self, k, Z, eps):
        B, n_col = self.B, self.n_col
        ref = Z[:, :B]
        cop = Z[:, B:].reshape(self.N, n_col, B)                     # [i, a, b] = copy a, trajectory b
        vals = {}
        if "x" in self.fields:
            vals["x"] = (cop - ref[:, None, :]) / eps
        if "phi" in self.fields:
            tr = xp.tanh(ref)
            vals["phi"] = (xp.tanh(cop) - tr[:, None, :]) / eps
            self.gain_sum += (1.0 - tr ** 2).sum(axis=1); self.n_gain += B
        for f, D in vals.items():
            for h, sl in enumerate(self.halves):
                self.sums[f][(k, h)] += D[:, :, sl].sum(axis=2)

    def gain_site(self):
        """Time mean of phi'(x_i) over the readouts and the reference trajectories, (N,) numpy."""
        return asnumpy(self.gain_sum) / max(self.n_gain, 1)

    def kicks_per_half(self):
        return [self.n_win * (self.B // 2), self.n_win * (self.B - self.B // 2)]

    def estimates(self, field):
        """dict k -> (R_half1, R_half2, R_mean), each (n_col, N) numpy: row a, column i = R_{i, S[a]}(k dt)."""
        n1, n2 = self.kicks_per_half()
        out = {}
        for k in self.readout_steps:
            r1 = asnumpy(self.sums[field][(k, 0)]).T / n1
            r2 = asnumpy(self.sums[field][(k, 1)]).T / n2
            out[k] = (r1, r2, 0.5 * (r1 + r2))
        return out


def theory_columns(J, gain, taus, S):
    """Direct evaluation: dict tau -> (n_col, N) numpy array, row a, column i = [expm(-(I - gain J) tau)]_{i, S[a]},
    the x response; the phi response is gain times it. One scipy expm (CPU, float64) per lag."""
    Jn = asnumpy(J)
    A = np.eye(Jn.shape[0]) - gain * Jn
    S = np.asarray(S, dtype=np.int64)
    return {float(t): expm(-A * float(t))[:, S].T.copy() for t in taus}


def theory_dense_columns(J, gain, dtau, n_lags, S):
    """Recursion: (n_lags, n_col, N) array at tau = dtau, 2 dtau, ..., n_lags dtau from one expm(-A dtau) and one
    N^2 n_col product per lag (R((k+1) dtau)[:, S] = R(dtau) R(k dtau)[:, S])."""
    Jn = asnumpy(J)
    E = expm(-(np.eye(Jn.shape[0]) - gain * Jn) * float(dtau))
    S = np.asarray(S, dtype=np.int64)
    C = E[:, S]
    out = [C.T.copy()]
    for _ in range(int(n_lags) - 1):
        C = E @ C
        out.append(C.T.copy())
    return np.stack(out)
