"""Fixed-J response theory of one (N, seed) (eq:responseintro in the time domain, 01_equations.md Sec. 3b):
R^x(tau) = expm(-(I - gain J) tau), columns S = the kicked sites of scripts/run_response.py (same seed rule).
CPU only (scipy expm, float64), so a job can run it in the background while the GPU simulates.
Output results/response_theory_{tag}.npz: S, taus_direct, x_direct (n_tau, n_col, N) by one expm per lag; taus_dense,
x_dense (n_lags, n_col, N) by the recursion from expm(-A dtau); gain, strip width (from the strip json when present,
else computed here) and the max difference between the two evaluations at the shared lags (a self-check, expected ~1e-13).
The phi theory is gain times the x theory. Layout: row a, column i = R_{i, S[a]}(tau).
usage: response_theory.py N SEED [SIGMA2] [--n_col 8 | --sites i j ...] [--taus 1 2 4] [--dtau 0.25] [--n_lags 16] [--tag TAG]"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["SCS_BACKEND"] = "numpy"
import numpy as np
from scs.network import make_J
from scs.response import theory_columns, theory_dense_columns, chosen_sites
from scs.theory import strip_width

p = argparse.ArgumentParser()
p.add_argument("N", type=int); p.add_argument("seed", type=int); p.add_argument("sigma2", type=float, nargs="?", default=9.0)
p.add_argument("--n_col", type=int, default=8, help="kicked sites drawn from the seed (as scripts/run_response.py)")
p.add_argument("--sites", type=int, nargs="+", default=None, help="the given kicked sites instead; must match the run")
p.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.0, 4.0])
p.add_argument("--dtau", type=float, default=0.25)
p.add_argument("--n_lags", type=int, default=16)
p.add_argument("--tag", default=None)
a = p.parse_args()

t0 = time.time()
tag = a.tag or f"N{a.N}_seed{a.seed}_sigma2_{a.sigma2:g}"
gain = float(np.load(f"results/dmft_sigma2_{a.sigma2:g}.npz")["gain"])
J = make_J(a.N, a.sigma2, a.seed)
S = chosen_sites(a.N, a.n_col, a.seed, a.sites)
strip_file = f"results/strip_N{a.N}_seed{a.seed}_sigma2_{a.sigma2:g}.json"
if os.path.exists(strip_file):
    sf = json.load(open(strip_file))
    assert sf["N"] == a.N and sf["seed"] == a.seed and abs(sf["gain"] - gain) < 1e-12, strip_file
    strip, strip_src = sf["strip_width"], strip_file
else:
    strip, _ = strip_width(J, gain); strip_src = f"computed here ({time.time() - t0:.0f}s)"
t1 = time.time()
direct = theory_columns(J, gain, a.taus, S)
t_direct = time.time() - t1
t1 = time.time()
dense = theory_dense_columns(J, gain, a.dtau, a.n_lags, S)
t_dense = time.time() - t1
taus_dense = a.dtau * np.arange(1, a.n_lags + 1)
diffs = {}
for tau in a.taus:
    k = int(round(tau / a.dtau))
    if abs(k * a.dtau - tau) < 1e-9 and 1 <= k <= a.n_lags:
        diffs[tau] = float(np.max(np.abs(dense[k - 1] - direct[tau])))
out = f"results/response_theory_{tag}.npz"
np.savez(out, S=S, taus_direct=np.array(a.taus), x_direct=np.stack([direct[t] for t in a.taus]), taus_dense=taus_dense, x_dense=dense,
         gain=gain, strip=strip, N=a.N, seed=a.seed, sigma2=a.sigma2, t_direct=t_direct, t_dense=t_dense,
         recursion_vs_direct=np.array([diffs.get(t, np.nan) for t in a.taus]))
print(f"N={a.N} seed={a.seed}: strip {strip:.4f} [{strip_src}]; direct expm {t_direct:.1f}s, dense recursion {t_dense:.1f}s; "
      f"recursion vs direct max|diff| " + ", ".join(f"tau={t:g}: {d:.1e}" for t, d in diffs.items()) + f"; -> {out} ({time.time() - t0:.0f}s)")
