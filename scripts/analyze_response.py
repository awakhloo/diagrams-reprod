"""Compare one response run (scripts/run_response.py) with its theory (scripts/response_theory.py).
Output: results/response_metrics_{tag}.json. The status (chaotic: all Lyapunov exponents > 0 and decorrelation < 0.5; theory defined:
strip > 0) is recorded; the theory entries are computed whenever the theory exists (strip > 0).
Metrics per field and lag (scs.twopoint.metrics on the (n_col, N) layout, rows = kicked sites): off-diagonal raw and noise-corrected MSE,
floor, diagonal MSE, regression slope, RMS of the theory, mean diagonal (sim, theory) and the DMFT single-site value e^{-tau} (x) /
gain e^{-tau} (phi).
usage: analyze_response.py results/response_{tag}.npz [--theory results/response_theory_{tag}.npz] [--tag TAG]"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scs.twopoint import metrics
from scs.io import load_run

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--theory", default=None)
p.add_argument("--tag", default=None)
a = p.parse_args()

arrays, meta = load_run(a.run)
N, seed, s2 = meta["N"], meta["seed"], meta["sigma2"]
tag = a.tag or meta["tag"]
th = np.load(a.theory or f"results/response_theory_{tag}.npz")
assert int(th["N"]) == N and int(th["seed"]) == seed and abs(float(th["sigma2"]) - s2) < 1e-12
assert np.array_equal(th["S"], arrays["S"]), "kicked sites differ between run and theory"
S, taus = arrays["S"], arrays["taus"]
gain, strip = float(th["gain"]), float(th["strip"])
status = "non-chaotic" if (meta["lyap_min"] <= 0 or meta["decorr"] > 0.5) else ("theory undefined" if strip <= 0 else "ok")
out = {k: meta[k] for k in ("N", "seed", "sigma2", "dt", "B", "n_col", "eps", "kicks_per_N", "tau_win", "dtau", "T_trans", "n_win", "n_kicks",
                            "lyap_mean", "lyap_min", "decorr", "ms_per_step", "t_sim", "backend")}
out.update(strip_width=strip, status=status, gain=gain, recursion_vs_direct=float(np.max(th["recursion_vs_direct"])))
print(f"N={N} seed={seed}: lyapunov min {meta['lyap_min']:.4f}, decorr {meta['decorr']:.4f}, strip {strip:.4f}; status: {status}; "
      f"{meta['n_kicks']} kicks per column, eps {meta['eps']:g}; theory recursion vs direct {out['recursion_vs_direct']:.1e}")
if strip <= 0:
    # The fixed-J theory is undefined: write the draw's record without theory entries and stop.
    json.dump(out, open(f"results/response_metrics_{tag}.json", "w"), indent=1)
    print(f"theory undefined (strip <= 0); wrote results/response_metrics_{tag}.json without theory entries")
    sys.exit(0)

th_x = {float(t): th["x_direct"][i] for i, t in enumerate(th["taus_direct"])}
for f in ("x", "phi"):
    h1, h2 = arrays[f"{f}_h1"], arrays[f"{f}_h2"]
    for tau, R_th in th_x.items():
        it = int(np.argmin(np.abs(taus - tau))); assert abs(taus[it] - tau) < 1e-9, (tau, taus)
        R_th = R_th if f == "x" else gain * R_th
        m = metrics(0.5 * (h1[it] + h2[it]), R_th, h1[it], h2[it], S, N)
        m["dmft_diag"] = np.exp(-tau) * (1.0 if f == "x" else gain)
        for k, v in m.items():
            out[f"{f}_tau{tau:g}_{k}"] = float(v)
        print(f"{f:3s} tau={tau:g}: RMS_off th {np.sqrt(m['signal_off']):.3e} | off: RMSE raw {np.sqrt(m['mse_off_plain']):.3e} floor sd {np.sqrt(m['floor_off']):.2e} "
              f"corr {np.sqrt(max(m['mse_off_corr'], 0)):.3e} rel {np.sqrt(m['mse_off_plain'] / m['signal_off']):.3f} slope {m['slope_off']:.4f} "
              f"| diag: RMSE raw {np.sqrt(m['mse_diag_plain']):.3e} floor sd {np.sqrt(m['floor_diag']):.2e} mean sim {m['mean_diag_emp']:.4f} th {m['mean_diag_th']:.4f} dmft {m['dmft_diag']:.4f}")
json.dump(out, open(f"results/response_metrics_{tag}.json", "w"), indent=1)
print(f"wrote results/response_metrics_{tag}.json")
