"""Compare one two-point run (scripts/run_twopoint.py) with the fixed-J theory for the same J.
Output: results/twopoint_metrics_{tag}.json (per field and lag: off-diagonal and diagonal MSE, raw and noise-corrected, noise floor,
regression slope, RMS of the theory) and results/twopoint_theory_{tag}.npz (the theory at the scaling and dense lags).
usage: analyze_twopoint.py results/twopoint_N1000_seed1_sigma2_9.npz --dw_auto"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scs.network import make_J
from scs.theory import theory_lags, strip_width
from scs.twopoint import metrics
from scs.io import load_run
from scs import dmft

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--dw", type=float, default=0.01)
p.add_argument("--dw_auto", action="store_true", help="use dw = min(dw, 0.4 * strip) once the strip is known (transform error <= 1.5e-7)")
p.add_argument("--w_max", type=float, default=8.0, help="omega cutoff of the theory transform")
p.add_argument("--dmft", default=None, help="DMFT npz; default results/dmft_sigma2_{sigma2}.npz of the run")
p.add_argument("--no_eigs", action="store_true")
p.add_argument("--strip_file", default=None, help="json from scripts/strip_only.py; default results/strip_N{N}_seed{seed}_sigma2_{s2}.json when it exists (skips the eigendecomposition)")
p.add_argument("--tag", default="")
a = p.parse_args()

arrays, meta = load_run(a.run)
N, seed, sigma2 = meta["N"], meta["seed"], meta["sigma2"]
d = np.load(a.dmft or f"results/dmft_sigma2_{sigma2:g}.npz")
assert abs(float(d["sigma2"]) - sigma2) < 1e-12
sol = dmft.DMFTSolution(sigma2, float(d["dt"]), d["tau"], 2 * np.pi * np.fft.fftfreq(len(d["tau"]), d=float(d["dt"])),
                        d["Cx_tau"], d["Cphi_tau"], float(d["gain"]), None, 0.0, 0)
J = make_J(N, sigma2, seed)
Cx = lambda w: sol.spectrum("x", w); Cp = lambda w: sol.spectrum("phi", w)
rows_s, rows_d = arrays["rows_scal"], arrays["rows_dense"]
taus_s, taus_d = arrays["taus_scal"], arrays["taus_dense"]
width = None
strip_file = a.strip_file or f"results/strip_N{N}_seed{seed}_sigma2_{sigma2:g}.json"
if not a.no_eigs:
    if os.path.exists(strip_file):
        sf = json.load(open(strip_file))
        assert sf["N"] == N and sf["seed"] == seed and abs(sf["sigma2"] - sigma2) < 1e-12 and abs(sf["gain"] - sol.gain) < 1e-12, strip_file
        width, max_abs_lam = sf["strip_width"], sf["max_abs_lam"]
        print(f"strip width {width:.4f} (1 - sigma*gain = {1 - np.sqrt(sigma2) * sol.gain:.4f}); max|lam| {max_abs_lam:.4f}  [from {strip_file}]", flush=True)
    else:
        width, lam = strip_width(J, sol.gain)
        print(f"strip width {width:.4f} (1 - sigma*gain = {1 - np.sqrt(sigma2) * sol.gain:.4f}); max|lam| {abs(lam).max():.4f}", flush=True)
    if width <= 0.005:
        print("WARNING: strip width <= 0.005; the fixed-J theory is unstable or hyper-resonant for this J", flush=True)
    if width <= 0:
        # The fixed-J theory is undefined: write the draw's record without theory entries and stop.
        tag = a.tag or f"N{N}_seed{seed}_sigma2_{sigma2:g}"
        out = {"N": N, "seed": seed, "sigma2": sigma2, "B": meta["B"], "T_traj": meta["T_traj"], "T_total": meta["T_total"],
               "lyap_mean": meta["lyap_mean"], "lyap_min": meta["lyap_min"], "decorr": meta["decorr"], "strip_width": width,
               "status": "theory undefined"}
        json.dump(out, open(f"results/twopoint_metrics_{tag}.json", "w"), indent=1)
        print(f"strip <= 0: theory undefined for this J; wrote results/twopoint_metrics_{tag}.json and stopped", flush=True)
        sys.exit(0)
    if a.dw_auto:
        a.dw = min(a.dw, 0.4 * width)
print(f"omega grid: dw = {a.dw:.5f}, w_max = {a.w_max:g}", flush=True)
t0 = time.time()
th_s = theory_lags(J, sol.gain, Cx, Cp, sigma2, rows_s, taus_s, a.dw, a.w_max, strip=width)
th_d = theory_lags(J, sol.gain, Cx, Cp, sigma2, rows_d, taus_d, a.dw, a.w_max, strip=width)
print(f"theory: {time.time()-t0:.0f}s", flush=True)

out = {"N": N, "seed": seed, "sigma2": sigma2, "B": meta["B"], "T_traj": meta["T_traj"], "T_total": meta["T_total"], "dt": meta["dt"], "T_trans": meta["T_trans"],
       "lyap_mean": meta["lyap_mean"], "lyap_min": meta["lyap_min"], "decorr": meta["decorr"], "gain_emp": meta.get("gain_emp"), "ms_per_step": meta.get("ms_per_step"),
       "strip_width": width, "dw": a.dw, "w_max": a.w_max, "n_rows": int(len(rows_s))}
for f in ("x", "phi"):
    h1, h2 = arrays[f"{f}_scal_h1"], arrays[f"{f}_scal_h2"]
    for it, tau in enumerate(taus_s):
        m = metrics(0.5 * (h1[it] + h2[it]), th_s[f][it], h1[it], h2[it], rows_s, N)
        for k, v in m.items():
            out[f"{f}_tau{abs(tau):g}_{k}"] = float(v)
        print(f"{f:3s} tau={abs(tau):5g}: RMS_off theory {np.sqrt(m['signal_off']):.3e} | MSE_off plain {m['mse_off_plain']:.3e} floor {m['floor_off']:.3e} corr {m['mse_off_corr']:.3e} "
              f"| MSE_diag plain {m['mse_diag_plain']:.3e} floor {m['floor_diag']:.3e} corr {m['mse_diag_corr']:.3e} | slope {m['slope_off']:.4f} | mean diag emp {m['mean_diag_emp']:.4f} th {m['mean_diag_th']:.4f}")
tag = a.tag or f"N{N}_seed{seed}_sigma2_{sigma2:g}"
json.dump(out, open(f"results/twopoint_metrics_{tag}.json", "w"), indent=1)
np.savez(f"results/twopoint_theory_{tag}.npz", **{f"{f}_scal": th_s[f] for f in ("x", "phi")}, **{f"{f}_dense": th_d[f] for f in ("x", "phi")})
