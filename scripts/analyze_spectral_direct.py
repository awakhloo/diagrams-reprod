"""Fourier-domain two-point test: the theory C^x(omega), C^phi(omega) evaluated at omega from J and the DMFT inputs only
(scs.theory.theory_spectrum, no lag transform, no window), against the rectangular-window transform of the simulation's dense lag
functions on the site subset S of one run of scripts/run_spectral.py:
    F^sim_ij(omega) = dtau sum_{|tau_l| <= tau_max} C^sim_ij(tau_l) exp(-i omega tau_l),   C^sim = mean of the two stored halves.
Error: raw RMSE_off = sqrt(mean_{i != j in S} |F^sim_ij(omega) - C^th_ij(omega)|^2). Also recorded: the half-split noise floor and
corrected MSE, the theory RMS, the regression slope and the diagonal RMSE (scs.twopoint.metrics).
Output: results/spectral_direct_metrics_{tag}.json.   usage: analyze_spectral_direct.py results/spectral_N1000_seed1_sigma2_9.npz
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scs.network import make_J
from scs.theory import theory_spectrum, strip_width
from scs.twopoint import metrics, two_sided, windowed_transform
from scs.io import load_run
from scs import dmft

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--tau_maxs", type=float, nargs="+", default=[12.5, 25.0, 50.0, 100.0])
p.add_argument("--omegas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
a = p.parse_args()

t_start = time.time()
arrays, meta = load_run(a.run)
N, seed, sigma2 = meta["N"], meta["seed"], meta["sigma2"]
tag = f"N{N}_seed{seed}_sigma2_{sigma2:g}"
d = np.load(f"results/dmft_sigma2_{sigma2:g}.npz")
sol = dmft.DMFTSolution(sigma2, float(d["dt"]), d["tau"], 2 * np.pi * np.fft.fftfreq(len(d["tau"]), d=float(d["dt"])),
                        d["Cx_tau"], d["Cphi_tau"], float(d["gain"]), None, 0.0, 0)
J = make_J(N, sigma2, seed)
strip_file = f"results/strip_N{N}_seed{seed}_sigma2_{sigma2:g}.json"      # from scripts/strip_only.py; computed here when absent
if os.path.exists(strip_file):
    sf = json.load(open(strip_file))
    assert sf["N"] == N and sf["seed"] == seed and abs(sf["sigma2"] - sigma2) < 1e-12 and abs(sf["gain"] - sol.gain) < 1e-12, strip_file
    width = sf["strip_width"]
else:
    width, _ = strip_width(J, sol.gain)
status = "non-chaotic" if (meta["lyap_min"] <= 0 or meta["decorr"] > 0.5) else ("theory undefined" if width <= 0 else "ok")
sites, taus_neg = arrays["sites"], arrays["taus_dense"]
n_S = len(sites); pos = np.arange(n_S); dtau = float(-taus_neg[1])
taus = np.concatenate([taus_neg[::-1], -taus_neg[1:]])                    # -tau_max .. tau_max
omegas = np.array(a.omegas)
base = {"N": N, "seed": seed, "sigma2": sigma2, "B": meta["B"], "T_traj": meta["T_traj"], "T_total": meta["T_total"], "dt": meta["dt"],
        "n_S": n_S, "dtau": dtau, "tau_max": float(-taus_neg[-1]), "lyap_min": meta["lyap_min"], "decorr": meta["decorr"],
        "strip_width": width, "status": status}
print(f"{tag}: status {status}, strip {width:.4f}", flush=True)
if width <= 0:                                                            # no theory for this J
    json.dump(base, open(f"results/spectral_direct_metrics_{tag}.json", "w"), indent=1)
    print("strip <= 0: theory undefined for this J; wrote the record and stopped", flush=True); sys.exit(0)

# ---- theory at the exact frequencies: rows S, columns S ----
t0 = time.time(); th = {"x": [], "phi": []}
for w in omegas:
    Cw = theory_spectrum(J, sol.gain, float(sol.spectrum("x", np.array([w]))[0]), float(sol.spectrum("phi", np.array([w]))[0]), sigma2, w, rows=sites)
    for f in ("x", "phi"):
        th[f].append(np.asarray(Cw[f])[:, sites])
base["t_theory"] = time.time() - t0
print(f"theory at {len(omegas)} frequencies: {base['t_theory']:.0f}s", flush=True)

# ---- simulation: two-sided lags, rectangular-window transform, metrics ----
for f in ("x", "phi"):
    t1, _ = two_sided(arrays[f"{f}_dense_h1"].astype(float), pos); t2, _ = two_sided(arrays[f"{f}_dense_h2"].astype(float), pos)
    print(f"{f}: tau_max omega | RMS th   RMSE raw  noise sd  RMSE corr | floor/MSE  rel err (raw)  slope | diag RMSE raw", flush=True)
    for tm in a.tau_maxs:
        F1 = windowed_transform(t1, taus, omegas, tm, "rect"); F2 = windowed_transform(t2, taus, omegas, tm, "rect")
        for iw, w in enumerate(omegas):
            key = f"{f}_rect_t{tm:g}_w{w:g}"
            m = metrics(0.5 * (F1[iw] + F2[iw]), th[f][iw], F1[iw], F2[iw], pos, n_S)
            for k, v in m.items():
                base[f"{key}_{k}"] = float(v)
            print(f"  {tm:6g} {w:5g} | {np.sqrt(m['signal_off']):8.3e} {np.sqrt(m['mse_off_plain']):9.3e} {np.sqrt(m['floor_off']):9.2e} "
                  f"{np.sqrt(max(m['mse_off_corr'], 0)):9.3e} | {m['floor_off'] / m['mse_off_plain']:8.3f} {np.sqrt(m['mse_off_plain'] / m['signal_off']):8.3f} "
                  f"{m['slope_off']:6.3f} | {np.sqrt(m['mse_diag_plain']):9.3e}", flush=True)
base["t_wall"] = time.time() - t_start
json.dump(base, open(f"results/spectral_direct_metrics_{tag}.json", "w"), indent=1)
print(f"saved results/spectral_direct_metrics_{tag}.json ({base['t_wall']:.0f}s wall)", flush=True)
