"""Dense cross-correlation functions C_ij(tau) on [-tau_max, tau_max] for one draw: simulation vs theory (the data of Fig 1, left).
Input: results/twopoint_dense_N{N}_seed{seed}_sigma2_{s2}.npz from scripts/run_twopoint.py (dense lags 0 .. -tau_max on the rows
`rows_dense`) and the theory npz / metrics json of scripts/analyze_twopoint.py run with --tag dense_N{N}_seed{seed}_sigma2_{s2}.
Both hold C_ij(-tau_l) for i in the row subset S and all j; for pairs inside S the positive lags follow from
C_ij(tau) = C_ji(-tau) (scs.twopoint.two_sided). Output results/dense_cc_sigma2_{s2}.npz, every key prefixed N{N}_:
  taus (-tau_max .. tau_max), rows, pairs (a <= b, positions in rows), and for each field the two half estimates and the theory,
  shape (n_pairs, n_tau), float32; the pooled noise sd per lag; the RMS statistics over all n_rows x N off-diagonal entries at
  tau <= 0 (theory RMS, raw RMSE, floor, corrected RMSE); meta (JSON string).
usage: dense_cc.py N [seed=1] [sigma2=9]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scs.io import load_run
from scs.twopoint import two_sided, noise_floor, offdiag_mask

N = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1; s2 = float(sys.argv[3]) if len(sys.argv) > 3 else 9.0
tag = f"N{N}_seed{seed}_sigma2_{s2:g}"
run_tag = "dense_" + tag          # the run, theory and metrics files of this experiment carry the prefix dense_
arrays, meta = load_run(f"results/twopoint_{run_tag}.npz")
th = np.load(f"results/twopoint_theory_{run_tag}.npz")
m = json.load(open(f"results/twopoint_metrics_{run_tag}.json"))
rows = arrays["rows_dense"]; taus_neg = arrays["taus_dense"]            # 0, -dtau, ..., -tau_max
n_rows, n_lag = len(rows), len(taus_neg)
taus = np.concatenate([taus_neg[::-1], -taus_neg[1:]])                  # -tau_max .. tau_max
a_idx, b_idx = np.triu_indices(n_rows)                                  # unordered pairs a <= b (a == b: diagonal)
off = offdiag_mask(rows, N)
out = {"taus": taus, "rows": rows, "pairs": np.stack([a_idx, b_idx], 1)}
print(f"{tag}: {n_rows} rows, {n_lag} lags to tau_max = {-taus_neg[-1]:g}, {len(a_idx)} pairs; T_total = {meta['T_total']:.3g}, dw = {m['dw']:.5f}, w_max = {m['w_max']:g}")
for f in ("x", "phi"):
    h1, h2 = arrays[f"{f}_dense_h1"], arrays[f"{f}_dense_h2"]; t = th[f"{f}_dense"]
    assert h1.shape == t.shape == (n_lag, n_rows, N), (h1.shape, t.shape)
    emp = 0.5 * (h1 + h2)
    # per-lag statistics over all n_rows x N off-diagonal entries (tau <= 0)
    rms_th = np.sqrt([np.mean(t[l][off] ** 2) for l in range(n_lag)])
    mse_raw = np.array([np.mean((emp[l] - t[l])[off] ** 2) for l in range(n_lag)])
    floor = np.array([noise_floor(h1[l], h2[l], off) for l in range(n_lag)])
    sd = np.sqrt(floor)                                                 # pooled noise sd of one entry at this |tau|
    out.update({f"{f}_rms_th": rms_th, f"{f}_rmse_raw": np.sqrt(mse_raw), f"{f}_floor": floor,
                f"{f}_rmse_corr": np.sqrt(np.maximum(mse_raw - floor, 0.0)), f"{f}_sd": np.concatenate([sd[::-1], sd[1:]])})
    for name, arr in (("h1", h1), ("h2", h2), ("th", t)):
        two, asym = two_sided(arr, rows)
        out[f"{f}_{name}"] = two[a_idx, b_idx].astype(np.float32)
        print(f"  {f:3s} {name}: tau = 0 mirror asymmetry {asym:.2e}" + ("  (theory: Hermiticity of the transform)" if name == "th" else "  (round-off of the accumulator)"))
    for l in [0, int(round(10 / abs(taus_neg[1]))), int(round(50 / abs(taus_neg[1]))), n_lag - 1]:
        print(f"  {f:3s} tau = {taus_neg[l]:6.1f}: RMS theory {rms_th[l]:.4e}  raw RMSE {np.sqrt(mse_raw[l]):.4e}  noise sd {sd[l]:.4e}  corrected RMSE {out[f'{f}_rmse_corr'][l]:.4e}")
out["meta"] = json.dumps({"N": N, "seed": seed, "sigma2": s2, "T_total": meta["T_total"], "B": meta["B"], "dt": meta["dt"],
                          "dw": m["dw"], "w_max": m["w_max"], "strip_width": m["strip_width"]})
np.savez(f"results/dense_cc_sigma2_{s2:g}.npz", **{f"N{N}_{k}": v for k, v in out.items()}, readme=np.array(
    "Dense cross-correlations C_ij(tau), sigma^2 = %g, seed 1. Per N: taus (signed lags), rows (the row subset S), pairs (a <= b as positions in rows; "
    "a == b is a diagonal entry), {x,phi}_{h1,h2} the two half-batch simulation estimates and {x,phi}_th the theory, all (n_pairs, n_tau) float32 "
    "(the estimate is 0.5 (h1 + h2), its noise variance per entry 0.25 <(h1 - h2)^2>); {x,phi}_sd the pooled noise sd per lag; {x,phi}_{rms_th,rmse_raw,floor,rmse_corr} "
    "over all rows x N off-diagonal entries at the lags 0, -dtau, ..., -tau_max. Convention C_ij(tau) = <f_i(t + tau) f_j(t)>." % s2))
print("saved", f"results/dense_cc_sigma2_{s2:g}.npz")
