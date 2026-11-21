"""Dense response functions R_ij(tau) on (0, tau_win] for one draw (the data of Fig 1, right): simulation (perturbed copies,
scripts/run_response.py --sites) vs theory (scripts/response_theory.py --sites), for the kicked sites j = S and the read-out rows
i = the dense rows of the C run (results/dense_cc_sigma2_{s2}.npz).
Input (all with the prefix dense_): results/response_dense_{tag}.npz, results/response_theory_dense_{tag}.npz,
results/response_metrics_dense_{tag}.json (scripts/analyze_response.py). Output
  results/dense_rc_{tag}.npz : taus (dtau .. tau_win; tau = 0 is not stored: R_ij(0) = delta_ij exactly), sites S, rows, and for
      each field the two half estimates and the theory, shape (n_tau, n_sites, n_rows), float32, [k, a, b] = R_{rows[b], S[a]}(taus[k]);
      per lag over all n_sites x N off-diagonal entries (float64): RMS theory, raw RMSE, noise floor, corrected RMSE, noise sd, relative
      error raw RMSE / RMS theory; the run's metadata as a JSON string.
usage: dense_rc.py N [seed=1] [sigma2=9]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scs.io import load_run
from scs.twopoint import noise_floor, offdiag_mask

N = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1; s2 = float(sys.argv[3]) if len(sys.argv) > 3 else 9.0
tag = f"N{N}_seed{seed}_sigma2_{s2:g}"
run_tag = "dense_" + tag
arrays, meta = load_run(f"results/response_{run_tag}.npz")
th = np.load(f"results/response_theory_{run_tag}.npz")
m = json.load(open(f"results/response_metrics_{run_tag}.json"))
cc = np.load(f"results/dense_cc_sigma2_{s2:g}.npz")
rows = cc[f"N{N}_rows"]                                                  # the dense rows of the C run of the same draw
S, taus = arrays["S"], arrays["taus"]
gain = float(th["gain"])
assert int(th["N"]) == N and int(th["seed"]) == seed and np.array_equal(th["S"], S), "theory and run must be the same draw and sites"
assert np.allclose(th["taus_dense"], taus) and th["x_dense"].shape == (len(taus), len(S), N), (th["x_dense"].shape, taus)
assert m["status"] == "ok", f"draw not admissible: {m['status']}"
assert all(j in rows for j in S), "every kicked site must be one of the dense rows of the C run (the plotted pairs live inside them)"
n_tau, n_sites, n_rows = len(taus), len(S), len(rows)
off = offdiag_mask(S, N)
print(f"{tag}: {n_sites} kicked sites {S.tolist()}, {n_rows} rows, {n_tau} lags to tau_win = {taus[-1]:g}; {meta['n_kicks']} kicks per site, "
      f"eps = {meta['eps']:g}, B = {meta['B']}; theory recursion vs direct {m['recursion_vs_direct']:.1e}, strip {float(th['strip']):.4f}")
out = {"taus": taus, "sites": S, "rows": rows}
for f in ("x", "phi"):
    h1, h2 = arrays[f"{f}_h1"], arrays[f"{f}_h2"]
    t = th["x_dense"] if f == "x" else gain * th["x_dense"]
    assert h1.shape == h2.shape == t.shape == (n_tau, n_sites, N)
    emp = 0.5 * (h1 + h2)
    rms_th = np.sqrt([np.mean(t[k][off] ** 2) for k in range(n_tau)])
    mse_raw = np.array([np.mean((emp[k] - t[k])[off] ** 2) for k in range(n_tau)])
    floor = np.array([noise_floor(h1[k], h2[k], off) for k in range(n_tau)])
    out.update({f"{f}_rms_th": rms_th, f"{f}_rmse_raw": np.sqrt(mse_raw), f"{f}_floor": floor, f"{f}_sd": np.sqrt(floor),
                f"{f}_rmse_corr": np.sqrt(np.maximum(mse_raw - floor, 0.0)), f"{f}_rel": np.sqrt(mse_raw) / rms_th})
    for name, arr in (("h1", h1), ("h2", h2), ("th", t)):
        out[f"{f}_{name}"] = arr[:, :, rows].astype(np.float32)          # [k, a, b] = R_{rows[b], S[a]}(taus[k])
    print(f"  {f:3s}   tau   RMS theory   raw RMSE   noise sd   corrected   rel (raw/th)   share (floor/raw MSE)")
    for tau in (0.5, 1, 2, 4, 6, 8, 10):
        if tau > taus[-1] + 1e-9:
            continue
        k = int(np.argmin(np.abs(taus - tau))); assert abs(taus[k] - tau) < 1e-9
        print(f"  {f:3s} {taus[k]:5.2f}   {rms_th[k]:.4e}   {np.sqrt(mse_raw[k]):.4e}   {np.sqrt(floor[k]):.2e}   {out[f'{f}_rmse_corr'][k]:.4e}   "
              f"{out[f'{f}_rel'][k]:.4f}   {floor[k] / mse_raw[k]:.3f}")
np.savez(f"results/dense_rc_{tag}.npz", **out,
         meta=json.dumps({"N": N, "seed": seed, "sigma2": s2, "sites": S.tolist(), "n_kicks": meta["n_kicks"], "kicks_per_half": meta["kicks_per_half"],
                          "eps": meta["eps"], "B": meta["B"], "tau_win": meta["tau_win"], "dtau": meta["dtau"], "dt": meta["dt"], "T_trans": meta["T_trans"],
                          "gain": gain, "gain_emp": meta["gain_emp"], "strip_width": float(th["strip"]), "lyap_mean": meta["lyap_mean"], "lyap_min": meta["lyap_min"],
                          "decorr": meta["decorr"], "ms_per_step": meta["ms_per_step"], "t_sim": meta["t_sim"], "backend": meta["backend"]}))
print("saved", f"results/dense_rc_{tag}.npz")
