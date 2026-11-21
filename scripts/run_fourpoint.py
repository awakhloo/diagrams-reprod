"""Four-point test (eq:simplemoment, m = 4) for one (N, seed): simulate the tanh network, accumulate the pair-product
Gram matrices on a site subset S per trajectory group, compare with the Wick prediction from (a) the run's own
two-point functions and (b) the fixed-J theory, and write the reduced metrics.

Outputs (tag = N{N}_seed{seed}_sigma2_{sigma2}, prefix results/fourpoint):
  results/fourpoint_metrics_{tag}.json   all metrics per (field, lag) and Wick version, chaos check, timings, parameters
  results/fourpoint_data_{tag}.npz       sites, per-group two-point functions on S (both lags), theory two-point on S,
                                         a random sample of masked entries (indices, per-group m and W) for figures
Conventions: the accumulator estimates m^l_{ab,cd} = <f_a(t-l) f_b(t-l) f_c(t) f_d(t)> and D_l[i,j] = <f_i(t-l) f_j(t)>;
the theory is evaluated at the signed lags (0, -lag) so that D_l^th[i, j] = C^th_ij(-lag) (scs/fourpoint.py).
Everything large stays on the device; the reduction runs here so that no P x P matrices are written to disk.
"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scs.backend import xp, asnumpy, BACKEND
from scs.network import make_J, initial_state, simulate, lyapunov_exponent, trajectory_decorrelation
from scs.fourpoint import PairProductAccumulator, reduce_block, pairing_consistency, two_point_offdiag
from scs.theory import theory_lags, strip_width
from scs import dmft

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, required=True)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--sigma2", type=float, default=9.0)
p.add_argument("--dt", type=float, default=0.05)
p.add_argument("--stepper", default="rk4")
p.add_argument("--B", type=int, default=128, help="trajectories (multiple of K)")
p.add_argument("--K", type=int, default=8, help="trajectory groups for the noise floor")
p.add_argument("--n_S", type=int, default=128, help="sites in the subset S (all pairs inside S are used)")
p.add_argument("--T_trans", type=float, default=500.0)
p.add_argument("--T_traj", type=float, required=True, help="post-transient time per trajectory")
p.add_argument("--T_lyap", type=float, default=100.0)
p.add_argument("--every", type=float, default=1.0, help="sample every this many time units")
p.add_argument("--lag", type=float, default=2.0, help="the lagged configuration (t-lag, t-lag, t, t); tau_c/2")
p.add_argument("--no_theory", action="store_true")
p.add_argument("--strip_file", default=None, help="json from scripts/strip_only.py; default results/strip_N{N}_seed{seed}_sigma2_{s2}.json")
p.add_argument("--strip_wait", type=float, default=0.0, help="seconds to wait for the strip file (a background job) before computing it here")
p.add_argument("--dw", type=float, default=0.01)
p.add_argument("--dw_auto", action="store_true", help="dw = min(dw, 0.4 strip)")
p.add_argument("--w_max", type=float, default=8.0, help="omega cutoff of the theory transform")
p.add_argument("--sample", type=int, default=10000, help="masked entries sampled per block for figures (~13 MB per run at 10000)")
p.add_argument("--chunk", type=int, default=256, help="pair rows per reduction chunk")
p.add_argument("--tag", default=None)
p.add_argument("--prefix", default="results/fourpoint")
a = p.parse_args()

t_start = time.time()
tag = a.tag or f"N{a.N}_seed{a.seed}_sigma2_{a.sigma2:g}"
assert a.B % a.K == 0 and a.K >= 3, "B must be a multiple of K and K >= 3 (jackknife)"
lag_steps = int(round(a.lag / a.dt)); every = int(round(a.every / a.dt))
assert abs(a.lag / a.dt - lag_steps) < 1e-9 and abs(a.every / a.dt - every) < 1e-9, "lag and every must be multiples of dt"
J = make_J(a.N, a.sigma2, a.seed)
sites = np.sort(np.random.default_rng(a.seed + 2024).choice(a.N, a.n_S, replace=False))
d = np.load(f"results/dmft_sigma2_{a.sigma2:g}.npz")
sol = dmft.DMFTSolution(a.sigma2, float(d["dt"]), d["tau"], 2 * np.pi * np.fft.fftfreq(len(d["tau"]), d=float(d["dt"])),
                        d["Cx_tau"], d["Cphi_tau"], float(d["gain"]), None, 0.0, 0)
var0 = {"x": float(d["Cx_tau"][0]), "phi": float(d["Cphi_tau"][0])}   # DMFT variances, for the normalized numbers

# ---- simulation ----
x = initial_state(a.N, a.B, seed=a.seed + 777)
x = simulate(x, J, a.dt, int(round(a.T_trans / a.dt)), a.stepper)
lam, x = lyapunov_exponent(x, J, a.dt, a.T_lyap, a.stepper)
decorr = trajectory_decorrelation(x)
lyap_mean, lyap_min = float(xp.mean(lam)), float(xp.min(lam))
print(f"N={a.N} seed={a.seed}: transient done in {time.time()-t_start:.0f}s; lyapunov mean {lyap_mean:.4f} min {lyap_min:.4f}; decorr {decorr:.4f}", flush=True)
acc = PairProductAccumulator(a.N, a.B, sites, a.K, [0, lag_steps], every=every)
mean_acc = xp.zeros((a.N, a.B)); gain_acc = xp.zeros((a.N, a.B))


class MeanHook:
    def __init__(self): self.n = 0
    def update(self, t, x, phi):
        if t % every == 0:
            mean_acc.__iadd__(x); gain_acc.__iadd__(1.0 - phi ** 2); self.n += 1


mh = MeanHook()
n_steps = int(round(a.T_traj / a.dt))
t0 = time.time()
x = simulate(x, J, a.dt, n_steps, a.stepper, hooks=[acc, mh])
if BACKEND == "cupy":
    xp.cuda.Stream.null.synchronize()
t_sim = time.time() - t0
T_total = a.T_traj * a.B
print(f"simulation: {n_steps} steps in {t_sim:.0f}s ({1e3 * t_sim / n_steps:.2f} ms/step), {acc.n_samples} samples, T_total = {T_total:.3g}", flush=True)
gain_emp = float(xp.mean(gain_acc)) / max(mh.n, 1)
time_mean = asnumpy(mean_acc) / max(mh.n, 1)
mean_sq_ratio = float(np.mean(time_mean ** 2) / var0["x"])
print(f"empirical mean gain {gain_emp:.5f} (DMFT {sol.gain:.5f}); mean_i m_i^2 / var = {mean_sq_ratio:.4f}", flush=True)

# ---- strip and theory ----
width, strip_secs = None, None
strip_file = a.strip_file or f"results/strip_N{a.N}_seed{a.seed}_sigma2_{a.sigma2:g}.json"
t0 = time.time()
while not os.path.exists(strip_file) and time.time() - t0 < a.strip_wait:
    time.sleep(10)
if os.path.exists(strip_file):
    sf = json.load(open(strip_file))
    assert sf["N"] == a.N and sf["seed"] == a.seed and abs(sf["sigma2"] - a.sigma2) < 1e-12 and abs(sf["gain"] - sol.gain) < 1e-12, strip_file
    width = sf["strip_width"]; strip_src = strip_file
else:
    t0 = time.time(); width, _ = strip_width(J, sol.gain); strip_secs = time.time() - t0; strip_src = f"computed here ({strip_secs:.0f}s)"
status = "non-chaotic" if (lyap_min <= 0 or decorr > 0.5) else ("theory undefined" if width <= 0 else "ok")
print(f"strip width {width:.4f} (1 - sigma*gain = {1 - np.sqrt(a.sigma2) * sol.gain:.4f}) [{strip_src}]; status: {status}", flush=True)
th = None
if not a.no_theory and width > 0:
    if a.dw_auto:
        a.dw = min(a.dw, 0.4 * width)
    Cx = lambda w: sol.spectrum("x", w); Cp = lambda w: sol.spectrum("phi", w)
    t0 = time.time()
    th_full = theory_lags(J, sol.gain, Cx, Cp, a.sigma2, sites, [0.0, -a.lag], a.dw, a.w_max, strip=width)
    t_th = time.time() - t0
    th = {f: (th_full[f][0][:, sites], th_full[f][1][:, sites]) for f in ("x", "phi")}   # (D0_th, Dl_th) on S x S
    print(f"theory on S (dw = {a.dw:.5f}, w_max = {a.w_max:g}): {t_th:.0f}s", flush=True)

# ---- reduction ----
out = {"N": a.N, "seed": a.seed, "sigma2": a.sigma2, "dt": a.dt, "stepper": a.stepper, "B": a.B, "K": a.K, "n_S": a.n_S, "P": acc.P,
       "T_trans": a.T_trans, "T_traj": a.T_traj, "T_total": T_total, "every": a.every, "lag": a.lag, "n_samples": acc.n_samples,
       "lyap_mean": lyap_mean, "lyap_min": lyap_min, "decorr": decorr, "gain_emp": gain_emp, "mean_sq_ratio": mean_sq_ratio,
       "strip_width": width, "status": status, "dw": a.dw if th else None, "w_max": a.w_max if th else None, "backend": BACKEND,
       "ms_per_step": 1e3 * t_sim / n_steps, "t_sim": t_sim, "t_theory": t_th if th else None, "var0_x": var0["x"], "var0_phi": var0["phi"]}
data = {"sites": sites, "lag_steps": np.array([0, lag_steps]), "lags_time": np.array([0.0, a.lag])}
t0 = time.time()
for f in ("x", "phi"):
    D0 = acc.two_point(f, 0)
    for l in (0, lag_steps):
        Dl = acc.two_point(f, l)
        data[f"{f}_two_lag{l}"] = asnumpy(Dl)                     # (K, n_S, n_S)
        D0_th, Dl_th = (None, None) if th is None else (th[f][0], th[f][1] if l else th[f][0])
        r = reduce_block(acc.gram[f][l], acc.norm, D0, Dl, acc.ia_d, acc.ib_d, D0_th=D0_th, Dl_th=Dl_th,
                         chunk=a.chunk, sample=a.sample, seed=a.seed)
        key = f"{f}_lag{l}"
        for k, v in r.items():
            if k.startswith("sample_"):
                for kk, vv in v.items():
                    data[f"{key}_{k}_{kk}"] = vv
            else:
                out[f"{key}_{k}"] = v
        if l == 0:
            out[f"{key}_pairing_consistency"] = pairing_consistency(acc.gram[f][0], a.n_S)
        tp = two_point_offdiag(Dl, None if th is None else Dl_th)
        for k, v in tp.items():
            out[f"{key}_two_{k}"] = v
        if th is not None:
            data[f"{f}_two_th_lag{l}"] = np.asarray(Dl_th)
        v2 = out[f"{f}_lag0_two_mean_diag"] ** 2                    # empirical variance^2 on S: the normalization
        line = (f"{f:3s} lag {l * a.dt:3g}: n {r['n']:.3g} | RMS Wick {r['rms_wick_emp']:.3e} | emp: RMSE raw {np.sqrt(r['mse_raw_emp']):.3e} "
                f"floor sd {np.sqrt(r['floor_emp']):.3e} corr {np.sqrt(max(r['mse_corr_emp'], 0)):.3e} (MSE {r['mse_corr_emp']:.3e} +- {r['mse_corr_emp_se']:.1e}) slope {r['slope_emp']:.4f}")
        if th is not None:
            line += f" | th: RMSE raw {np.sqrt(r['mse_raw_th']):.3e} corr {np.sqrt(max(r['mse_corr_th'], 0)):.3e} (MSE {r['mse_corr_th']:.3e} +- {r['mse_corr_th_se']:.1e}) slope {r['slope_th']:.4f}"
        line += (f" | normalized: Wick {r['rms_wick_emp'] / v2:.3e} corr {np.sqrt(max(r['mse_corr_emp'], 0)) / v2:.3e} noise sd^2 x T {r['floor_emp'] * T_total / v2 ** 2:.2f}"
                 f" (idiosyncratic {r['floor_idio_emp'] * T_total / v2 ** 2:.2f}; common-mode sd(eps_k) {np.std(r['eps_emp'], ddof=1):.4f})")
        print(line, flush=True)
        if l == 0:
            print(f"    pairing consistency (max rel. diff of the three pairings and the transpose): {out[f'{key}_pairing_consistency']:.1e}", flush=True)
        tl = f"    two-point on S x S, lag {l * a.dt:g}: RMS {tp['rms']:.3e} noise sd {np.sqrt(tp['floor']):.2e} mean diag {tp['mean_diag']:.4f}"
        if th is not None:
            tl += f" | vs theory: RMS th {tp['rms_th']:.3e} RMSE raw {np.sqrt(tp['mse_raw']):.3e} corr {np.sqrt(max(tp['mse_corr'], 0)):.3e} slope {tp['slope']:.4f} mean diag th {tp['mean_diag_th']:.4f}"
        print(tl, flush=True)
out["t_reduce"] = time.time() - t0
out["t_wall"] = time.time() - t_start
print(f"reduction {out['t_reduce']:.0f}s; total wall {out['t_wall']:.0f}s", flush=True)
json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}, open(f"{a.prefix}_metrics_{tag}.json", "w"), indent=1)
np.savez(f"{a.prefix}_data_{tag}.npz", **data)
print("saved", f"{a.prefix}_metrics_{tag}.json", f"{a.prefix}_data_{tag}.npz")
