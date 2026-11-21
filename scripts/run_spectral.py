"""Fourier-domain two-point test, simulation for one (N, seed): the tanh network with the dense S x S accumulator
(lags 0 .. tau_max step dtau on a random site subset S) plus the scaling-lag accumulator of run_twopoint.py as a cross-check.

Output results/spectral_{tag}.npz (tag = N{N}_seed{seed}_sigma2_{s2}, or --tag) with
  sites (n_S,), taus_dense (n_lag,) = 0, -dtau, ..., -tau_max (signed: the accumulator estimates C_ij(-l dt), i, j in S)
  {x,phi}_dense_h1, _h2 : (n_lag, n_S, n_S) half-batch estimates, float32 (relative precision 1e-7)
  rows_scal, taus_scal, {x,phi}_scal_h1, _h2 : the scaling-lag accumulator of run_twopoint.py (rows x N), a cross-check
  lyap, time_mean, gain_site, and every parameter, the chaos check and timings in the metadata.
The transform to omega and the comparison with the theory is scripts/analyze_spectral_direct.py.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scs.backend import xp, asnumpy, BACKEND
from scs.network import make_J, initial_state, simulate, lyapunov_exponent, trajectory_decorrelation
from scs.twopoint import LagAccumulator, DenseAccumulator
from scs.io import save_run

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, required=True)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--sigma2", type=float, default=9.0)
p.add_argument("--dt", type=float, default=0.05)
p.add_argument("--stepper", default="rk4")
p.add_argument("--B", type=int, default=128)
p.add_argument("--T_trans", type=float, default=500.0)
p.add_argument("--T_traj", type=float, required=True, help="post-transient time per trajectory (T_total = B T_traj)")
p.add_argument("--every", type=float, default=1.0, help="accumulate every this many time units")
p.add_argument("--n_S", type=int, default=256, help="sites in the dense subset S (all ordered pairs inside S are kept)")
p.add_argument("--dtau", type=float, default=0.25, help="dense lag spacing")
p.add_argument("--tau_max", type=float, default=100.0, help="dense lag horizon")
p.add_argument("--taus", type=float, nargs="+", default=[0.0, 2.0, 4.0], help="scaling lags (cross-check)")
p.add_argument("--n_rows_max", type=int, default=2000, help="rows kept for the scaling lags (all if N <= this)")
p.add_argument("--T_lyap", type=float, default=100.0)
p.add_argument("--tag", default=None)
p.add_argument("--out", default=None)
a = p.parse_args()

t_start = time.time()
tag = a.tag or f"N{a.N}_seed{a.seed}_sigma2_{a.sigma2:g}"
J = make_J(a.N, a.sigma2, a.seed)
rng = np.random.default_rng(a.seed + 12345)                       # same stream as run_twopoint.py: identical rows_scal
rows_scal = np.arange(a.N) if a.N <= a.n_rows_max else np.sort(rng.choice(a.N, a.n_rows_max, replace=False))
sites = np.sort(np.random.default_rng(a.seed + 3232).choice(a.N, a.n_S, replace=False))
lag_scal = [int(round(t / a.dt)) for t in a.taus]
lag_dense = [int(round(t / a.dt)) for t in np.arange(0.0, a.tau_max + 1e-9, a.dtau)]
every = int(round(a.every / a.dt))
assert all(abs(t / a.dt - round(t / a.dt)) < 1e-9 for t in a.taus), "lags must be multiples of dt"
assert abs(a.dtau / a.dt - round(a.dtau / a.dt)) < 1e-9 and abs(a.every / a.dt - every) < 1e-9, "dtau and every must be multiples of dt"

x = initial_state(a.N, a.B, seed=a.seed + 777)
x = simulate(x, J, a.dt, int(round(a.T_trans / a.dt)), a.stepper)
lam, x = lyapunov_exponent(x, J, a.dt, a.T_lyap, a.stepper)
decorr = trajectory_decorrelation(x)
print(f"N={a.N} seed={a.seed}: transient done in {time.time() - t_start:.0f}s; lyapunov mean {float(xp.mean(lam)):.4f} min {float(xp.min(lam)):.4f}; decorr {decorr:.4f}", flush=True)
print(f"dense subset: n_S = {a.n_S}, {len(lag_dense)} lags to tau_max = {a.tau_max:g} step {a.dtau:g}; samples every {every} steps; T_total = {a.B * a.T_traj:.3g}", flush=True)

acc_s = LagAccumulator(a.N, a.B, rows_scal, lag_scal, every=every)
acc_d = DenseAccumulator(a.N, a.B, sites, lag_dense, every=every)
mean_acc = xp.zeros((a.N, a.B)); gain_acc = xp.zeros((a.N, a.B))


class Delayed:
    """Feed a hook the step index t - t0 so that it fills its history and samples in step with the dense accumulator
    (both then sample at the same times t = L_dense - 1, + every, ...; the gate between them is exact)."""
    def __init__(self, hook, t0): self.hook, self.t0 = hook, t0
    def update(self, t, x, phi):
        if t >= self.t0:
            self.hook.update(t - self.t0, x, phi)


t0_scal = (acc_d.L - 1) - (acc_s.L - 1)
assert t0_scal >= 0 and (acc_d.L - 1) % every == 0 and (acc_s.L - 1) % every == 0, "lag horizons must be multiples of the sampling interval"


class MeanHook:
    def __init__(self): self.n = 0
    def update(self, t, x, phi):
        if t % every == 0:
            mean_acc.__iadd__(x); gain_acc.__iadd__(1.0 - phi ** 2); self.n += 1


mh = MeanHook()
n_steps = int(round(a.T_traj / a.dt))
t0 = time.time()
x = simulate(x, J, a.dt, n_steps, a.stepper, hooks=[Delayed(acc_s, t0_scal), acc_d, mh])
if BACKEND == "cupy":
    xp.cuda.Device().synchronize()
t_sim = time.time() - t0
print(f"simulation: {n_steps} steps in {t_sim:.0f}s ({1e3 * t_sim / n_steps:.2f} ms/step), {acc_d.n_samples} samples", flush=True)
assert acc_d.n_samples == acc_s.n_samples, (acc_d.n_samples, acc_s.n_samples)

arrays = {"sites": sites, "taus_dense": -np.array(lag_dense) * a.dt, "rows_scal": rows_scal, "taus_scal": -np.array(a.taus),
          "lyap": asnumpy(lam), "time_mean": asnumpy(mean_acc) / max(mh.n, 1), "gain_site": asnumpy(gain_acc) / max(mh.n, 1)}
for f in ("x", "phi"):
    h1, h2 = acc_d.estimates(f)
    arrays[f"{f}_dense_h1"] = h1.astype(np.float32); arrays[f"{f}_dense_h2"] = h2.astype(np.float32)
    es = acc_s.estimates(f)
    arrays[f"{f}_scal_h1"] = np.stack([es[l][0] for l in lag_scal]); arrays[f"{f}_scal_h2"] = np.stack([es[l][1] for l in lag_scal])
    c = 0.5 * (h1 + h2); off = ~np.eye(a.n_S, dtype=bool); sd0 = np.sqrt(0.25 * np.mean((h1[0] - h2[0])[off] ** 2))
    print(f"  {f:3s} dense tau=0: RMS off-diag {np.sqrt(np.mean(c[0][off] ** 2)):.4e}  noise sd {sd0:.2e}  mean diag {np.mean(np.diag(c[0])):.4f}", flush=True)
gain_emp = float(xp.mean(gain_acc)) / max(mh.n, 1)
print(f"empirical mean gain <phi'> = {gain_emp:.5f}  (DMFT {float(np.load(f'results/dmft_sigma2_{a.sigma2:g}.npz')['gain']):.5f})", flush=True)
meta = dict(vars(a), tag=tag, backend=BACKEND, gain_emp=gain_emp, n_samples=acc_d.n_samples, T_total=a.T_traj * a.B, decorr=decorr,
            lyap_mean=float(xp.mean(lam)), lyap_min=float(xp.min(lam)), t_sim=t_sim, ms_per_step=1e3 * t_sim / n_steps,
            max_abs_x=float(xp.max(xp.abs(x))), t_wall=time.time() - t_start)
out = a.out or f"results/spectral_{tag}.npz"
save_run(out, arrays, meta)
print(f"saved {out} ({time.time() - t_start:.0f}s wall)")
