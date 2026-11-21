"""Simulate the tanh network for one (N, seed) and accumulate two-point statistics.

Outputs results/twopoint_N{N}_seed{seed}_sigma2_{sigma2}.npz with, for fields x and phi:
  scaling lags:        C_half1, C_half2 at lags tau in {0, -tau1, -tau2} for rows `rows_scal` (all rows if N <= n_rows_max)
  dense lags:          C at tau = 0, -dtau, ..., -tau_max for a small row subset `rows_dense`
plus the chaos check and all parameters. The accumulator estimates <x_i(t - l dt) x_j(t)> = C_ij(-l dt).
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scs.backend import xp, asnumpy, BACKEND
from scs.network import make_J, initial_state, simulate, lyapunov_exponent, trajectory_decorrelation
from scs.twopoint import LagAccumulator
from scs.io import save_run

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, required=True)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--sigma2", type=float, default=9.0)
p.add_argument("--dt", type=float, default=0.05)
p.add_argument("--stepper", default="rk4")
p.add_argument("--B", type=int, default=32)
p.add_argument("--T_trans", type=float, default=500.0)
p.add_argument("--T_traj", type=float, required=True, help="post-transient time per trajectory")
p.add_argument("--every", type=float, default=1.0, help="accumulate every this many time units")
p.add_argument("--taus", type=float, nargs="+", default=[0.0, 2.0, 4.0], help="scaling lags (positive; stored as C(-tau)); 0, tau_c/2, tau_c (tau_c = 4.5 at sigma^2 = 9)")
p.add_argument("--n_rows_max", type=int, default=2000, help="rows kept for the scaling lags (all if N <= this)")
p.add_argument("--n_rows_dense", type=int, default=16)
p.add_argument("--dense_dtau", type=float, default=0.5)
p.add_argument("--dense_tau_max", type=float, default=40.0)   # ~9 tau_c at sigma^2 = 9
p.add_argument("--T_lyap", type=float, default=100.0)
p.add_argument("--out", default=None)
a = p.parse_args()

t_start = time.time()
J = make_J(a.N, a.sigma2, a.seed)
rng = np.random.default_rng(a.seed + 12345)
rows_scal = np.arange(a.N) if a.N <= a.n_rows_max else np.sort(rng.choice(a.N, a.n_rows_max, replace=False))
rows_dense = np.sort(rng.choice(a.N, a.n_rows_dense, replace=False))
lag_scal = [int(round(t / a.dt)) for t in a.taus]
lag_dense = [int(round(t / a.dt)) for t in np.arange(0.0, a.dense_tau_max + 1e-9, a.dense_dtau)]
every = int(round(a.every / a.dt))
assert all(abs(t / a.dt - round(t / a.dt)) < 1e-9 for t in a.taus), "lags must be multiples of dt"

x = initial_state(a.N, a.B, seed=a.seed + 777)
x = simulate(x, J, a.dt, int(round(a.T_trans / a.dt)), a.stepper)
lam, x = lyapunov_exponent(x, J, a.dt, a.T_lyap, a.stepper)
decorr = trajectory_decorrelation(x)
t_setup = time.time() - t_start
print(f"N={a.N} seed={a.seed}: transient done in {t_setup:.0f}s; lyapunov mean {float(xp.mean(lam)):.4f} min {float(xp.min(lam)):.4f}; decorr {decorr:.4f}", flush=True)

acc_s = LagAccumulator(a.N, a.B, rows_scal, lag_scal, every=every)
acc_d = LagAccumulator(a.N, a.B, rows_dense, lag_dense, every=every)
mean_acc = xp.zeros((a.N, a.B))  # time means per trajectory
gain_acc = xp.zeros((a.N, a.B))  # time means of phi'(x) = 1 - tanh^2 per site and trajectory


class MeanHook:
    def __init__(self): self.n = 0
    def update(self, t, x, phi):
        if t % every == 0:
            mean_acc.__iadd__(x); gain_acc.__iadd__(1.0 - phi ** 2); self.n += 1


mh = MeanHook()
n_steps = int(round(a.T_traj / a.dt))
t0 = time.time()
x = simulate(x, J, a.dt, n_steps, a.stepper, hooks=[acc_s, acc_d, mh])
t_sim = time.time() - t0
print(f"simulation: {n_steps} steps in {t_sim:.0f}s ({1e3 * t_sim / n_steps:.2f} ms/step), {acc_s.n_samples} samples", flush=True)

arrays = {"rows_scal": rows_scal, "rows_dense": rows_dense, "taus_scal": -np.array(a.taus),
          "taus_dense": -np.array(lag_dense) * a.dt, "lyap": asnumpy(lam),
          "time_mean": asnumpy(mean_acc) / max(mh.n, 1), "gain_site": asnumpy(gain_acc) / max(mh.n, 1)}
for f in ("x", "phi"):
    es = acc_s.estimates(f)
    arrays[f"{f}_scal_h1"] = np.stack([es[l][0] for l in lag_scal])
    arrays[f"{f}_scal_h2"] = np.stack([es[l][1] for l in lag_scal])
    ed = acc_d.estimates(f)
    arrays[f"{f}_dense_h1"] = np.stack([ed[l][0] for l in lag_dense])
    arrays[f"{f}_dense_h2"] = np.stack([ed[l][1] for l in lag_dense])
gain_emp = float(xp.mean(gain_acc)) / max(mh.n, 1)
print(f"empirical mean gain <phi'> = {gain_emp:.5f}  (DMFT {float(np.load(f'results/dmft_sigma2_{a.sigma2:g}.npz')['gain']):.5f})", flush=True)
meta = dict(vars(a), backend=BACKEND, gain_emp=gain_emp, n_samples=acc_s.n_samples, T_total=a.T_traj * a.B, decorr=decorr,
            lyap_mean=float(xp.mean(lam)), lyap_min=float(xp.min(lam)), t_sim=t_sim, ms_per_step=1e3 * t_sim / n_steps,
            max_abs_x=float(xp.max(xp.abs(x))))
out = a.out or f"results/twopoint_N{a.N}_seed{a.seed}_sigma2_{a.sigma2:g}.npz"
save_run(out, arrays, meta)
print("saved", out)
