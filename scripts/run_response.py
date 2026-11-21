"""Response functions of one (N, seed) by perturbed copies: simulate the tanh network, kick n_col sites
(drawn from the seed, or the sites given with --sites) by eps at the start of every window of tau_win, read out the
finite differences every dtau, keep half-split sums.

Output results/response_{tag}.npz (tag = N{N}_seed{seed}_sigma2_{s2}, or --tag) with, for fields x and phi,
  {f}_h1, {f}_h2 : (n_lag, n_col, N) half-split estimates, row a, column i = R_{i, S[a]}(tau) (the kicked sites S are the rows)
  S, taus, lyap  : kicked sites, lags (dtau .. tau_win), Lyapunov exponent per trajectory
plus every parameter, the chaos check and timings in the metadata. The theory is scripts/response_theory.py, the
comparison scripts/analyze_response.py.
"""
import sys, os, time, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scs.backend import xp, asnumpy, BACKEND
from scs.network import make_J, initial_state, simulate, lyapunov_exponent, trajectory_decorrelation
from scs.response import KickAccumulator, chosen_sites
from scs.twopoint import offdiag_mask, noise_floor
from scs.io import save_run

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, required=True)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--sigma2", type=float, default=9.0)
p.add_argument("--dt", type=float, default=0.05)
p.add_argument("--stepper", default="rk4")
p.add_argument("--B", type=int, default=128, help="reference trajectories; the state array has B (1 + n_col) columns")
p.add_argument("--n_col", type=int, default=8, help="kicked sites (columns of R), drawn from the seed (scs.response.kick_sites)")
p.add_argument("--sites", type=int, nargs="+", default=None, help="kick these sites instead of the seed's draw; n_col = their number")
p.add_argument("--eps", type=float, default=1e-4, help="kick size (forward difference; relative bias ~1.8 eps)")
p.add_argument("--kicks_per_N", type=float, default=4.0, help="kicks per column = kicks_per_N * N, rounded up to whole windows")
p.add_argument("--tau_win", type=float, default=4.0, help="window length = largest lag")
p.add_argument("--dtau", type=float, default=0.25, help="readout spacing")
p.add_argument("--T_trans", type=float, default=500.0)
p.add_argument("--T_lyap", type=float, default=100.0)
p.add_argument("--tag", default=None)
p.add_argument("--out", default=None)
a = p.parse_args()

t_start = time.time()
tag = a.tag or f"N{a.N}_seed{a.seed}_sigma2_{a.sigma2:g}"
n_steps = int(round(a.tau_win / a.dt)); every = int(round(a.dtau / a.dt))
assert abs(a.tau_win / a.dt - n_steps) < 1e-9 and abs(a.dtau / a.dt - every) < 1e-9 and n_steps % every == 0, "tau_win and dtau must be multiples of dt"
readout = list(range(every, n_steps + 1, every))
n_win = math.ceil(a.kicks_per_N * a.N / a.B)
J = make_J(a.N, a.sigma2, a.seed)
S = chosen_sites(a.N, a.n_col, a.seed, a.sites)
a.n_col = len(S)                      # with --sites, n_col is their number (recorded in the metadata through vars(a))

x = initial_state(a.N, a.B, seed=a.seed + 777)
x = simulate(x, J, a.dt, int(round(a.T_trans / a.dt)), a.stepper)
lam, x = lyapunov_exponent(x, J, a.dt, a.T_lyap, a.stepper)
decorr = trajectory_decorrelation(x)
lyap_mean, lyap_min = float(xp.mean(lam)), float(xp.min(lam))
print(f"N={a.N} seed={a.seed}: transient done in {time.time()-t_start:.0f}s; lyapunov mean {lyap_mean:.4f} min {lyap_min:.4f}; decorr {decorr:.4f}", flush=True)
print(f"kicks: n_col={a.n_col} sites {S.tolist()}, eps={a.eps:g}, {n_win} windows of {n_steps} steps x B={a.B} = {n_win * a.B} kicks per column, readouts every {every} steps", flush=True)

acc = KickAccumulator(a.N, a.B, S, readout)
t0 = time.time()
x = acc.run(x, J, a.eps, a.dt, n_win, n_steps, a.stepper)
if BACKEND == "cupy":
    xp.cuda.Device().synchronize()
t_sim = time.time() - t0
total_steps = n_win * n_steps
print(f"simulation: {total_steps} steps in {t_sim:.0f}s ({1e3 * t_sim / total_steps:.2f} ms/step with {a.B * (1 + a.n_col)} state columns); max|x| {float(xp.max(xp.abs(x))):.3f}", flush=True)

arrays = {"S": S, "taus": np.array(readout) * a.dt, "lyap": asnumpy(lam), "gain_site": acc.gain_site()}
off = offdiag_mask(S, a.N)
for f in ("x", "phi"):
    est = acc.estimates(f)
    arrays[f"{f}_h1"] = np.stack([est[k][0] for k in readout])
    arrays[f"{f}_h2"] = np.stack([est[k][1] for k in readout])
    for tau in (1.0, 2.0, 4.0):
        if tau > a.tau_win + 1e-9:
            continue
        k = int(round(tau / a.dt)); r1, r2, r = est[k]
        print(f"  {f:3s} tau={tau:g}: RMS off-diag {np.sqrt(np.mean(r[off] ** 2)):.4e}  noise sd {np.sqrt(noise_floor(r1, r2, off)):.2e}  "
              f"mean diag {np.mean(r[~off]):.4f}  max|R| {np.max(np.abs(r)):.3f}", flush=True)
gain_emp = float(np.mean(arrays["gain_site"]))
print(f"empirical mean gain <phi'> = {gain_emp:.5f} (DMFT {float(np.load(f'results/dmft_sigma2_{a.sigma2:g}.npz')['gain']):.5f}); kicked sites: "
      + " ".join(f"{arrays['gain_site'][j]:.4f}" for j in S), flush=True)
meta = dict(vars(a), tag=tag, gain_emp=gain_emp, backend=BACKEND, n_win=n_win, n_steps=n_steps, n_kicks=n_win * a.B, kicks_per_half=acc.kicks_per_half(),
            decorr=decorr, lyap_mean=lyap_mean, lyap_min=lyap_min, t_sim=t_sim, ms_per_step=1e3 * t_sim / total_steps,
            t_wall=time.time() - t_start, max_abs_x=float(xp.max(xp.abs(x))))
out = a.out or f"results/response_{tag}.npz"
save_run(out, arrays, meta)
print(f"saved {out} ({time.time() - t_start:.0f}s wall)")
