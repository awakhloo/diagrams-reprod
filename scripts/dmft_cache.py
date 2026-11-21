"""Solve the DMFT of the tanh network at coupling variance sigma^2 and store the solution used by every theory evaluation:
results/dmft_sigma2_{s2}.npz with the lag grid tau, the single-site correlation functions Cx_tau, Cphi_tau and the gain <phi'>.
usage: dmft_cache.py SIGMA2 [n_gh=256]      (the paper: dmft_cache.py 9 256)"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scs import dmft
s2 = float(sys.argv[1]); n_gh = int(sys.argv[2]) if len(sys.argv) > 2 else 256
t0 = time.time()
sol = dmft.solve(s2, dt=0.05, T_D=400.0, n_gh=n_gh, tol=1e-12)
D0e = dmft.Delta0_from_energy(s2, n_gh=n_gh)
print(f"sigma^2={s2} n_gh={n_gh}: Delta0={sol.Cx_tau[0]:.10f} energy={D0e:.10f} diff={sol.Cx_tau[0]-D0e:.1e}; gain={sol.gain:.10f} sigma*gain={np.sqrt(s2)*sol.gain:.6f} Cphi0={sol.Cphi_tau[0]:.8f} ({time.time()-t0:.0f}s)")
np.savez(f"results/dmft_sigma2_{s2:g}.npz", sigma2=s2, dt=sol.dt, tau=sol.tau, Cx_tau=sol.Cx_tau, Cphi_tau=sol.Cphi_tau, gain=sol.gain, Delta0_energy=D0e, n_gh=n_gh)
print("saved", f"results/dmft_sigma2_{s2:g}.npz")
