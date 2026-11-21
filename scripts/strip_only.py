"""Strip width min_k Re(1 - gain lambda_k) of one J, written to a small json so that the analysis can read it instead of
recomputing the eigenvalues. CPU only (numpy eigvals), so the job scripts run it in the background while the GPU simulates.
usage: strip_only.py N SEED SIGMA2 [out.json]     default out: results/strip_N{N}_seed{SEED}_sigma2_{SIGMA2}.json"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["SCS_BACKEND"] = "numpy"
import numpy as np
from scs.network import make_J
from scs.theory import strip_width
N, seed, s2 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else f"results/strip_N{N}_seed{seed}_sigma2_{s2:g}.json"
gain = float(np.load(f"results/dmft_sigma2_{s2:g}.npz")["gain"])
t0 = time.time()
width, lam = strip_width(make_J(N, s2, seed), gain)
json.dump({"N": N, "seed": seed, "sigma2": s2, "gain": gain, "strip_width": width, "max_abs_lam": float(abs(lam).max()),
           "max_re_lam": float(lam.real.max()), "secs": time.time() - t0}, open(out, "w"), indent=1)
print(f"N={N} seed={seed} sigma^2={s2:g}: strip {width:.4f}, max|lam| {abs(lam).max():.4f} ({time.time()-t0:.0f}s) -> {out}")
