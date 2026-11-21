#!/bin/bash
# Two-point test in the Fourier domain (figs 2-4, panels C_ij(omega)): per draw the simulation with dense lags on a site subset
# (scripts/run_spectral.py), then the comparison of its rectangular-window transform with the theory C(omega)
# (scripts/analyze_spectral_direct.py) -> results/spectral_direct_metrics_N*_seed*_sigma2_9.json. Sequential and resumable.
# Protocol: RK4 dt 0.05, transient 500, B = 128 trajectories, n_S = min(N, 256) sites, lags 0 .. 100 in steps of 0.25.
# T_total = B T_traj: 8e5 up to N = 2000, 1.6e6 at 4000, 6.4e6 at 8000, 1e7 at 10000.
# Measured on one H100: 2 min per draw up to N = 2000, 6 min at 4000, 58 min at 8000, 2 h 9 min at 10000; 10.5 h in all.
# The run files are large (0.4 GB at N = 250 to 2.4 GB at N = 10000 each, 30 GB in all).
# usage: sbatch --gpus=1 --cpus-per-task=8 --mem=48G --time=20:00:00 [options of your cluster] slurm/spectral.sbatch
#        (the resources of the paper's runs; the script carries no #SBATCH lines)
set -u
cd "$SLURM_SUBMIT_DIR"            # submit from the repository root; activate the conda env `scs` (with cupy) before sbatch
mkdir -p logs
export SCS_BACKEND=cupy
python -c "import numpy, scipy, cupy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cupy', cupy.__version__)" \
    || { echo "environment problem: the conda env scs with cupy must be active"; exit 1; }
for f in results/dmft_sigma2_9.npz; do [ -f $f ] || { echo "$f missing"; exit 1; }; done
S2=9; B=128; NS=256

run_draw () {   # run_draw N SEED T_TRAJ
    local N=$1 SEED=$2 TTRAJ=$3 TAG NPZ STRIP NSD
    NSD=$(( N < NS ? N : NS ))     # the site subset cannot exceed N (N = 250: the whole network)
    TAG=N${N}_seed${SEED}_sigma2_${S2}; NPZ=results/spectral_${TAG}.npz; STRIP=results/strip_${TAG}.json
    if [ -f results/spectral_direct_metrics_${TAG}.json ]; then echo "$(date) $TAG already done, skipping"; return; fi
    echo "$(date) START $TAG  (B = $B, T_traj = $TTRAJ, T_total = $((B * TTRAJ)))"
    [ -f "$STRIP" ] || python scripts/strip_only.py $N $SEED $S2 $STRIP > logs/spectral_strip_${TAG}.log 2>&1 &
    if [ ! -f "$NPZ" ]; then
        python scripts/run_spectral.py --N $N --seed $SEED --sigma2 $S2 --B $B --T_traj $TTRAJ --n_S $NSD \
            > logs/spectral_run_${TAG}.log 2>&1 || { echo "$(date) run FAILED for $TAG (see logs/spectral_run_${TAG}.log)"; wait; return; }
    fi
    wait
    python scripts/analyze_spectral_direct.py $NPZ > logs/spectral_analyze_${TAG}.log 2>&1 \
        || echo "$(date) analysis FAILED for $TAG (see logs/spectral_analyze_${TAG}.log)"
    echo "$(date) DONE  $TAG"
}
for N in 250 500 1000 2000; do for SEED in 1 2 3 4 5; do run_draw $N $SEED 6250; done; done
for SEED in 1 2 3 4 5; do run_draw 4000  $SEED 12500; done
for SEED in 1 2 3;     do run_draw 8000  $SEED 50000; done
for SEED in 1 2 3;     do run_draw 10000 $SEED 78125; done
echo "$(date) Fourier-domain job finished"
