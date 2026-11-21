#!/bin/bash
# Two-point test in the time domain (figs 2-4, panels C_ij(tau)): per draw (N, seed) the simulation (scripts/run_twopoint.py),
# then the comparison with the fixed-J theory (scripts/analyze_twopoint.py) -> results/twopoint_metrics_N*_seed*_sigma2_9.json.
# Sequential and resumable: a draw whose metrics file exists is skipped, so a job that hits the time limit is resubmitted.
# Protocol: RK4, dt = 0.05, transient 500, B = 128 trajectories, samples every 1 time unit, lags 0, 2, 4; all rows kept for
# N <= 2000 and 2000 random rows above. T_total = B T_traj: 2.05e5 up to N = 2000, 4e5 at 4000, 1.6e6 at 8000, 2.5e6 at 10000.
# Theory: dw = min(0.01, 0.4 strip), w_max = 8. The strip (a CPU eigendecomposition) runs in the background during the simulation.
# Seeds 1-5 for N <= 4000, 1-3 for N = 8000, 10000. (N = 100 was run too; the figures start at N = 250.)
# Measured on one A100: about 1 min per draw up to N = 1000, 1.6 min at 2000, 4.3 min at 4000, 31 min at 8000, 67 min at 10000; 6 h in all.
# usage: sbatch --gpus=1 --cpus-per-task=8 --mem=48G --time=20:00:00 [options of your cluster] slurm/twopoint.sbatch
#        (the resources of the paper's runs; the script carries no #SBATCH lines)
set -u
cd "$SLURM_SUBMIT_DIR"            # submit from the repository root; activate the conda env `scs` (with cupy) before sbatch
mkdir -p logs
export SCS_BACKEND=cupy
python -c "import numpy, scipy, cupy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cupy', cupy.__version__)" \
    || { echo "environment problem: the conda env scs with cupy must be active"; exit 1; }
for f in results/dmft_sigma2_9.npz; do [ -f $f ] || { echo "$f missing"; exit 1; }; done
S2=9; TAUS="0 2 4"; DENSE=40; WMAX=8; DT=0.05; B=128

run_draw () {   # run_draw N SEED T_TRAJ
    local N=$1 SEED=$2 TTRAJ=$3 TAG NPZ STRIP
    TAG=N${N}_seed${SEED}_sigma2_${S2}; NPZ=results/twopoint_${TAG}.npz; STRIP=results/strip_${TAG}.json
    if [ -f results/twopoint_metrics_${TAG}.json ]; then echo "$(date) $TAG already done, skipping"; return; fi
    echo "$(date) START $TAG  (B = $B, T_traj = $TTRAJ, T_total = $((B * TTRAJ)))"
    [ -f "$STRIP" ] || python scripts/strip_only.py $N $SEED $S2 $STRIP > logs/twopoint_strip_${TAG}.log 2>&1 &
    if [ ! -f "$NPZ" ]; then
        python scripts/run_twopoint.py --N $N --seed $SEED --sigma2 $S2 --dt $DT --B $B --T_traj $TTRAJ --taus $TAUS --dense_tau_max $DENSE --out $NPZ \
            > logs/twopoint_run_${TAG}.log 2>&1 || { echo "$(date) run FAILED for $TAG (see logs/twopoint_run_${TAG}.log)"; wait; return; }
    fi
    wait   # the strip job (if its file is missing the analysis computes the eigenvalues itself)
    # The analysis records the draw even when the strip is <= 0 (theory undefined): that is a result, not an error.
    python scripts/analyze_twopoint.py $NPZ --tag $TAG --w_max $WMAX --dw 0.01 --dw_auto > logs/twopoint_analyze_${TAG}.log 2>&1 \
        || echo "$(date) analysis FAILED for $TAG (see logs/twopoint_analyze_${TAG}.log)"
    echo "$(date) DONE  $TAG"
}
for N in 100 250 500 1000 2000; do for SEED in 1 2 3 4 5; do run_draw $N $SEED 1600;  done; done
for SEED in 1 2 3 4 5;              do run_draw 4000  $SEED 3125;  done
for SEED in 1 2 3;                  do run_draw 8000  $SEED 12500; done
for SEED in 1 2 3;                  do run_draw 10000 $SEED 19600; done
echo "$(date) two-point job finished"
