#!/bin/bash
# Four-point test (figs 2-4, panels C_ijkl): scripts/run_fourpoint.py does the simulation, the theory and the reduction of one draw
# -> results/fourpoint_metrics_N*_seed*_sigma2_9.json (and results/fourpoint_data_*.npz). Sequential and resumable.
# Protocol: RK4 dt 0.05, transient 500, K = 8 groups of trajectories, n_S = 128 sites, lags 0 and 2. T_total = B T_traj: 4.1e6
# (N <= 2000), 8.2e6 (4000), 1.6e7 (8000), 2.6e7 (10000). Theory: dw = min(0.01, 0.4 strip), w_max = 8.
# Measured on one H100: 3.5 min per draw up to N = 2000, 26 min at 4000, 2 h 30 min at 8000, 6 h 50 min at 10000: about 31 h in all,
# i.e. several submissions (resubmit after the time limit; finished draws are skipped). 17 GB of GPU memory for the accumulators.
# The paper's data also hold seeds 4, 5 at N = 8000; the figures use seeds 1-3 at N >= 8000, as in the other experiments.
# usage: sbatch --gpus=1 --cpus-per-task=8 --mem=64G --time=20:00:00 [options of your cluster] slurm/fourpoint.sbatch
#        (the resources of the paper's runs; the script carries no #SBATCH lines)
set -u
cd "$SLURM_SUBMIT_DIR"            # submit from the repository root; activate the conda env `scs` (with cupy) before sbatch
mkdir -p logs
export SCS_BACKEND=cupy
python -c "import numpy, scipy, cupy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cupy', cupy.__version__)" \
    || { echo "environment problem: the conda env scs with cupy must be active"; exit 1; }
for f in results/dmft_sigma2_9.npz; do [ -f $f ] || { echo "$f missing"; exit 1; }; done
S2=9; WMAX=8; DT=0.05; K=8; NS=128; LAG=2

run_draw () {   # run_draw N SEED B T_TRAJ
    local N=$1 SEED=$2 B=$3 TTRAJ=$4 TAG STRIP
    TAG=N${N}_seed${SEED}_sigma2_${S2}; STRIP=results/strip_${TAG}.json
    if [ -f results/fourpoint_metrics_${TAG}.json ]; then echo "$(date) $TAG already done, skipping"; return; fi
    echo "$(date) START $TAG  (B = $B, T_traj = $TTRAJ, T_total = $((B * TTRAJ)))"
    [ -f "$STRIP" ] || python scripts/strip_only.py $N $SEED $S2 $STRIP > logs/fourpoint_strip_${TAG}.log 2>&1 &
    python scripts/run_fourpoint.py --N $N --seed $SEED --sigma2 $S2 --dt $DT --B $B --K $K --n_S $NS --T_traj $TTRAJ --lag $LAG \
        --dw 0.01 --dw_auto --w_max $WMAX --strip_wait 3600 > logs/fourpoint_run_${TAG}.log 2>&1 \
        || echo "$(date) run FAILED for $TAG (see logs/fourpoint_run_${TAG}.log)"
    wait
    echo "$(date) DONE  $TAG"
}
for N in 250 500 1000 2000; do for SEED in 1 2 3 4 5; do run_draw $N $SEED 1024 4000; done; done
for SEED in 1 2 3 4 5; do run_draw 4000  $SEED 512  16000;  done
for SEED in 1 2 3;     do run_draw 8000  $SEED 256  64000;  done
for SEED in 1 2 3;     do run_draw 10000 $SEED 128  200000; done
echo "$(date) four-point job finished"
