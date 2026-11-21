#!/bin/bash
# Response functions (figs 2-4, panels R_ij(tau)) by perturbed copies: per draw the simulation (scripts/run_response.py, GPU) with
# the theory (scripts/response_theory.py, CPU matrix exponential) in the background, then scripts/analyze_response.py
# -> results/response_metrics_N*_seed*_sigma2_9.json. Sequential and resumable.
# Protocol: RK4 dt 0.05, transient 500, B = 128 trajectories, kicks of eps = 1e-4 (forward differences), windows of 4 time units,
# readouts every 0.25; per N (kicked sites, kicks per site): 250-2000 (16, 4N), 4000 (16, 8N), 8000 and 10000 (8, 16N).
# Measured on one A100: 8-40 s per draw up to N = 2000, 6 min at 4000, 46 min at 8000, 89 min at 10000; 7.5 h in all.
# The theory needs about 5 GB of host memory at N = 10000.
# usage: sbatch --gpus=1 --cpus-per-task=8 --mem=48G --time=20:00:00 [options of your cluster] slurm/response.sbatch
#        (the resources of the paper's runs; the script carries no #SBATCH lines)
set -u
cd "$SLURM_SUBMIT_DIR"            # submit from the repository root; activate the conda env `scs` (with cupy) before sbatch
mkdir -p logs
export SCS_BACKEND=cupy
python -c "import numpy, scipy, cupy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cupy', cupy.__version__)" \
    || { echo "environment problem: the conda env scs with cupy must be active"; exit 1; }
for f in results/dmft_sigma2_9.npz; do [ -f $f ] || { echo "$f missing"; exit 1; }; done
S2=9; B=128; EPS=1e-4

run_draw () {   # run_draw N SEED NCOL KPN
    local N=$1 SEED=$2 NCOL=$3 KPN=$4 TAG
    TAG=N${N}_seed${SEED}_sigma2_${S2}
    if [ -f results/response_metrics_${TAG}.json ]; then echo "$(date) $TAG already done, skipping"; return; fi
    echo "$(date) START $TAG  (n_col = $NCOL, kicks per site = $KPN N)"
    [ -f results/response_theory_${TAG}.npz ] || python scripts/response_theory.py $N $SEED $S2 --n_col $NCOL > logs/response_theory_${TAG}.log 2>&1 &
    if [ ! -f results/response_${TAG}.npz ]; then
        python scripts/run_response.py --N $N --seed $SEED --sigma2 $S2 --B $B --n_col $NCOL --eps $EPS --kicks_per_N $KPN > logs/response_run_${TAG}.log 2>&1 \
            || { echo "$(date) run FAILED for $TAG (see logs/response_run_${TAG}.log)"; wait; return; }
    fi
    wait
    python scripts/analyze_response.py results/response_${TAG}.npz > logs/response_analyze_${TAG}.log 2>&1 \
        || echo "$(date) analysis FAILED for $TAG (see logs/response_analyze_${TAG}.log)"
    echo "$(date) DONE  $TAG"
}
for N in 250 500 1000 2000; do for SEED in 1 2 3 4 5; do run_draw $N $SEED 16 4; done; done
for SEED in 1 2 3 4 5; do run_draw 4000  $SEED 16 8;  done
for SEED in 1 2 3;     do run_draw 8000  $SEED 8  16; done
for SEED in 1 2 3;     do run_draw 10000 $SEED 8  16; done
echo "$(date) response job finished"
