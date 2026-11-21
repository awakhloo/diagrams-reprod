#!/bin/bash
# Data of Fig 1, right: dense response functions R_ij(tau) on (0, 10] of the same network (N = 10000, seed 1) for five kicked sites,
# read out on the 32 rows of the dense correlation run (run slurm/dense_cc.sbatch first: results/dense_cc_sigma2_9.npz).
# scripts/response_theory.py (CPU, background) + scripts/run_response.py (GPU) -> scripts/analyze_response.py -> scripts/dense_rc.py
# -> results/dense_rc_N10000_seed1_sigma2_9.npz.
# Protocol: RK4 dt 0.05, B = 512 trajectories, kicks of eps = 1e-5, windows of 10 time units, readouts every 0.25, 86 N kicks per
# site (1680 windows). Measured on one H100: 8 h 18 min.
# usage: sbatch --gpus=1 --cpus-per-task=8 --mem=48G --time=24:00:00 [options of your cluster] slurm/dense_response.sbatch
#        (the resources of the paper's runs; the script carries no #SBATCH lines)
set -u
cd "$SLURM_SUBMIT_DIR"            # submit from the repository root; activate the conda env `scs` (with cupy) before sbatch
mkdir -p logs
export SCS_BACKEND=cupy
python -c "import numpy, scipy, cupy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cupy', cupy.__version__)" \
    || { echo "environment problem: the conda env scs with cupy must be active"; exit 1; }
for f in results/dmft_sigma2_9.npz results/dense_cc_sigma2_9.npz; do [ -f $f ] || { echo "$f missing"; exit 1; }; done
S2=9; SEED=1; N=10000; SITES="3273 4762 4880 7130 8515"; B=512; EPS=1e-5; KPN=86; TAUWIN=10; DTAU=0.25; NLAGS=40
TAG=N${N}_seed${SEED}_sigma2_${S2}; RTAG=dense_${TAG}

if [ ! -f results/response_metrics_${RTAG}.json ]; then
    echo "$(date) START $RTAG  (sites $SITES, B = $B, eps = $EPS, tau_win = $TAUWIN, kicks per site = $KPN N)"
    [ -f results/response_theory_${RTAG}.npz ] || python scripts/response_theory.py $N $SEED $S2 --sites $SITES --taus 1 2 4 --dtau $DTAU --n_lags $NLAGS --tag $RTAG > logs/dense_response_theory_${TAG}.log 2>&1 &
    if [ ! -f results/response_${RTAG}.npz ]; then
        python scripts/run_response.py --N $N --seed $SEED --sigma2 $S2 --B $B --sites $SITES --eps $EPS --kicks_per_N $KPN --tau_win $TAUWIN --dtau $DTAU --tag $RTAG > logs/dense_response_run_${TAG}.log 2>&1 \
            || { echo "$(date) run FAILED (see logs/dense_response_run_${TAG}.log)"; wait; exit 1; }
    fi
    wait
    [ -f results/response_theory_${RTAG}.npz ] || { echo "theory FAILED (see logs/dense_response_theory_${TAG}.log)"; exit 1; }
    python scripts/analyze_response.py results/response_${RTAG}.npz --tag $RTAG > logs/dense_response_analyze_${TAG}.log 2>&1 \
        || { echo "$(date) analysis FAILED (see logs/dense_response_analyze_${TAG}.log)"; exit 1; }
fi
export SCS_BACKEND=numpy
python scripts/dense_rc.py $N $SEED $S2 2>&1 | tee logs/dense_rc_${TAG}.log
echo "$(date) dense response job finished"
