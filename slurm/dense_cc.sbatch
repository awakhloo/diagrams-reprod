#!/bin/bash
# Data of Fig 1, left: dense cross-correlation functions C_ij(tau) on [-100, 100] of one network (N = 10000, seed 1) on 32 rows.
# scripts/run_twopoint.py with dense lags -> scripts/analyze_twopoint.py (theory at the same lags) -> scripts/dense_cc.py
# -> results/dense_cc_sigma2_9.npz.
# Protocol: RK4 dt 0.05, B = 128, T_traj = 19700 (T_total = 2.5e6), dense lags 0 .. 100 in steps of 0.25; theory w_max = 8,
# dw = min(0.01, 0.4 strip). Measured on one A100: 72 min simulation, 20 min theory. The run file is about 6 GB; host memory 12 GB.
# usage: sbatch --gpus=1 --cpus-per-task=8 --mem=48G --time=06:00:00 [options of your cluster] slurm/dense_cc.sbatch
#        (the resources of the paper's runs; the script carries no #SBATCH lines)
set -u
cd "$SLURM_SUBMIT_DIR"            # submit from the repository root; activate the conda env `scs` (with cupy) before sbatch
mkdir -p logs
export SCS_BACKEND=cupy
python -c "import numpy, scipy, cupy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cupy', cupy.__version__)" \
    || { echo "environment problem: the conda env scs with cupy must be active"; exit 1; }
for f in results/dmft_sigma2_9.npz; do [ -f $f ] || { echo "$f missing"; exit 1; }; done
S2=9; SEED=1; N=10000; TTRAJ=19700; TAUS="0 2 4"; WMAX=8; DT=0.05; B=128; NROWS=32; DTAU=0.25; TAUMAX=100
TAG=N${N}_seed${SEED}_sigma2_${S2}; RTAG=dense_${TAG}; NPZ=results/twopoint_${RTAG}.npz; STRIP=results/strip_${TAG}.json

echo "$(date) START $TAG  (B = $B, T_traj = $TTRAJ, dense: $NROWS rows, lags 0 .. $TAUMAX step $DTAU)"
[ -f "$STRIP" ] || python scripts/strip_only.py $N $SEED $S2 $STRIP > logs/dense_cc_strip_${TAG}.log 2>&1 &
if [ ! -f "$NPZ" ]; then
    python scripts/run_twopoint.py --N $N --seed $SEED --sigma2 $S2 --dt $DT --B $B --T_traj $TTRAJ --taus $TAUS \
        --n_rows_dense $NROWS --dense_dtau $DTAU --dense_tau_max $TAUMAX --out $NPZ > logs/dense_cc_run_${TAG}.log 2>&1 \
        || { echo "$(date) run FAILED (see logs/dense_cc_run_${TAG}.log)"; wait; exit 1; }
fi
wait   # the strip job
if [ ! -f results/twopoint_metrics_${RTAG}.json ]; then
    python scripts/analyze_twopoint.py $NPZ --tag $RTAG --w_max $WMAX --dw 0.01 --dw_auto > logs/dense_cc_analyze_${TAG}.log 2>&1 \
        || { echo "$(date) analysis FAILED (see logs/dense_cc_analyze_${TAG}.log)"; exit 1; }
fi
export SCS_BACKEND=numpy
python scripts/dense_cc.py $N $SEED $S2 2>&1 | tee logs/dense_cc_${TAG}.log
echo "$(date) dense correlation job finished"
