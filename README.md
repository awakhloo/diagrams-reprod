This repository contains code to reproduce the numerical results in the paper *Solution of a chaotic neural network at fixed connectivity*. 

To reproduce the figures using the results stored on the repo, run

```
conda env create -f environment.yml && conda activate scs
python figures/make_fig1.py
python figures/make_figs2-4.py
```

This rewrites the four PDFs (and PNG copies) in `figures/` and the plotted numbers in `figures/figure_data_*.txt`.
With matplotlib 3.11.1 the PDFs are byte-identical to the ones in the paper apart from the creation date.

The repo is organized as follows. 

```
scs/        the library: network and integrator (network.py), estimators of C (twopoint.py), R (response.py) and the
            four-point function (fourpoint.py), the DMFT solver (dmft.py), the fixed-J theory (theory.py, response.py)
scripts/    one script per stage of each experiment 
slurm/      the six GPU jobs with the parameters of the paper
figures/    the two figure scripts and their output
results/    the data the figures read, and where the scripts write
tests/      unit tests of the library (pytest, about 3 minutes on a CPU, no data needed)
```

All code was written in collaboration with Claude. 
