#!/bin/bash
#BATCH --job-name=dask
#SBATCH --partition='donoho'
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

python3 run_all.py -i ./counts.pkl -n 10
