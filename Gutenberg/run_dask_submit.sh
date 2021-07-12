#!/bin/bash
#BATCH --job-name=dask
#SBATCH --partition='hns'
#SBATCH --time=06:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

python3 run_all_dask.py -i /scratch/users/kipnisal/Data/Gutenberg_reduced.csv

