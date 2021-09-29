#!/bin/bash
#BATCH --job-name=dask
#SBATCH --partition='owners'
#SBATCH --time=48:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

python3 run_all.py -i /scratch/users/kipnisal/Data/counts.pkl -n 10 >> run_all_all.log
