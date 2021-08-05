#!/bin/bash
#BATCH --job-name=dask
<<<<<<< HEAD
#SBATCH --partition='donoho'
#SBATCH --time=24:00:00
=======
#SBATCH --partition='hns'
#SBATCH --time=06:00:00
>>>>>>> 33cf8c3 (merging)
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

<<<<<<< HEAD
python3 run_all.py -i /scratch/users/kipnisal/Data/Gutenberg_reduced.csv -n 10
=======
python3 evaluate_all.py -i /scratch/users/kipnisal/Data/Gutenberg_reduced.csv
>>>>>>> 33cf8c3 (merging)

