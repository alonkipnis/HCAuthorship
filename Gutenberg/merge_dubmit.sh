#!/bin/bash
#BATCH --job-name=dask
#SBATCH --partition='hns'
#SBATCH --time=06:00:00

python3 merge_to_collection.py -i /scratch/users/kipnisal/Data/Gutenberg -o /scratch/users/kipnisal/Data/Gutenberg_red -l list_of_titles.csv

