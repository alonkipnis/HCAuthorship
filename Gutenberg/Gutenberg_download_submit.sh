#!/bin/bash
#BATCH --job-name=dask
#SBATCH --partition='hns'
#SBATCH --time=06:00:00

python3 Gutenberg_download.py -i full_catalog.csv -o /scratch/users/kipnisal/Data/Gutenberg  >> $HOME/cron.log

