#!/bin/bash
#SBATCH --job-name=count_ds
#SBATCH --partition=seas_compute
#--gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH--time=1-00:00
#--time=0-12:00
#SBATCH --output=pn_cars_fasrc_count_%j.out  
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_50,TIME_LIMIT,REQUEUE

source /n/home04/$USER/.bashrc
module load python cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01
mamba activate /n/holylabs/LABS/janapa_reddi_lab/Users/mmaz/wakevision_work/wakevision_env
cd /n/holylabs/LABS/janapa_reddi_lab/Users/mmaz/wakevision_work/Wake_Vision

python experiments/query_ds.py
# python experiments/query_ds.py car
# python experiments/query_ds.py birds
