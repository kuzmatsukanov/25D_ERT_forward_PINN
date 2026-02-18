#!/bin/bash

cd /home/ARO.local/kuzma/Projects/proj_ert_ver

source /data/bin/miniconda2/etc/profile.d/conda.sh
conda activate /home/ARO.local/kuzma/.conda/envs/env_pinn_gpu

python -O run_pinn_training.py
