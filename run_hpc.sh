#!/bin/bash
# NOTE: Run once before using this script:
#       chmod +x run_hpc.sh OR: chmod +x run_hpc.sh && ./run_hpc.sh
#
# This script feeds automatic answers into the HPC interactive wizard.
# Queue type: 1 = Normal Queue, 2 = Short Queue, 3 = Long Queue
# Resource type: 1 = CPU mode, 2 = GPU mode
# Number of A100 GPUs: 1, (max: 2)
# Running mode: 1 = Pthreads, 5 = GUI, 6 = Interactive
# Number of CPUs [1-72]
# Amount of RAM in GB [10-1744]
hpc run_pinn_training.sh << 'EOF'
1
2
1
1
20
128

run_pinn_training.sh
EOF