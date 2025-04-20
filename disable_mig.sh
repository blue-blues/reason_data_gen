#!/bin/bash
# Script to disable MIG mode and restart the GPU
echo "Checking if we have permission to disable MIG mode..."
sudo nvidia-smi -i 0 -mig 0

# If the above command succeeds, we need to restart the GPU
if [ $? -eq 0 ]; then
    echo "MIG mode disabled. Restarting GPU..."
    sudo nvidia-smi --gpu-reset
    echo "GPU restarted. Checking current configuration..."
    nvidia-smi
else
    echo "Failed to disable MIG mode. You may not have sufficient permissions."
    echo "Current MIG configuration:"
    nvidia-smi --query-gpu=mig.mode.current --format=csv
fi
