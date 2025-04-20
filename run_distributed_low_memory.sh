#!/bin/bash
# Script to run the fixed distributed pipeline in low memory mode on Linux

echo "==================================================="
echo "Mathematical Reasoning Pipeline - Fixed Distributed Low Memory Mode"
echo "==================================================="
echo ""
echo "This script runs the fixed distributed pipeline in low memory mode."
echo ""

# Source the CUDA warning suppression script if it exists
if [ -f "./suppress_cuda_warnings.sh" ]; then
    echo "Sourcing CUDA warning suppression script..."
    source ./suppress_cuda_warnings.sh
fi

# Set additional environment variables to suppress CUDA warnings
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export NCCL_DEBUG=WARN

# Set distributed environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Count available GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "Error checking CUDA availability"
    exit 1
fi

echo "Found $GPU_COUNT GPU(s)"

# Set world size to the number of GPUs
export WORLD_SIZE=$GPU_COUNT

# Run the fixed distributed pipeline in low memory mode
python run_distributed_fixed.py --dataset numina_math --force-download --batch-size 1 --max-tokens 4096 --cleanup-delay 2.0 --optimize-gpu --no-warnings --low-memory
if [ $? -ne 0 ]; then
    echo "Error running fixed distributed pipeline in low memory mode"
    exit 1
fi

echo "Pipeline completed successfully!"
echo ""
echo "Press Enter to exit..."
read
