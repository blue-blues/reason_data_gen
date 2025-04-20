#!/bin/bash

echo "==================================================="
echo "Mathematical Reasoning Pipeline - Memory Optimized"
echo "==================================================="
echo ""
echo "This script runs the pipeline using all available GPUs with memory optimizations."
echo ""

# Check CUDA availability and count GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [ $? -ne 0 ]; then
    echo "Error checking CUDA availability"
    exit 1
fi

echo "Found $GPU_COUNT GPU(s)"

# Set world size to the number of GPUs
export WORLD_SIZE=$GPU_COUNT

# Run the distributed pipeline with memory optimizations
python run_distributed.py --dataset numina_math --force-download --batch-size 1 --max-tokens 4096 --cleanup-delay 2.0 --optimize-gpu --no-warnings
if [ $? -ne 0 ]; then
    echo "Error running distributed pipeline"
    exit 1
fi

echo "Pipeline completed successfully!"
echo ""
echo "Press Enter to exit..."
read
