#!/bin/bash
# Script to test the pipeline with a small subset of data on Linux

echo "==================================================="
echo "Mathematical Reasoning Pipeline - Test Subset"
echo "==================================================="
echo ""
echo "This script runs the pipeline on a small subset of the dataset to test if the approach works."
echo ""

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
if [ $? -ne 0 ]; then
    echo "Error checking CUDA availability"
    exit 1
fi

# Run the optimized pipeline with warning suppression in low memory mode on a small subset
python run_optimized_dataset_pipeline_low_memory.py --dataset numina_math --force-download --batch-size 1 --max-tokens 4096 --cleanup-delay 2.0 --optimize-gpu --no-warnings --num-examples 10
if [ $? -ne 0 ]; then
    echo "Error running test subset"
    exit 1
fi

echo "Test subset completed successfully!"
echo ""
echo "Press Enter to exit..."
read
