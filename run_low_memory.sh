#!/bin/bash
# Script to run the pipeline in low memory mode on Linux

echo "==================================================="
echo "Mathematical Reasoning Pipeline - Low Memory Mode"
echo "==================================================="
echo ""
echo "This script runs the pipeline in low memory mode, which:"
echo " - Processes examples one at a time (batch size = 1)"
echo " - Uses reduced token length for generation"
echo " - Adds delays between examples for memory cleanup"
echo " - Enables all memory optimization features"
echo ""
echo "This mode is slower but more stable for large datasets."
echo ""

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
if [ $? -ne 0 ]; then
    echo "Error checking CUDA availability"
    exit 1
fi

# Run the optimized pipeline with warning suppression in low memory mode
python run_optimized_dataset_pipeline_low_memory.py --dataset numina_math --force-download --batch-size 1 --max-tokens 4096 --cleanup-delay 2.0 --optimize-gpu --no-warnings
if [ $? -ne 0 ]; then
    echo "Error running optimized pipeline in low memory mode"
    exit 1
fi

echo "Pipeline completed successfully!"
echo ""
echo "Press Enter to exit..."
read
