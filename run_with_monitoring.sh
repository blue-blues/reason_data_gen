#!/bin/bash
# Script to run the pipeline with GPU monitoring on Linux

echo "==================================================="
echo "Mathematical Reasoning Pipeline with GPU Monitoring"
echo "==================================================="
echo ""
echo "This script runs the pipeline with GPU monitoring."
echo ""

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
if [ $? -ne 0 ]; then
    echo "Error checking CUDA availability"
    exit 1
fi

# Generate timestamp for the output file
timestamp=$(date +"%Y%m%d_%H%M%S")

# Start GPU monitoring in a separate terminal
gnome-terminal -- python monitor_gpu.py --interval 5.0 --output "gpu_usage_${timestamp}.csv" || \
xterm -e "python monitor_gpu.py --interval 5.0 --output gpu_usage_${timestamp}.csv" || \
konsole -e "python monitor_gpu.py --interval 5.0 --output gpu_usage_${timestamp}.csv" || \
(python monitor_gpu.py --interval 5.0 --output "gpu_usage_${timestamp}.csv" &)

# Wait a moment for monitoring to start
sleep 3

# Run the optimized pipeline with warning suppression
python run_optimized_dataset_pipeline.py --dataset numina_math --force-download --batch-size 4 --num-examples -1 --optimize-gpu --no-warnings
if [ $? -ne 0 ]; then
    echo "Error running optimized pipeline"
    exit 1
fi

echo "Pipeline completed successfully!"
echo ""
echo "GPU monitoring is still running in a separate terminal."
echo "Press Ctrl+C in that terminal to stop monitoring."
echo ""
echo "Press Enter to exit..."
read
