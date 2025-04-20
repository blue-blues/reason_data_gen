#!/bin/bash
# Script to make all shell scripts executable

echo "Making all shell scripts executable..."

# Find all .sh files in the current directory and make them executable
chmod +x *.sh

echo "Done! All shell scripts are now executable."
echo ""
echo "You can now run the pipeline using:"
echo "./run_pipeline.sh"
echo ""
echo "Or any of the other scripts:"
echo "./run_low_memory.sh"
echo "./run_multi_gpu.sh"
echo "./run_multi_gpu_low_memory.sh"
echo "./run_with_monitoring.sh"
echo "./run_test_subset.sh"
echo ""
echo "Press Enter to exit..."
read
