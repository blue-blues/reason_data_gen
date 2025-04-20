#!/bin/bash
# Main entry point for the mathematical reasoning pipeline on Linux

echo "==================================================="
echo "Mathematical Reasoning Pipeline"
echo "==================================================="
echo ""
echo "Available options:"
echo "1. Run optimized pipeline (NuminaMath dataset) - RECOMMENDED"
echo "   * Fastest and most efficient option"
echo "   * Uses GPU optimizations and warning suppression"
echo "   * Processes the NuminaMath dataset"
echo ""
echo "2. Run with GPU monitoring (NuminaMath dataset)"
echo "   * Same as option 1 but with GPU usage monitoring"
echo "   * Useful for performance analysis"
echo ""
echo "3. Run all datasets (may take a VERY long time)"
echo "   * Processes all available datasets sequentially"
echo "   * Can take many hours to complete"
echo ""
echo "4. Exit"
echo ""

# Function to check CUDA availability
check_cuda() {
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
    if [ $? -ne 0 ]; then
        echo "Error checking CUDA availability"
        exit 1
    fi
}

# Menu loop
while true; do
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            echo ""
            echo "Running optimized pipeline with NuminaMath dataset..."
            echo ""
            
            # Check CUDA availability
            check_cuda
            
            # Run the optimized pipeline with warning suppression
            python run_optimized_dataset_pipeline.py --dataset numina_math --force-download --batch-size 4 --num-examples -1 --optimize-gpu --no-warnings
            if [ $? -ne 0 ]; then
                echo "Error running optimized pipeline"
                exit 1
            fi
            
            echo "Pipeline completed successfully!"
            break
            ;;
            
        2)
            echo ""
            echo "Running pipeline with GPU monitoring..."
            echo ""
            
            # Generate timestamp for the output file
            timestamp=$(date +"%Y%m%d_%H%M%S")
            
            # Start GPU monitoring in a separate terminal
            gnome-terminal -- python monitor_gpu.py --interval 2 --output "gpu_usage_${timestamp}.csv" || \
            xterm -e "python monitor_gpu.py --interval 2 --output gpu_usage_${timestamp}.csv" || \
            konsole -e "python monitor_gpu.py --interval 2 --output gpu_usage_${timestamp}.csv" || \
            (python monitor_gpu.py --interval 2 --output "gpu_usage_${timestamp}.csv" &)
            
            # Wait a moment for monitoring to start
            sleep 2
            
            # Run the optimized pipeline with warning suppression
            python run_optimized_dataset_pipeline.py --dataset numina_math --force-download --batch-size 4 --num-examples -1 --optimize-gpu --no-warnings
            
            # Display completion message
            echo "Pipeline execution completed."
            echo "GPU monitoring is still running in the other terminal."
            echo "You can close it when you're done analyzing the results."
            break
            ;;
            
        3)
            echo ""
            echo "Running pipeline for all available datasets..."
            echo "This may take a long time to complete."
            echo ""
            
            echo "Processing GSM8K dataset..."
            python run_dataset.py --dataset gsm8k --force-download
            if [ $? -ne 0 ]; then
                echo "Error processing GSM8K dataset"
                exit 1
            fi
            
            echo "Processing MathQA dataset..."
            python run_dataset.py --dataset math_qa --force-download
            if [ $? -ne 0 ]; then
                echo "Error processing MathQA dataset"
                exit 1
            fi
            
            echo "Processing MATH dataset..."
            python run_dataset.py --dataset math --force-download
            if [ $? -ne 0 ]; then
                echo "Error processing MATH dataset"
                exit 1
            fi
            
            echo "Processing ASDiv dataset..."
            python run_dataset.py --dataset asdiv --force-download
            if [ $? -ne 0 ]; then
                echo "Error processing ASDiv dataset"
                exit 1
            fi
            
            echo "Processing AQuA dataset..."
            python run_dataset.py --dataset aqua --force-download
            if [ $? -ne 0 ]; then
                echo "Error processing AQuA dataset"
                exit 1
            fi
            
            echo "Processing Maths-College dataset..."
            python run_dataset.py --dataset maths_college --force-download
            if [ $? -ne 0 ]; then
                echo "Error processing Maths-College dataset"
                exit 1
            fi
            
            echo "Processing NuminaMath dataset..."
            python run_dataset.py --dataset numina_math --force-download
            if [ $? -ne 0 ]; then
                echo "Error processing NuminaMath dataset"
                exit 1
            fi
            
            echo "All datasets processed successfully!"
            break
            ;;
            
        4)
            echo "Exiting..."
            exit 0
            ;;
            
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
done

echo ""
echo "Press Enter to exit..."
read
