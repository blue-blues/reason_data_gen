# Mathematical Reasoning Pipeline - Quick Start Guide

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- Required Python packages installed (`pip install -r requirements.txt`)

## Running the Pipeline

### Option 1: Using the Menu-Driven Interface (Recommended)

1. Open a command prompt in the project directory
2. Run:
   ```
   run_pipeline.bat
   ```
3. Select an option from the menu:
   - **Option 1**: Run optimized pipeline (NuminaMath dataset)
   - **Option 2**: Run with GPU monitoring
   - **Option 3**: Run all datasets
   - **Option 4**: Exit

### Option 2: Direct Command Line

Run the optimized pipeline with the NuminaMath dataset:
```
python run_optimized_dataset_pipeline.py --dataset numina_math --force-download --batch-size 4 --num-examples -1 --optimize-gpu --no-warnings
```

## Command Line Arguments

- `--dataset`: Dataset to use (e.g., numina_math, gsm8k, math_qa)
- `--force-download`: Force download of dataset even if cached data exists
- `--batch-size`: Number of examples to process in parallel (default: 4)
- `--num-examples`: Number of examples to process (-1 for all)
- `--optimize-gpu`: Apply GPU optimizations
- `--no-warnings`: Suppress warning messages

## Available Datasets

- `gsm8k`: Grade School Math 8K
- `math_qa`: Mathematics question answering
- `math`: Challenging math problems
- `asdiv`: Arithmetic word problems
- `aqua`: Algebra word problems
- `maths_college`: College-level mathematics
- `numina_math`: NuminaMath competition problems

## Output

After the pipeline completes, you can find the results in:
- `./data/final_data.json` - JSON format
- A CSV file will also be generated in the same directory
