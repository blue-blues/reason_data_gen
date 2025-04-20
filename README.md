# Mathematical Reasoning Data Pipeline

A comprehensive pipeline for generating high-quality mathematical reasoning data using Chain-of-Thought (CoT) techniques and iterative self-improvement.

## Overview

This pipeline implements a self-improving CoT data generation system that:

1. **Downloads seed data** from sources like Hugging Face
2. **Generates mathematical reasoning** using Chain-of-Thought prompting
3. **Self-evaluates** the quality of the generated reasoning
4. **Iteratively improves** the reasoning based on evaluation feedback
5. **Uploads the final high-quality data** for use in training or fine-tuning models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mathematical-reasoning-pipeline.git
cd mathematical-reasoning-pipeline

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

The easiest way to run the pipeline is using the provided script for your platform:

### Windows
```bash
# Run the menu-driven interface
run_pipeline.bat
```

### Linux
```bash
# First, make all scripts executable
chmod +x *.sh
# Or run the helper script
./make_scripts_executable.sh

# Then run the menu-driven interface
./run_pipeline.sh
```

This will present a menu with options to:
1. Run the optimized pipeline with NuminaMath dataset
2. Run with GPU monitoring
3. Run all available datasets
4. Exit

## Advanced Usage

```bash
# Run with optimized settings (recommended)
python run_optimized_dataset_pipeline.py --dataset numina_math --force-download --batch-size 4 --num-examples -1 --optimize-gpu --no-warnings
```

Parameters:
- `--dataset`: Dataset to use (e.g., numina_math, gsm8k, math_qa)
- `--force-download`: Force download of dataset even if cached data exists
- `--batch-size`: Number of examples to process in parallel (default: 4)
- `--num-examples`: Number of examples to process (-1 for all)
- `--optimize-gpu`: Apply GPU optimizations
- `--no-warnings`: Suppress warning messages

## Available Datasets

- **GSM8K**: Grade School Math 8K - Elementary math word problems
- **MathQA**: Mathematics question answering dataset
- **MATH**: Challenging math problems (algebra, geometry, etc.)
- **ASDiv**: Arithmetic word problems with diverse structures
- **AQuA**: Algebra word problems with rationales
- **Maths-College**: College-level mathematics problems with detailed solutions
- **NuminaMath**: High-quality competition-level math problems with detailed solutions

You can list all available datasets with:

```bash
python run_optimized_dataset_pipeline.py --list-datasets
```

## Project Structure

### Core Components
- `config.py` - Central configuration file with all settings
- `data_downloader.py` - Downloads mathematical problems from datasets
- `reasoning_generator.py` - Generates step-by-step reasoning
- `self_evaluator.py` - Evaluates the quality of generated reasoning
- `iterative_improver.py` - Improves reasoning based on evaluation
- `data_uploader.py` - Saves the final data to disk

### Utilities
- `utils.py` - General utility functions used throughout the pipeline
- `dataset_utils.py` - Utilities for working with Hugging Face datasets
- `model_utils.py` - Utilities for loading and optimizing models

### Optimization
- `optimize_gpu.py` - Functions to optimize GPU settings
- `suppress_warnings.py` - Functions to suppress unnecessary warnings
- `monitor_gpu.py` - Tool to monitor GPU usage during processing

### Runner
- `run_pipeline.bat` / `run_pipeline.sh` - Main entry point (menu-driven interface)
- `run_optimized_dataset_pipeline.py` - Optimized pipeline runner
- `run_optimized_dataset_pipeline_low_memory.py` - Low memory version for large datasets
- `run_distributed.py` - Distributed processing across multiple GPUs (fixed version)
- `run_low_memory.bat` / `run_low_memory.sh` - Run with low memory optimizations
- `run_distributed.bat` / `run_distributed.sh` - Run with multiple GPUs (fixed version)
- `run_distributed_low_memory.bat` / `run_distributed_low_memory.sh` - Run with multiple GPUs in low memory mode (fixed version)
- `run_with_monitoring.bat` / `run_with_monitoring.sh` - Run with GPU monitoring
- `run_test_subset.bat` / `run_test_subset.sh` - Test with a small subset of data
- `make_scripts_executable.sh` - Make all shell scripts executable (Linux only)

## Output

The pipeline generates data in both JSON and CSV formats:

- `./data/final_data.json` - Complete dataset in JSON format
- `./data/final_data_YYYYMMDD_HHMMSS.csv` - Dataset in CSV format for easy viewing

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets (Hugging Face)
- NVIDIA GPU with CUDA support (recommended)

## Advanced Features

### Low Memory Mode

If you encounter memory issues with large datasets like NuminaMath, use the low memory mode:

#### Windows
```bash
# Run with low memory settings
run_low_memory.bat
```

#### Linux
```bash
# Run with low memory settings
./run_low_memory.sh
```

Low memory mode processes examples one at a time with enhanced memory management. See `README_LOW_MEMORY.md` for more details.

### Multi-GPU Processing

To accelerate processing using multiple GPUs:

#### Windows
```bash
# Run with multiple GPUs
run_distributed.bat
```

#### Linux
```bash
# Run with multiple GPUs
./run_distributed.sh
```

Or with low memory mode:

#### Windows
```bash
# Run with multiple GPUs in low memory mode
run_distributed_low_memory.bat
```

#### Linux
```bash
# Run with multiple GPUs in low memory mode
./run_distributed_low_memory.sh
```

Multi-GPU processing distributes the workload across all available GPUs, significantly reducing processing time. See `MULTI_GPU_GUIDE.md` for detailed instructions and advanced configuration options.

### Linux-Specific Instructions

For detailed instructions on running the pipeline on Linux systems, see `LINUX_GUIDE.md`.
