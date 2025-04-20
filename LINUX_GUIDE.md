# Linux Guide for Mathematical Reasoning Pipeline

This guide provides specific instructions for running the pipeline on Linux systems.

## Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- PyTorch with CUDA support
- Required Python packages

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mathematical-reasoning-pipeline.git
   cd mathematical-reasoning-pipeline
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make all shell scripts executable:
   ```bash
   chmod +x *.sh
   ```
   Or use the helper script:
   ```bash
   ./make_scripts_executable.sh
   ```

## Running the Pipeline

### Option 1: Menu-Driven Interface

Run the main script:
```bash
./run_pipeline.sh
```

This will present a menu with options to:
1. Run the optimized pipeline with NuminaMath dataset
2. Run with GPU monitoring
3. Run all available datasets
4. Exit

### Option 2: Direct Command Line

For standard processing:
```bash
python run_optimized_dataset_pipeline.py --dataset numina_math --force-download --batch-size 4 --num-examples -1 --optimize-gpu --no-warnings
```

For low memory mode:
```bash
python run_optimized_dataset_pipeline_low_memory.py --dataset numina_math --force-download --batch-size 1 --max-tokens 4096 --cleanup-delay 2.0 --optimize-gpu --no-warnings
```

For multi-GPU processing:
```bash
python run_distributed.py --dataset numina_math --force-download --batch-size 4 --optimize-gpu --no-warnings
```

### Option 3: Convenience Scripts

Several scripts are provided for common use cases:

- `./run_pipeline.sh` - Main menu-driven interface
- `./run_low_memory.sh` - Run with low memory optimizations
- `./run_distributed.sh` - Run with multiple GPUs
- `./run_distributed_low_memory.sh` - Run with multiple GPUs in low memory mode
- `./run_with_monitoring.sh` - Run with GPU monitoring
- `./run_test_subset.sh` - Test with a small subset of data

## GPU Monitoring

The pipeline includes a GPU monitoring tool that works on Linux:

```bash
python monitor_gpu.py --interval 5.0 --output gpu_usage.csv
```

When using `run_with_monitoring.sh`, the script attempts to launch the monitoring tool in a separate terminal using one of these methods:
1. gnome-terminal (for GNOME desktop environments)
2. xterm (for X11 environments)
3. konsole (for KDE desktop environments)
4. As a background process if none of the above are available

## Multi-Node Processing

For distributed processing across multiple machines:

On the master node:
```bash
python run_distributed_pipeline.py --num-nodes 2 --node-rank 0 --master-addr <MASTER_IP> --master-port 12355
```

On the worker node:
```bash
python run_distributed_pipeline.py --num-nodes 2 --node-rank 1 --master-addr <MASTER_IP> --master-port 12355
```

Replace `<MASTER_IP>` with the IP address of the master node.

## Troubleshooting

### Common Issues

1. **Permission Denied when Running Scripts**

   If you see "Permission denied" when trying to run a script, make it executable:
   ```bash
   chmod +x script_name.sh
   ```

2. **CUDA Not Available**

   Verify CUDA is properly installed and detected:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

3. **Terminal Not Found for GPU Monitoring**

   If none of the supported terminals are available, the monitoring will run in the background. You can check its output in the specified CSV file.

4. **Process Group Initialization Failed**

   For multi-GPU processing, ensure all GPUs are visible to the system:
   ```bash
   nvidia-smi
   ```

   If using Docker or containers, make sure all GPUs are properly exposed to the container.

5. **CUDA Registration Warnings**

   If you see warnings like these:
   ```
   Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
   Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
   Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
   ```

   Use the provided suppression script before running the pipeline:
   ```bash
   source ./suppress_cuda_warnings.sh && ./run_multi_gpu.sh
   ```

   Or use the specialized wrapper script that handles these warnings:
   ```bash
   ./run_distributed_linux.sh
   ```

6. **DTensor Error**

   If you see this error:
   ```
   'DTensor' object has no attribute 'CB'
   ```

   This is a known issue with distributed tensor operations in PyTorch. Use the fixed distributed scripts instead:
   ```bash
   ./run_distributed_fixed.sh
   ```

   Or for low memory mode:
   ```bash
   ./run_distributed_fixed_low_memory.sh
   ```

   These scripts use a different approach to distributed processing that avoids the DTensor error.

### Environment Variables

You can set these environment variables to control distributed processing:

```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=2  # Number of GPUs
export RANK=0        # Process rank
export LOCAL_RANK=0  # Local process rank
```

## Performance Tips

1. **Disable GUI for Better Performance**

   When running on a headless server, make sure no GUI processes are consuming GPU resources.

2. **Monitor System Resources**

   Use tools like `htop`, `nvidia-smi`, and `free -h` to monitor system resources during processing.

3. **Adjust Process Priority**

   For long-running processes, consider using `nice` to adjust process priority:
   ```bash
   nice -n 10 ./run_multi_gpu.sh
   ```
