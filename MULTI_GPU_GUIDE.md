# Multi-GPU Processing Guide

This guide explains how to use multiple GPUs to process large datasets like NuminaMath more efficiently.

## Overview

The multi-GPU implementation:
1. Splits the dataset across all available GPUs
2. Processes each subset in parallel
3. Combines the results at the end
4. Supports both standard and low memory modes

## Requirements

- Multiple NVIDIA GPUs with CUDA support
- PyTorch with CUDA support
- Enough system memory to load the full dataset

## Quick Start

### Option 1: Using Scripts (Recommended)

#### Windows
For standard multi-GPU processing:
```
run_distributed.bat
```

For multi-GPU processing in low memory mode:
```
run_distributed_low_memory.bat
```

#### Linux
First, make the scripts executable:
```
chmod +x *.sh
```
Or use the helper script:
```
./make_scripts_executable.sh
```

Then run:

For standard multi-GPU processing:
```
./run_distributed.sh
```

For multi-GPU processing in low memory mode:
```
./run_distributed_low_memory.sh
```

### Option 2: Command Line

For standard multi-GPU processing:
```
python run_distributed.py --dataset numina_math --force-download --batch-size 4 --optimize-gpu --no-warnings
```

For multi-GPU processing in low memory mode:
```
python run_distributed.py --dataset numina_math --force-download --batch-size 1 --max-tokens 4096 --cleanup-delay 2.0 --optimize-gpu --no-warnings --low-memory
```

## Advanced Configuration

### Controlling GPU Usage

By default, the pipeline uses all available GPUs. You can limit the number of GPUs:

```
python run_distributed_pipeline.py --world-size 2
```

This will use only 2 GPUs even if more are available.

### Multi-Node Processing

For distributed processing across multiple machines:

On the master node:
```
python run_distributed_pipeline.py --num-nodes 2 --node-rank 0 --master-addr <MASTER_IP> --master-port 12355
```

On the worker node:
```
python run_distributed_pipeline.py --num-nodes 2 --node-rank 1 --master-addr <MASTER_IP> --master-port 12355
```

Replace `<MASTER_IP>` with the IP address of the master node.

## How It Works

1. **Data Distribution**: The dataset is loaded on rank 0 (the first GPU) and then distributed to all other GPUs.

2. **Parallel Processing**: Each GPU processes its subset of the data independently:
   - Generating reasoning
   - Evaluating reasoning
   - Improving reasoning

3. **Result Combination**: After all GPUs finish processing, rank 0 combines the results from all GPUs into a single output file.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Use the low memory mode: `--low-memory`
   - Reduce the batch size: `--batch-size 1`
   - Reduce the maximum token length: `--max-tokens 2048`

2. **Process Hanging**:
   - Increase the cleanup delay: `--cleanup-delay 5.0`
   - Check if all GPUs are functioning properly

3. **Uneven GPU Utilization**:
   - This is normal as different examples may take different amounts of time to process
   - The pipeline automatically balances the workload by assigning roughly equal numbers of examples to each GPU

### Monitoring GPU Usage

You can monitor GPU usage during processing:

```
python monitor_gpu.py --interval 5.0 --output gpu_usage_multi.csv
```

This will log GPU usage every 5 seconds to a CSV file.

## Performance Considerations

- **Speedup**: Using multiple GPUs provides an almost linear speedup (e.g., 2 GPUs ≈ 2x faster)
- **Memory Usage**: Each GPU needs enough memory to load the model and process its subset of data
- **System Memory**: The full dataset is loaded into system memory before being distributed to GPUs

## Advanced Customization

You can modify the distributed processing behavior by editing:
- `config.py`: Change multi-GPU settings
- `run_distributed_pipeline.py`: Customize the distributed processing logic
- `model_utils.py`: Adjust how models are loaded in a distributed environment
