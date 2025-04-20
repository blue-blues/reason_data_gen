# Memory Optimization Guide

This guide explains how to optimize memory usage when running the Mathematical Reasoning Pipeline, especially when working with large models like DeepSeek-R1-Distill-Qwen-7B.

## Common Memory Issues

When running the pipeline, you might encounter CUDA out of memory errors like:

```
CUDA out of memory. Tried to allocate 1.02 GiB. GPU 0 has a total capacity of 14.74 GiB of which 640.12 MiB is free.
```

This typically happens when:
1. The model is too large for your GPU
2. Multiple copies of the model are loaded simultaneously
3. Batch sizes are too large
4. Generated sequences are too long

## Memory-Optimized Scripts

We've provided memory-optimized scripts that implement several techniques to reduce memory usage:

- **Windows**: `run_distributed_memory_optimized.bat`
- **Linux**: `run_distributed_memory_optimized.sh`

These scripts:
1. Use a smaller batch size (1 instead of 4)
2. Reduce maximum token length (4096 instead of 8196)
3. Reuse the same model instance across pipeline stages
4. Implement aggressive memory cleanup between stages

## Configuration Options

You can further optimize memory usage by modifying these settings in `config.py`:

```python
# Memory optimization settings
LOW_MEMORY_MODE = True  # Enable memory optimizations
CLEANUP_BETWEEN_BATCHES = True  # Clean up memory between batches
MAX_GPU_MEMORY = "12GiB"  # Limit GPU memory usage
CPU_OFFLOAD = True  # Enable CPU offloading for layers that don't fit in GPU memory

# Pipeline settings
MAX_ITERATIONS = 2  # Number of improvement iterations
BATCH_SIZE = 1  # Number of examples to process in parallel
MAX_TOKENS = 4096  # Maximum number of tokens to generate
```

## Command Line Options

When running the pipeline manually, you can use these command line options:

```bash
python run_distributed.py --batch-size 1 --max-tokens 4096 --cleanup-delay 2.0 --optimize-gpu
```

## Advanced Memory Optimization Techniques

1. **Model Quantization**: The pipeline uses 8-bit quantization by default. You can experiment with 4-bit quantization by changing `MODEL_QUANTIZATION = "4bit"` in `config.py`.

2. **Gradient Checkpointing**: This is enabled automatically in low memory mode to reduce memory usage during forward passes.

3. **CPU Offloading**: Moves some model layers to CPU when not in use. This is enabled by default with `CPU_OFFLOAD = True`.

4. **Model Sharing**: The pipeline now reuses the same model instance across different stages instead of loading multiple copies.

5. **Memory Cleanup**: Aggressive garbage collection and CUDA cache clearing between processing stages.

## Monitoring Memory Usage

You can monitor GPU memory usage by running:

```bash
python monitor_gpu.py
```

This will show real-time memory usage for all GPUs while the pipeline is running.
