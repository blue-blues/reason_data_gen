#!/bin/bash
# Script to suppress CUDA registration warnings on Linux

echo "Setting environment variables to suppress CUDA registration warnings..."

# Suppress CUDA registration warnings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Suppress JAX/XLA warnings
export JAX_PLATFORMS=cuda
export JAX_DISABLE_JIT=0

# Suppress NCCL warnings
export NCCL_DEBUG=WARN

# Suppress ABSL warnings
export ABSL_LOGGING_LEVEL=3

echo "Environment variables set. CUDA registration warnings should be suppressed."
echo "Run your command with these environment variables, for example:"
echo ""
echo "source ./suppress_cuda_warnings.sh && ./run_multi_gpu.sh"
echo ""
