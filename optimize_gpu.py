"""
Script to optimize GPU utilization for the mathematical reasoning pipeline.
"""

import os
import torch
import logging
import subprocess
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_info():
    """Check GPU information and print details."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} CUDA device(s)")
        
        for i in range(gpu_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1024**3:.2f} GB")
            logger.info(f"  - CUDA Capability: {device_properties.major}.{device_properties.minor}")
            logger.info(f"  - Multi-processor count: {device_properties.multi_processor_count}")
            
        # Get current memory usage
        current_device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        logger.info(f"Current GPU memory usage:")
        logger.info(f"  - Allocated: {memory_allocated:.2f} GB")
        logger.info(f"  - Reserved: {memory_reserved:.2f} GB")
    else:
        logger.warning("No CUDA devices available")

def optimize_torch_settings():
    """Optimize PyTorch settings for better GPU utilization."""
    # Enable TF32 precision for better performance on A100 GPUs
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        logger.info("Enabling TF32 precision for better performance on A100 GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmark mode for optimized performance
    logger.info("Enabling cuDNN benchmark mode")
    torch.backends.cudnn.benchmark = True
    
    # Set deterministic algorithms for reproducibility if needed
    if Config.SEED is not None:
        logger.info(f"Setting deterministic algorithms with seed {Config.SEED}")
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)

def optimize_memory_usage():
    """Optimize memory usage for better GPU utilization."""
    # Empty CUDA cache
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache")
        torch.cuda.empty_cache()
    
    # Set memory fraction to use
    if hasattr(Config, 'MAX_GPU_MEMORY') and Config.MAX_GPU_MEMORY:
        logger.info(f"Setting maximum GPU memory usage to {Config.MAX_GPU_MEMORY}")
        # This will be handled by the model loading code
    
    # Enable garbage collection
    import gc
    logger.info("Running garbage collection")
    gc.collect()

def check_mig_configuration():
    """Check if MIG is enabled and print configuration."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        mig_mode = result.stdout.strip()
        logger.info(f"MIG mode: {mig_mode}")
        
        if mig_mode == "Enabled":
            # Get MIG device info
            result = subprocess.run(
                ["nvidia-smi", "--query-mig", "--format=csv"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("MIG configuration:")
            logger.info(result.stdout)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"Failed to check MIG configuration: {e}")

def main():
    """Main entry point."""
    logger.info("Starting GPU optimization")
    
    # Check GPU information
    check_gpu_info()
    
    # Check MIG configuration
    check_mig_configuration()
    
    # Optimize PyTorch settings
    optimize_torch_settings()
    
    # Optimize memory usage
    optimize_memory_usage()
    
    logger.info("GPU optimization completed")

if __name__ == "__main__":
    main()
