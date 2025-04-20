"""
Utility functions for loading and optimizing models.
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import Config

# Set up logging
logger = logging.getLogger(__name__)

def load_optimized_model(model_name, device=None, dtype=None, quantization=None):
    """
    Load a model with optimized settings for better GPU utilization.

    Args:
        model_name: The name or path of the model to load
        device: The device to load the model on (default: Config.MODEL_DEVICE)
        dtype: The data type to use (default: Config.MODEL_DTYPE)
        quantization: The quantization method to use (default: Config.MODEL_QUANTIZATION)

    Returns:
        The loaded model and tokenizer
    """
    # Use default values from Config if not specified
    device = device or Config.MODEL_DEVICE
    dtype = dtype or Config.MODEL_DTYPE
    quantization = quantization or Config.MODEL_QUANTIZATION

    logger.info(f"Loading model {model_name} with optimized settings")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Data type: {dtype}")
    logger.info(f"  - Quantization: {quantization}")

    # Convert dtype string to torch dtype
    # When using 8-bit quantization, it's better to use float16 instead of bfloat16
    # to avoid the dtype conversion warning
    if quantization == "8bit" and dtype == "bfloat16":
        logger.info("Using float16 instead of bfloat16 for 8-bit quantization to avoid dtype conversion")
        torch_dtype = torch.float16
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        logger.warning(f"Unknown dtype {dtype}, falling back to float16")
        torch_dtype = torch.float16

    # Set up quantization config if needed
    quantization_config = None
    if quantization:
        if quantization == "4bit":
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch_dtype
            )

    # Set up device map
    if Config.USE_MULTI_GPU:
        if torch.cuda.device_count() > 1:
            # Check if we're in a distributed environment
            try:
                is_distributed = torch.distributed.is_initialized()
                local_rank = torch.distributed.get_rank() if is_distributed else 0
                logger.info(f"Distributed environment detected with rank {local_rank}")

                # In distributed mode, each process should only use its assigned GPU
                device_map = f"cuda:{local_rank}"
                logger.info(f"Using GPU {local_rank} for this distributed process")
            except:
                # If not in a distributed environment, use the configured device map
                device_map = Config.DEVICE_MAP
                logger.info(f"Using multi-GPU with device map: {device_map}")
        else:
            # Only one GPU available
            device_map = device
            logger.info(f"Multi-GPU requested but only one GPU available: {device_map}")
    else:
        device_map = device
        logger.info(f"Using single device: {device_map}")

    # Set up max memory
    max_memory = None
    if hasattr(Config, 'MAX_GPU_MEMORY') and Config.MAX_GPU_MEMORY:
        if isinstance(device_map, dict):
            # For multi-GPU setup
            max_memory = {i: Config.MAX_GPU_MEMORY for i in device_map.keys()}
            max_memory["cpu"] = "32GiB"  # Allow CPU offloading
        else:
            # For single GPU setup
            max_memory = {0: Config.MAX_GPU_MEMORY, "cpu": "32GiB"}

        logger.info(f"Setting max memory: {max_memory}")

    # Load the model with optimized settings
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    if max_memory:
        model_kwargs["max_memory"] = max_memory

    # Enable CPU offloading if configured
    if hasattr(Config, 'CPU_OFFLOAD') and Config.CPU_OFFLOAD:
        logger.info("Enabling CPU offloading")
        model_kwargs["offload_folder"] = "offload"
        os.makedirs("offload", exist_ok=True)

    # Load the model
    try:
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def optimize_inference_settings(model):
    """
    Optimize model settings for inference.

    Args:
        model: The model to optimize

    Returns:
        The optimized model
    """
    logger.info("Optimizing model for inference")

    # Use torch.compile for PyTorch 2.0+ if available
    if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
        try:
            logger.info("Using torch.compile to optimize model")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")

    # Enable flash attention if available
    try:
        if hasattr(model.config, 'use_flash_attention'):
            logger.info("Enabling flash attention")
            model.config.use_flash_attention = True
    except Exception as e:
        logger.warning(f"Failed to enable flash attention: {e}")

    # Set eval mode for inference
    model.eval()

    return model
