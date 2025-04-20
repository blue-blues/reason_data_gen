"""
Utility functions for the synthetic data pipeline.
"""

import os
import json
import logging
import random
import numpy as np
import torch
from datetime import datetime
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config

def setup_logging(log_level=None, rank_suffix=None):
    """Set up logging configuration.

    Args:
        log_level: Optional log level to use (e.g., 'DEBUG', 'INFO'). If None, uses Config.LOG_LEVEL.
        rank_suffix: Optional suffix for the log file name, used for multi-GPU processing.
    """
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create rank-specific log file if rank_suffix is provided
    if rank_suffix:
        log_file = f"{log_dir}/pipeline_{timestamp}_{rank_suffix}.log"
    else:
        log_file = f"{log_dir}/pipeline_{timestamp}.log"

    # Use provided log_level or fall back to Config.LOG_LEVEL
    level = getattr(logging, log_level if log_level is not None else Config.LOG_LEVEL)

    # Reset root logger to avoid duplicate handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    level_name = logging._levelToName.get(level, str(level))
    logger.info(f"Logging initialized with level: {level_name}" + (f" for {rank_suffix}" if rank_suffix else ""))

    return logger

def set_seed(seed=None):
    """Set random seed for reproducibility."""
    if seed is None:
        seed = Config.SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logging.info(f"Random seed set to {seed}")

def ensure_directories():
    """Ensure all necessary directories exist."""
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.TEMP_DATA_DIR, exist_ok=True)

    logging.info(f"Directories created: {Config.DATA_DIR}, {Config.TEMP_DATA_DIR}")

def log_gpu_info():
    """Log information about available GPUs."""
    if not torch.cuda.is_available():
        logging.info("CUDA is not available. Running on CPU.")
        return

    gpu_count = torch.cuda.device_count()
    logging.info(f"Found {gpu_count} GPU(s)")

    for i in range(gpu_count):
        device_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert to GB
        free_memory = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 3)
        logging.info(f"GPU {i}: {device_name}, Total Memory: {total_memory:.2f} GB, Free Memory: {free_memory:.2f} GB")

def init_distributed_mode(use_multi_gpu=None, init_distributed=None, backend=None, rank=None, world_size=None, master_addr=None, master_port=None):
    """Initialize distributed processing environment if using multiple GPUs."""
    if use_multi_gpu is None:
        use_multi_gpu = Config.USE_MULTI_GPU

    if init_distributed is None:
        init_distributed = Config.INIT_DISTRIBUTED

    if backend is None:
        backend = Config.DISTRIBUTED_BACKEND

    if master_addr is None:
        master_addr = Config.MASTER_ADDR

    if master_port is None:
        master_port = Config.MASTER_PORT

    if world_size is None:
        world_size = Config.WORLD_SIZE
        if world_size == -1:
            world_size = torch.cuda.device_count()

    # Check if distributed is already initialized
    try:
        if torch.distributed.is_initialized():
            logging.info("Distributed environment is already initialized")
            return True
    except:
        pass

    # Only initialize if multi-GPU is enabled, initialization is requested, and we have more than 1 GPU
    if use_multi_gpu and init_distributed and torch.cuda.device_count() > 1:
        try:
            logging.info("Initializing distributed environment for multi-GPU processing")

            # Set environment variables required for distributed initialization
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port

            # Determine world size and rank
            if "WORLD_SIZE" not in os.environ:
                os.environ["WORLD_SIZE"] = str(world_size)

            if rank is None and "RANK" in os.environ:
                rank = int(os.environ["RANK"])
            elif rank is None and "LOCAL_RANK" in os.environ:
                rank = int(os.environ["LOCAL_RANK"])
            else:
                rank = 0  # Default to rank 0
                os.environ["RANK"] = str(rank)

            if "LOCAL_RANK" not in os.environ:
                os.environ["LOCAL_RANK"] = str(rank)

            # Initialize the process group with specified backend
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://"
            )

            # Set the device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(rank)

            logging.info(f"Distributed environment initialized successfully: rank={rank}, world_size={world_size}")
            return True
        except Exception as e:
            logging.warning(f"Failed to initialize distributed environment: {e}")
            logging.warning("Falling back to single-GPU mode")
            return False
    elif not init_distributed and use_multi_gpu and torch.cuda.device_count() > 1:
        logging.info("Multi-GPU is enabled but distributed initialization is disabled in config")
    return False

def save_json(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info(f"Data saved to {filepath}")

def load_json(filepath):
    """Load data from a JSON file."""
    if not os.path.exists(filepath):
        logging.warning(f"File not found: {filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logging.info(f"Data loaded from {filepath}")
    return data

def calculate_quality_score(evaluation_results):
    """Calculate overall quality score from evaluation results."""
    score = 0
    for criterion, weight in Config.EVALUATION_WEIGHTS.items():
        if criterion in evaluation_results:
            score += evaluation_results[criterion] * weight

    return score

def format_example(problem, solution, reasoning, quality_score=None, metadata=None):
    """Format a single example in a standardized way."""
    example = {
        "problem": problem,
        "solution": solution,
        "reasoning": reasoning,
        "timestamp": datetime.now().isoformat()
    }

    if quality_score is not None:
        example["quality_score"] = quality_score

    if metadata is not None:
        example["metadata"] = metadata

    return example


# DeepSeek model helper functions
_model_cache = {}

def get_model_and_tokenizer(model_name=None, device=None, dtype=None, quantization=None, low_memory_mode=None, use_multi_gpu=None, device_map=None):
    """Get or load a model and tokenizer with memory optimization options and multi-GPU support."""
    import gc

    if model_name is None:
        model_name = Config.REASONING_MODEL

    if device is None:
        device = Config.MODEL_DEVICE

    if dtype is None:
        dtype = Config.MODEL_DTYPE

    if quantization is None:
        quantization = Config.MODEL_QUANTIZATION

    if low_memory_mode is None:
        # Default to True to enable memory optimizations
        low_memory_mode = Config.LOW_MEMORY_MODE

    if use_multi_gpu is None:
        use_multi_gpu = Config.USE_MULTI_GPU

    if device_map is None:
        device_map = Config.DEVICE_MAP

    # Determine the actual device map to use
    actual_device_map = device
    if use_multi_gpu and torch.cuda.device_count() > 1:
        logging.info(f"Multiple GPUs detected: {torch.cuda.device_count()} GPUs")

        # Check if distributed is initialized when using device_map="auto"
        if device_map == "auto" or isinstance(device_map, dict):
            try:
                is_distributed = torch.distributed.is_initialized()
            except:
                is_distributed = False

            if not is_distributed:
                logging.warning("Using device_map='auto' without distributed initialization may cause issues")
                logging.warning("Consider setting INIT_DISTRIBUTED=True in config or using a different device_map")

        actual_device_map = device_map

        # Log available GPU memory
        for i in range(torch.cuda.device_count()):
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            free_memory_gb = free_memory / (1024 ** 3)
            logging.info(f"GPU {i}: {free_memory_gb:.2f} GB free")
    else:
        if use_multi_gpu:
            logging.info("Multi-GPU requested but only one GPU detected or CUDA not available")

    # Create a cache key based on the parameters
    cache_key = f"{model_name}_{actual_device_map}_{dtype}_{quantization}_{low_memory_mode}_{use_multi_gpu}"

    # Check if the model is already loaded
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Clean up memory before loading a new model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info(f"Loading model {model_name} with device_map={actual_device_map}, dtype={dtype}, low_memory_mode={low_memory_mode}")

    # Set up quantization parameters if needed
    quantization_config = None
    if quantization == "4bit":
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif quantization == "8bit":
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_8bit=True
        )

    # Convert dtype string to torch dtype
    torch_dtype = getattr(torch, dtype)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Load model with memory optimizations and multi-GPU support
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=actual_device_map,  # This handles multi-GPU distribution
        quantization_config=quantization_config,
        trust_remote_code=True,
        # Memory optimization options
        low_cpu_mem_usage=low_memory_mode,
        offload_folder="offload_folder" if low_memory_mode else None
    )

    # Enable gradient checkpointing for memory efficiency if in low memory mode
    if low_memory_mode and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Log model distribution across devices
    if hasattr(model, 'hf_device_map'):
        logging.info(f"Model distribution across devices: {model.hf_device_map}")

    # Cache the model and tokenizer
    _model_cache[cache_key] = (model, tokenizer)

    return model, tokenizer


def format_prompt(prompt_template, **kwargs):
    """Format a prompt template with the given arguments."""
    # If no kwargs are provided, return the template as is (already formatted)
    if not kwargs:
        return prompt_template

    # Format the prompt template
    try:
        # Try with named parameters
        prompt = prompt_template.format(**kwargs)
    except Exception as e:
        logging.error(f"Error formatting prompt template with named parameters: {e}")
        # Try with positional parameters as fallback
        try:
            prompt = prompt_template.format(*kwargs.values())
        except Exception as e:
            logging.error(f"Error formatting prompt template with positional parameters: {e}")
            # Return the template as is if all formatting attempts fail
            return prompt_template

    # Apply the chat template if using a chat model
    if Config.USE_LOCAL_MODEL and hasattr(Config, 'CHAT_TEMPLATE'):
        # Use positional formatting with the prompt as the first argument
        try:
            prompt = Config.CHAT_TEMPLATE.format(prompt)
        except Exception as e:
            logging.error(f"Error applying chat template with format(prompt): {e}")
            # Try with explicit positional index
            try:
                prompt = Config.CHAT_TEMPLATE.format(*[prompt])
            except Exception as e:
                logging.error(f"Error applying chat template with format(*[prompt]): {e}")
                # Fall back to just using the prompt without the template

    return prompt


def generate_with_model(prompt, model=None, tokenizer=None, max_tokens=None, temperature=None, top_p=None):
    """Generate text using a local model with multi-GPU support."""
    import gc

    if model is None or tokenizer is None:
        model, tokenizer = get_model_and_tokenizer()

    if max_tokens is None:
        max_tokens = Config.MAX_TOKENS

    if temperature is None:
        temperature = Config.TEMPERATURE

    if top_p is None:
        top_p = Config.TOP_P

    try:
        # Tokenize the prompt with attention mask
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)

        # Create attention mask if not present
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        # Move to device - handle multi-GPU case
        # In multi-GPU setups with device_map="auto", the model spans multiple devices
        # We'll use the first device in the model's device map for input tensors
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            # Get the device of the first layer
            first_device = next(iter(model.hf_device_map.values()))
            if isinstance(first_device, str) and first_device.startswith('cuda'):
                device = first_device
            else:
                device = model.device if hasattr(model, 'device') else 'cuda:0'
        else:
            device = model.device if hasattr(model, 'device') else 'cuda:0'

        # Move inputs to the appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate with memory optimization
        with torch.no_grad():
            # Check if distributed is initialized before using synced_gpus
            use_synced_gpus = False
            if torch.cuda.device_count() > 1 and Config.USE_MULTI_GPU:
                try:
                    # Check if distributed is initialized
                    if torch.distributed.is_initialized():
                        use_synced_gpus = True
                except:
                    # If torch.distributed is not available or not initialized, don't use synced_gpus
                    pass

            # Use smaller chunks for generation to reduce memory usage
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for memory efficiency
                # Only use synced_gpus if distributed is properly initialized
                **({
                    "synced_gpus": True
                } if use_synced_gpus else {})
            )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the assistant's response (remove the prompt)
        if "<|im_start|>assistant" in generated_text:
            response = generated_text.split("<|im_start|>assistant", 1)[1].strip()
        else:
            # If we can't find the assistant marker, just return everything after the prompt
            response = generated_text[len(prompt):].strip()

        # Clean up memory
        del inputs, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response

    except Exception as e:
        logging.error(f"Error in generate_with_model: {e}")
        # Clean up memory even on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
