"""
Utility functions for loading models in distributed mode.
"""

import os
import torch
import logging
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import Config

# Set up logging
logger = logging.getLogger(__name__)

def load_model_for_rank(model_name=None, rank=0, dtype=None, quantization=None, low_memory_mode=None):
    """
    Load a model for a specific rank in distributed mode.
    
    Args:
        model_name: The name or path of the model to load
        rank: The rank of the current process
        dtype: The data type to use
        quantization: The quantization method to use
        low_memory_mode: Whether to enable memory optimizations
        
    Returns:
        The loaded model and tokenizer
    """
    if model_name is None:
        model_name = Config.REASONING_MODEL
        
    if dtype is None:
        dtype = Config.MODEL_DTYPE
        
    if quantization is None:
        quantization = Config.MODEL_QUANTIZATION
        
    if low_memory_mode is None:
        low_memory_mode = Config.LOW_MEMORY_MODE
    
    # Clean up memory before loading a new model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set the device for this rank
    device = f"cuda:{rank}"
    logger.info(f"Loading model {model_name} on device {device}")
    
    # Convert dtype string to torch dtype
    torch_dtype = getattr(torch, dtype)
    
    # Set up quantization parameters if needed
    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Set up max memory
    max_memory = None
    if hasattr(Config, 'MAX_GPU_MEMORY') and Config.MAX_GPU_MEMORY:
        max_memory = {rank: Config.MAX_GPU_MEMORY, "cpu": "32GiB"}
        logger.info(f"Setting max memory: {max_memory}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device,  # Explicitly use the device for this rank
        "trust_remote_code": True,
        "low_cpu_mem_usage": low_memory_mode,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        
    if max_memory:
        model_kwargs["max_memory"] = max_memory
        
    # Enable CPU offloading if configured
    if hasattr(Config, 'CPU_OFFLOAD') and Config.CPU_OFFLOAD:
        logger.info("Enabling CPU offloading")
        model_kwargs["offload_folder"] = f"offload_rank_{rank}"
        os.makedirs(f"offload_rank_{rank}", exist_ok=True)
    
    # Load the model
    try:
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Enable gradient checkpointing for memory efficiency if in low memory mode
        if low_memory_mode and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            
        logger.info("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def generate_with_model_distributed(prompt, model, tokenizer, max_tokens=None, temperature=None, top_p=None, rank=0):
    """
    Generate text using a model in distributed mode.
    
    Args:
        prompt: The prompt to generate from
        model: The model to use
        tokenizer: The tokenizer to use
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p for sampling
        rank: The rank of the current process
        
    Returns:
        The generated text
    """
    if max_tokens is None:
        max_tokens = Config.MAX_TOKENS
        
    if temperature is None:
        temperature = Config.TEMPERATURE
        
    if top_p is None:
        top_p = Config.TOP_P
    
    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Create attention mask if not present
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        # Move inputs to the device for this rank
        device = f"cuda:{rank}"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with memory optimization
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache for memory efficiency
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
        logger.error(f"Error in generate_with_model_distributed: {e}")
        # Clean up memory even on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
