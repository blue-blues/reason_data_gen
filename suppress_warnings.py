"""
Script to suppress unnecessary warnings.
"""

import os
import logging
import warnings

def suppress_tensorflow_warnings():
    """Suppress TensorFlow warnings."""
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

    # Suppress TensorFlow warnings
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        pass

def suppress_pytorch_warnings():
    """Suppress PyTorch warnings."""
    # Filter out specific PyTorch warnings
    warnings.filterwarnings("ignore", message=".*MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16.*")
    warnings.filterwarnings("ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*")
    warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint: please pass in use_reentrant=.*")

    # Suppress CUDA registration warnings
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Suppress CUDA factory registration warnings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Suppress JAX/XLA warnings
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

def suppress_huggingface_warnings():
    """Suppress Hugging Face warnings."""
    # Filter out specific Hugging Face warnings
    warnings.filterwarnings("ignore", message=".*The attention mask and the pad token id were not set.*")
    warnings.filterwarnings("ignore", message=".*You have modified the pretrained model configuration to control generation.*")
    warnings.filterwarnings("ignore", message=".*You are using a model of type .* to generate text, but the generation config.*")

    # Set transformers logging level
    try:
        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except ImportError:
        pass

def suppress_all_warnings():
    """Suppress all unnecessary warnings."""
    # Set Python logging level
    logging.getLogger().setLevel(logging.ERROR)

    # Suppress specific module warnings
    suppress_tensorflow_warnings()
    suppress_pytorch_warnings()
    suppress_huggingface_warnings()

    # Filter out other common warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    suppress_all_warnings()
    print("Warnings suppressed successfully.")
