"""
Configuration settings for the synthetic data pipeline.
"""

class Config:
    # General settings
    SEED = 42
    DEBUG = False
    LOG_LEVEL = "INFO"

    # Data settings
    DATA_DIR = "./data"
    SEED_DATA_PATH = f"{DATA_DIR}/seed_data.json"
    OUTPUT_DATA_PATH = f"{DATA_DIR}/final_data.json"
    TEMP_DATA_DIR = f"{DATA_DIR}/temp"
    CACHE_DIR = f"{DATA_DIR}/cache"  # Cache directory for Hugging Face datasets

    # Hugging Face dataset settings
    DEFAULT_DATASET = "gsm8k"  # Default dataset to use
    DEFAULT_DATASET_CONFIG = "main"  # Default dataset configuration
    DEFAULT_SPLIT = "train"  # Default split to use
    USE_AUTH_TOKEN = False  # Whether to use HF_API_KEY for dataset access

    # Dataset configurations
    DATASET_CONFIGS = {
        "gsm8k": {
            "name": "gsm8k",
            "config": "main",
            "split": "train",
            "problem_field": "question",
            "answer_field": "answer",
            "description": "Grade School Math 8K - Elementary math word problems"
        },
        "math_qa": {
            "name": "math_qa",
            "config": None,
            "split": "train",
            "problem_field": "Problem",
            "answer_field": "Rationale",
            "description": "MathQA - Mathematics question answering dataset"
        },
        "math": {
            "name": "hendrycks/math",
            "config": "algebra",  # Can be: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus
            "split": "train",
            "problem_field": "problem",
            "answer_field": "solution",
            "description": "MATH dataset - Challenging math problems"
        },
        "asdiv": {
            "name": "mawic/asdiv",
            "config": None,
            "split": "train",
            "problem_field": "body",
            "answer_field": "answer",
            "description": "ASDiv - Arithmetic word problems with diverse structures"
        },
        "aqua": {
            "name": "allenai/aqua_rat",
            "config": None,
            "split": "train",
            "problem_field": "question",
            "answer_field": "rationale",
            "description": "AQuA - Algebra word problems with rationales"
        },
        "numina_math": {
            "name": "AI-MO/NuminaMath-1.5",
            "config": None,
            "split": "train",
            "problem_field": "problem",
            "answer_field": "solution",
            "description": "NuminaMath - High-quality competition-level math problems with detailed solutions"
        }
    }

    # Model settings
    REASONING_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # DeepSeek model for reasoning
    EVALUATION_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Same model for evaluation
    USE_LOCAL_MODEL = True  # Set to True to use local Hugging Face model instead of API
    MODEL_DEVICE = "cuda:0"  # Explicitly use the first GPU (MIG instance)
    MODEL_DTYPE = "float16"  # Using float16 instead of bfloat16 for better compatibility with 8-bit quantization
    MODEL_QUANTIZATION = "8bit"  # Use 8-bit quantization to reduce memory usage and increase speed

    # Multi-GPU settings
    USE_MULTI_GPU = True  # Enable multi-GPU support
    DEVICE_MAP = "auto"  # Let the model decide how to distribute across GPUs
    DISTRIBUTED_BACKEND = "nccl"  # Backend for distributed training (nccl for NVIDIA GPUs)
    INIT_DISTRIBUTED = True  # Initialize distributed environment for multi-GPU processing
    WORLD_SIZE = -1  # Number of GPUs to use (-1 means use all available)
    MASTER_ADDR = "localhost"  # Master address for distributed processing
    MASTER_PORT = "12355"  # Master port for distributed processing

    # Memory optimization settings
    LOW_MEMORY_MODE = True  # Enable memory optimizations
    CLEANUP_BETWEEN_BATCHES = True  # Clean up memory between batches
    MAX_GPU_MEMORY = "18GiB"  # Limit GPU memory usage to avoid OOM errors with MIG
    CPU_OFFLOAD = True  # Enable CPU offloading for layers that don't fit in GPU memory

    # Pipeline settings
    MAX_ITERATIONS = 3  # Maximum number of improvement iterations
    BATCH_SIZE = 4  # Increased batch size to better utilize GPU
    NUM_EXAMPLES = -1  # Process all available examples (-1 means all)
    PARALLEL_PROCESSING = True  # Enable parallel processing where possible

    # Reasoning generation settings
    TEMPERATURE = 0.6  # Recommended temperature for DeepSeek-R1 models
    MAX_TOKENS = 8196  # Reduced to save memory
    TOP_P = 0.95
    # Using a simple string with named parameter for maximum compatibility
    REASONING_PROMPT_TEMPLATE = "Solve the following mathematical problem step by step. Please reason step by step, and put your final answer within \\boxed{{}}. \n\nProblem: {problem}\n\n<think>"

    # Self-evaluation settings
    EVALUATION_CRITERIA = [
        "correctness",
        "step_by_step_clarity",
        "logical_coherence",
        "mathematical_precision"
    ]
    EVALUATION_WEIGHTS = {
        "correctness": 0.4,
        "step_by_step_clarity": 0.2,
        "logical_coherence": 0.2,
        "mathematical_precision": 0.2
    }
    QUALITY_THRESHOLD = 0.8  # Minimum quality score to accept

    # Improvement settings
    # Using a simple string with named parameters for maximum compatibility
    IMPROVEMENT_PROMPT_TEMPLATE = "Here is a mathematical problem and a solution with reasoning:\n\nProblem: {problem}\n\nCurrent solution:\n{current_solution}\n\nFeedback on the solution:\n{feedback}\n\nPlease provide an improved solution with clearer reasoning. Reason step by step, and put your final answer within \\boxed{{}}.\n\n<think>"

    # API settings
    OPENAI_API_KEY = "YOUR_API_KEY_HERE"  # Only needed if not using local model
    HF_API_KEY = "YOUR_API_KEY_HERE"  # For downloading models from Hugging Face

    # DeepSeek model specific settings
    # Using {0} for positional formatting to ensure compatibility
    CHAT_TEMPLATE = """<|im_start|>user
{0}<|im_end|>
<|im_start|>assistant
"""  # Chat template for DeepSeek models
    SYSTEM_PROMPT = ""  # DeepSeek-R1 recommends not using system prompts

    # Tracking settings
    USE_WANDB = False
    WANDB_PROJECT = "math-reasoning-data"
