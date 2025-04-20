"""
Utility functions for working with Hugging Face datasets.
"""

import os
import logging
import json
import random
from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, DatasetDict
from config import Config

logger = logging.getLogger(__name__)

def get_dataset_config(dataset_key: str) -> Dict[str, Any]:
    """
    Get the configuration for a specific dataset.

    Args:
        dataset_key: The key of the dataset in Config.DATASET_CONFIGS

    Returns:
        The dataset configuration dictionary
    """
    if dataset_key in Config.DATASET_CONFIGS:
        return Config.DATASET_CONFIGS[dataset_key]
    else:
        logger.warning(f"Dataset key '{dataset_key}' not found in configurations. Using default.")
        return Config.DATASET_CONFIGS[Config.DEFAULT_DATASET]

def load_hf_dataset(
    dataset_key: str = None,
    dataset_name: str = None,
    config_name: str = None,
    split: str = None,
    cache_dir: str = None,
    use_auth_token: bool = None,
    num_examples: int = None
) -> Dataset:
    """
    Load a dataset from Hugging Face.

    Args:
        dataset_key: The key of the dataset in Config.DATASET_CONFIGS
        dataset_name: The name of the dataset on Hugging Face (overrides dataset_key)
        config_name: The configuration name (overrides dataset_key)
        split: The split to load (overrides dataset_key)
        cache_dir: Directory to cache the dataset
        use_auth_token: Whether to use the Hugging Face API token
        num_examples: Number of examples to load (None for all)

    Returns:
        The loaded dataset
    """
    # Get configuration
    if dataset_key is not None:
        config = get_dataset_config(dataset_key)
    else:
        config = {
            "name": dataset_name or Config.DEFAULT_DATASET,
            "config": config_name or Config.DEFAULT_DATASET_CONFIG,
            "split": split or Config.DEFAULT_SPLIT
        }

    # Override with explicit parameters if provided
    if dataset_name is not None:
        config["name"] = dataset_name
    if config_name is not None:
        config["config"] = config_name
    if split is not None:
        config["split"] = split

    # Set up cache directory
    if cache_dir is None:
        cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Set up authentication
    if use_auth_token is None:
        use_auth_token = Config.USE_AUTH_TOKEN

    token = Config.HF_API_KEY if use_auth_token else None

    logger.info(f"Loading dataset: {config['name']}, config: {config['config']}, split: {config['split']}")

    try:
        # Load the dataset
        dataset = load_dataset(
            config["name"],
            config["config"],
            split=config["split"],
            cache_dir=cache_dir,
            token=token
        )

        # Limit the number of examples if specified
        if num_examples is not None and num_examples != -1 and num_examples < len(dataset):
            dataset = dataset.select(range(num_examples))

        logger.info(f"Loaded {len(dataset)} examples from {config['name']}")
        return dataset

    except Exception as e:
        logger.error(f"Error loading dataset {config['name']}: {e}")
        return None

def explore_dataset(dataset: Dataset, num_examples: int = 3) -> None:
    """
    Explore a dataset by printing its features and some examples.

    Args:
        dataset: The dataset to explore
        num_examples: Number of examples to print
    """
    if dataset is None:
        logger.error("Cannot explore None dataset")
        return

    logger.info(f"Dataset features: {dataset.features}")
    logger.info(f"Dataset size: {len(dataset)} examples")

    # Print some examples
    logger.info(f"Sample examples:")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        logger.info(f"Example {i}:")
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 100:
                logger.info(f"  {key}: {value[:100]}...")
            else:
                logger.info(f"  {key}: {value}")

def extract_problems_from_dataset(
    dataset: Dataset,
    dataset_key: str = None,
    problem_field: str = None,
    answer_field: str = None,
    num_examples: int = None,
    shuffle: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract problems from a dataset.

    Args:
        dataset: The dataset to extract problems from
        dataset_key: The key of the dataset in Config.DATASET_CONFIGS
        problem_field: The field containing the problem (overrides dataset_key)
        answer_field: The field containing the answer (overrides dataset_key)
        num_examples: Number of examples to extract (None for all)
        shuffle: Whether to shuffle the dataset before extraction

    Returns:
        A list of problem dictionaries
    """
    if dataset is None:
        logger.error("Cannot extract problems from None dataset")
        return []

    # Get field names from configuration if not provided
    if dataset_key is not None and (problem_field is None or answer_field is None):
        config = get_dataset_config(dataset_key)
        problem_field = problem_field or config.get("problem_field")
        answer_field = answer_field or config.get("answer_field")

    # Default field names if still not set
    problem_field = problem_field or "question"
    answer_field = answer_field or "answer"

    # Check if the fields exist in the dataset
    features = list(dataset.features.keys())
    if problem_field not in features:
        logger.warning(f"Problem field '{problem_field}' not found in dataset. Available fields: {features}")
        return []

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(seed=Config.SEED)

    # Limit the number of examples if specified
    if num_examples is not None and num_examples != -1 and num_examples < len(dataset):
        dataset = dataset.select(range(num_examples))

    # Extract problems
    problems = []
    for item in dataset:
        problem = item[problem_field]

        # Create a problem dictionary
        problem_dict = {
            "problem": problem,
            "original_data": item
        }

        # Add the answer if available
        if answer_field in item:
            problem_dict["reference_answer"] = item[answer_field]

        problems.append(problem_dict)

    logger.info(f"Extracted {len(problems)} problems from dataset")
    return problems

def get_available_datasets() -> List[str]:
    """
    Get a list of available dataset keys.

    Returns:
        A list of dataset keys
    """
    return list(Config.DATASET_CONFIGS.keys())

def get_dataset_info(dataset_key: str = None) -> Dict[str, Any]:
    """
    Get information about a dataset.

    Args:
        dataset_key: The key of the dataset in Config.DATASET_CONFIGS

    Returns:
        A dictionary with dataset information
    """
    if dataset_key is None:
        # Return info for all datasets
        return {key: config.get("description", "No description available")
                for key, config in Config.DATASET_CONFIGS.items()}

    if dataset_key in Config.DATASET_CONFIGS:
        config = Config.DATASET_CONFIGS[dataset_key]
        return {
            "name": config.get("name"),
            "config": config.get("config"),
            "description": config.get("description", "No description available"),
            "problem_field": config.get("problem_field"),
            "answer_field": config.get("answer_field")
        }
    else:
        logger.warning(f"Dataset key '{dataset_key}' not found in configurations.")
        return {"error": f"Dataset key '{dataset_key}' not found"}

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Print available datasets
    logger.info(f"Available datasets: {get_available_datasets()}")

    # Load and explore a dataset
    dataset = load_hf_dataset("gsm8k", num_examples=5)
    if dataset:
        explore_dataset(dataset)

        # Extract problems
        problems = extract_problems_from_dataset(dataset, "gsm8k")
        logger.info(f"First problem: {problems[0]['problem']}")
