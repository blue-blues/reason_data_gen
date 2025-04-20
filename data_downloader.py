"""
Module for downloading or generating seed data for the pipeline.
"""

import os
import logging
import random
import json
from typing import Dict, List, Any
from datasets import Dataset
from config import Config
from utils import ensure_directories, save_json
from dataset_utils import load_hf_dataset, extract_problems_from_dataset, get_dataset_config

logger = logging.getLogger(__name__)

class DataDownloader:
    """Class for downloading or generating seed data."""

    def __init__(self):
        """Initialize the data downloader."""
        ensure_directories()
        self.seed_data_path = Config.SEED_DATA_PATH

    def download_from_huggingface(self, dataset_key=None, dataset_name=None, config_name=None, split=None, subset=None, num_examples=None):
        """Download data from Hugging Face datasets."""
        # Handle the case when num_examples is None or -1 for fallback to synthetic data
        if num_examples is None or num_examples == -1:
            # Use a reasonable default for synthetic data if needed
            fallback_num_examples = 100
        else:
            fallback_num_examples = num_examples

        try:
            # Load the dataset using the utility function
            dataset = load_hf_dataset(
                dataset_key=dataset_key,
                dataset_name=dataset_name,
                config_name=config_name,
                split=split,
                num_examples=num_examples
            )

            if dataset is None:
                logger.error(f"Failed to load dataset")
                return self.generate_synthetic_seed_data(fallback_num_examples)

            # Apply subset filtering if needed
            if subset:
                dataset = dataset.filter(lambda x: x.get("type") == subset)

            # Extract problems using the utility function
            problems = extract_problems_from_dataset(
                dataset=dataset,
                dataset_key=dataset_key,
                num_examples=num_examples
            )

            if not problems:
                logger.error(f"No problems extracted from dataset")
                return self.generate_synthetic_seed_data(fallback_num_examples)

            # Save the data
            save_json(problems, self.seed_data_path)
            logger.info(f"Downloaded and processed {len(problems)} examples")

            return problems

        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return self.generate_synthetic_seed_data(fallback_num_examples)

    def generate_synthetic_seed_data(self, num_examples=100):
        """Generate synthetic seed data if download fails."""
        # If num_examples is None or -1, use a default value
        if num_examples is None or num_examples == -1:
            num_examples = 100

        logger.info(f"Generating {num_examples} synthetic seed examples")

        # Example templates for mathematical problems
        templates = [
            "What is {a} + {b}?",
            "Calculate {a} - {b}.",
            "Multiply {a} × {b}.",
            "If {a} is divided by {b}, what is the result?",
            "What is {a}% of {b}?",
            "Solve for x: {a}x + {b} = {c}",
            "If a rectangle has length {a} and width {b}, what is its area?",
            "A train travels at {a} km/h for {b} hours. How far does it travel?",
            "If {a} workers can complete a job in {b} days, how many days would it take {c} workers?",
            "What is the average of {a}, {b}, and {c}?"
        ]

        data = []
        for _ in range(num_examples):
            template = random.choice(templates)

            # Generate random values
            values = {
                'a': random.randint(1, 100),
                'b': random.randint(1, 100),
                'c': random.randint(1, 100)
            }

            # Format the problem
            problem = template.format(**values)

            data.append({
                "problem": problem,
                "values": values,
                "template": template
            })

        # Save the data
        save_json(data, self.seed_data_path)
        logger.info(f"Generated {len(data)} synthetic seed examples")

        return data

    def load_existing_data(self):
        """Load existing seed data if available."""
        if os.path.exists(self.seed_data_path):
            logger.info(f"Loading existing seed data from {self.seed_data_path}")
            with open(self.seed_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        else:
            logger.warning(f"No existing seed data found at {self.seed_data_path}")
            return None

    def get_data(self, force_download=False, dataset_key=None, dataset_name=None, config_name=None, split=None, subset=None, num_examples=None):
        """Get seed data, either by loading existing data or downloading/generating new data."""
        if not force_download:
            existing_data = self.load_existing_data()
            if existing_data:
                return existing_data

        # If no existing data or force_download is True
        if dataset_key or dataset_name:
            return self.download_from_huggingface(
                dataset_key=dataset_key or Config.DEFAULT_DATASET,
                dataset_name=dataset_name,
                config_name=config_name,
                split=split,
                subset=subset,
                num_examples=num_examples
            )
        else:
            # Handle the case when num_examples is None or -1
            if num_examples is None or num_examples == -1:
                fallback_num_examples = 100
            else:
                fallback_num_examples = num_examples
            return self.generate_synthetic_seed_data(fallback_num_examples)


if __name__ == "__main__":
    # Test the data downloader
    from utils import setup_logging
    logger = setup_logging()

    downloader = DataDownloader()
    data = downloader.get_data(num_examples=10)

    logger.info(f"Downloaded {len(data)} examples")
    if data:
        logger.info(f"First example: {data[0]}")
