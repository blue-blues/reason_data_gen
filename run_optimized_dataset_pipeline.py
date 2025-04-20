"""
Script to run the synthetic data pipeline with optimized settings and warning suppression.
"""

import os
import logging
import argparse
import torch

# Suppress warnings before importing other modules
from suppress_warnings import suppress_all_warnings
suppress_all_warnings()

from config import Config
from utils import setup_logging, set_seed, ensure_directories, log_gpu_info, init_distributed_mode
from data_downloader import DataDownloader
from reasoning_generator import ReasoningGenerator
from self_evaluator import SelfEvaluator
from iterative_improver import IterativeImprover
from data_uploader import DataUploader
from dataset_utils import get_dataset_info
from model_utils import load_optimized_model, optimize_inference_settings
from optimize_gpu import optimize_torch_settings, optimize_memory_usage

# Global logger
logger = None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the synthetic data pipeline with optimized settings")

    # Dataset selection arguments
    parser.add_argument("--dataset", type=str, default=None,
                        help=f"Dataset key to use. Available options: {', '.join(get_dataset_info().keys())}")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Custom dataset name (if not using a predefined dataset key)")
    parser.add_argument("--config", type=str, default=None,
                        help="Dataset configuration name")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split to use")

    # Pipeline control arguments
    parser.add_argument("--num-examples", type=int, default=-1,
                        help="Number of examples to process (-1 for all)")
    parser.add_argument("--force-download", action="store_true",
                        help="Force download of dataset even if cached data exists")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Maximum number of improvement iterations")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for processing")

    # Output control arguments
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List available datasets and exit")
    parser.add_argument("--dataset-info", type=str, default=None,
                        help="Show information about a specific dataset and exit")
    
    # GPU optimization arguments
    parser.add_argument("--no-warnings", action="store_true",
                        help="Suppress all warnings")
    parser.add_argument("--optimize-gpu", action="store_true",
                        help="Apply GPU optimizations")

    return parser.parse_args()

def list_available_datasets():
    """List all available datasets with descriptions."""
    datasets = get_dataset_info()
    logger.info("Available datasets:")
    for key, description in datasets.items():
        logger.info(f"  {key}: {description}")

def show_dataset_info(dataset_key):
    """Show detailed information about a specific dataset."""
    info = get_dataset_info(dataset_key)
    if "error" in info:
        logger.error(info["error"])
        return

    logger.info(f"Dataset: {dataset_key}")
    logger.info(f"  Name: {info['name']}")
    logger.info(f"  Config: {info['config']}")
    logger.info(f"  Description: {info['description']}")
    logger.info(f"  Problem field: {info['problem_field']}")
    logger.info(f"  Answer field: {info['answer_field']}")

def run_pipeline(args):
    """Run the full pipeline with the specified arguments."""
    # Initialize components
    downloader = DataDownloader()
    generator = ReasoningGenerator()
    evaluator = SelfEvaluator()
    improver = IterativeImprover()
    uploader = DataUploader()

    # Set up output path
    output_path = args.output or Config.OUTPUT_DATA_PATH
    
    # Set batch size if specified
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
        logger.info(f"Using batch size: {Config.BATCH_SIZE}")

    # Download or load data
    logger.info("Step 1: Loading dataset")
    data = downloader.get_data(
        force_download=args.force_download,
        dataset_key=args.dataset,
        dataset_name=args.dataset_name,
        config_name=args.config,
        split=args.split,
        num_examples=args.num_examples
    )
    logger.info(f"Loaded {len(data)} examples")

    # Generate reasoning
    logger.info("Step 2: Generating reasoning")
    examples_with_reasoning = generator.generate_batch(data)
    logger.info(f"Generated reasoning for {len(examples_with_reasoning)} examples")

    # Evaluate reasoning
    logger.info("Step 3: Evaluating reasoning")
    evaluated_examples = evaluator.evaluate_batch(examples_with_reasoning)
    logger.info(f"Evaluated {len(evaluated_examples)} examples")

    # Iteratively improve reasoning
    logger.info("Step 4: Iteratively improving reasoning")
    max_iterations = args.max_iterations or Config.MAX_ITERATIONS
    improved_examples = improver.run_improvement_loop(
        evaluated_examples,
        evaluator,
        max_iterations=max_iterations
    )

    # Upload final data
    logger.info("Step 5: Saving final data")
    uploader.save_to_local(improved_examples, output_path)
    csv_path = uploader.save_to_csv(improved_examples)

    # Print final statistics
    filtered = evaluator.filter_examples(improved_examples)
    good_examples = filtered["good_examples"]
    needs_improvement = filtered["needs_improvement"]

    logger.info(f"Pipeline completed successfully")
    logger.info(f"Final examples: {len(improved_examples)} total, {len(good_examples)} good, {len(needs_improvement)} need improvement")
    logger.info(f"Data saved to {output_path} and {csv_path}")

    return {
        "output_path": output_path,
        "csv_path": csv_path,
        "total_examples": len(improved_examples),
        "good_examples": len(good_examples),
        "needs_improvement": len(needs_improvement)
    }

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Suppress warnings if requested
    if args.no_warnings:
        suppress_all_warnings()

    # Set up logging
    log_level = "DEBUG" if args.debug else Config.LOG_LEVEL
    global logger
    logger = setup_logging(log_level)

    # Handle informational commands
    if args.list_datasets:
        list_available_datasets()
        return

    if args.dataset_info:
        show_dataset_info(args.dataset_info)
        return

    # Update config if needed
    if args.seed is not None:
        Config.SEED = args.seed
        set_seed(args.seed)

    # Ensure directories exist
    ensure_directories()

    # Apply GPU optimizations if requested
    if args.optimize_gpu:
        logger.info("Applying GPU optimizations")
        optimize_torch_settings()
        optimize_memory_usage()

    # Log GPU information
    log_gpu_info()

    # Initialize distributed environment for multi-GPU processing if enabled
    distributed_initialized = False
    if Config.USE_MULTI_GPU and Config.INIT_DISTRIBUTED:
        distributed_initialized = init_distributed_mode()
        
        # Check if multi-GPU is enabled
        if distributed_initialized:
            logger.info(f"Multi-GPU processing is enabled with device_map={Config.DEVICE_MAP} and distributed environment initialized")
        else:
            logger.info(f"Multi-GPU processing is enabled with device_map={Config.DEVICE_MAP} but distributed environment could not be initialized")

    # Run the pipeline
    result = run_pipeline(args)
    logger.info(f"Pipeline completed with result: {result}")

if __name__ == "__main__":
    main()
