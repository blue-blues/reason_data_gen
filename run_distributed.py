"""
Script to run the synthetic data pipeline with distributed processing across multiple GPUs.
This version fixes the 'DTensor' object has no attribute 'CB' error.
"""

import os
import logging
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial

# Suppress warnings before importing other modules
from suppress_warnings import suppress_all_warnings
suppress_all_warnings()

from config import Config
from utils import setup_logging, set_seed, ensure_directories, log_gpu_info
from data_downloader import DataDownloader
from reasoning_generator_distributed import ReasoningGeneratorDistributed
from self_evaluator import SelfEvaluator
from iterative_improver import IterativeImprover
from data_uploader import DataUploader
from dataset_utils import get_dataset_info

# Global logger
logger = None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the synthetic data pipeline with distributed processing (fixed version)")

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
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")

    # Distributed processing arguments
    parser.add_argument("--world-size", type=int, default=None,
                        help="Number of GPUs to use (-1 for all available)")
    parser.add_argument("--master-addr", type=str, default=None,
                        help="Master address for distributed processing")
    parser.add_argument("--master-port", type=str, default=None,
                        help="Master port for distributed processing")
    parser.add_argument("--node-rank", type=int, default=0,
                        help="Rank of the current node in multi-node setup")
    parser.add_argument("--num-nodes", type=int, default=1,
                        help="Total number of nodes in multi-node setup")

    # Optimization arguments
    parser.add_argument("--optimize-gpu", action="store_true",
                        help="Apply GPU optimizations")
    parser.add_argument("--no-warnings", action="store_true",
                        help="Suppress warning messages")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    # Low memory mode arguments
    parser.add_argument("--low-memory", action="store_true",
                        help="Enable low memory mode (process one example at a time)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum number of tokens for generation")
    parser.add_argument("--cleanup-delay", type=float, default=None,
                        help="Delay between examples for memory cleanup in seconds")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start processing from this example index")

    # Informational arguments
    parser.add_argument("--list-datasets", action="store_true",
                        help="List available datasets and exit")
    parser.add_argument("--dataset-info", type=str,
                        help="Show information about a specific dataset and exit")

    return parser.parse_args()

def list_available_datasets():
    """List available datasets and their descriptions."""
    logger.info("Available datasets:")
    for key, description in get_dataset_info().items():
        logger.info(f"  - {key}: {description}")

def show_dataset_info(dataset_key):
    """Show detailed information about a specific dataset."""
    info = get_dataset_info(dataset_key)
    logger.info(f"Dataset information for '{dataset_key}':")
    for key, value in info.items():
        logger.info(f"  - {key}: {value}")

def setup_distributed(rank, world_size, master_addr=None, master_port=None):
    """
    Initialize the distributed environment.
    
    Args:
        rank: The rank of the current process
        world_size: The total number of processes
        master_addr: The address of the master process
        master_port: The port of the master process
    """
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = master_addr or Config.MASTER_ADDR
    os.environ["MASTER_PORT"] = master_port or Config.MASTER_PORT
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    
    # Initialize the process group
    dist.init_process_group(
        backend=Config.DISTRIBUTED_BACKEND,
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    logger.info(f"Initialized process {rank}/{world_size} on GPU {rank}")

def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def split_data_for_rank(data, rank, world_size):
    """
    Split the data for the current rank.
    
    Args:
        data: The full dataset
        rank: The rank of the current process
        world_size: The total number of processes
        
    Returns:
        The subset of data for this rank
    """
    # Calculate the number of examples per process
    examples_per_process = len(data) // world_size
    remainder = len(data) % world_size
    
    # Calculate start and end indices for this rank
    start_idx = rank * examples_per_process + min(rank, remainder)
    end_idx = start_idx + examples_per_process + (1 if rank < remainder else 0)
    
    # Get the subset of data for this rank
    rank_data = data[start_idx:end_idx]
    
    logger.info(f"Rank {rank}: Processing {len(rank_data)} examples from index {start_idx} to {end_idx-1}")
    
    return rank_data

def run_pipeline_on_rank(rank, world_size, args):
    """
    Run the pipeline on a specific rank.
    
    Args:
        rank: The rank of the current process
        world_size: The total number of processes
        args: The command line arguments
    """
    # Set up logging for this rank
    global logger
    log_level = "DEBUG" if args.debug else Config.LOG_LEVEL
    logger = setup_logging(log_level, f"rank_{rank}")
    
    # Set up the distributed environment
    setup_distributed(rank, world_size, args.master_addr, args.master_port)
    
    # Set random seed
    seed = args.seed or Config.SEED
    set_seed(seed + rank)  # Add rank to seed for diversity
    
    # Apply GPU optimizations if requested
    if args.optimize_gpu:
        logger.info("Applying GPU optimizations")
        from optimize_gpu import optimize_torch_settings, optimize_memory_usage
        optimize_torch_settings()
        optimize_memory_usage()
    
    # Log GPU information
    log_gpu_info()
    
    # Initialize components
    downloader = DataDownloader()
    
    # Set up output path
    output_path = args.output or Config.OUTPUT_DATA_PATH
    rank_output_path = f"{output_path}.rank_{rank}"
    
    # Set batch size if specified
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
        logger.info(f"Using batch size: {Config.BATCH_SIZE}")
    
    # Set max tokens if specified
    if args.max_tokens is not None:
        Config.MAX_TOKENS = args.max_tokens
        logger.info(f"Using max tokens: {Config.MAX_TOKENS}")
    
    try:
        # Download or load data (only on rank 0)
        if rank == 0:
            logger.info("Step 1: Loading dataset")
            all_data = downloader.get_data(
                force_download=args.force_download,
                dataset_key=args.dataset,
                dataset_name=args.dataset_name,
                config_name=args.config,
                split=args.split,
                num_examples=args.num_examples
            )
            logger.info(f"Loaded {len(all_data)} examples")
        else:
            all_data = None
        
        # Broadcast data from rank 0 to all processes
        if world_size > 1:
            # Wait for all processes to reach this point
            dist.barrier()
            
            # Broadcast data from rank 0
            if rank == 0:
                # Convert data to tensor for broadcasting
                import pickle
                import zlib
                data_bytes = zlib.compress(pickle.dumps(all_data))
                data_size = torch.tensor([len(data_bytes)], dtype=torch.long, device=f"cuda:{rank}")
                dist.broadcast(data_size, 0)
                data_tensor = torch.ByteTensor(list(data_bytes)).to(f"cuda:{rank}")
                dist.broadcast(data_tensor, 0)
            else:
                # Receive data size
                data_size = torch.tensor([0], dtype=torch.long, device=f"cuda:{rank}")
                dist.broadcast(data_size, 0)
                # Receive data
                data_tensor = torch.ByteTensor(data_size.item()).to(f"cuda:{rank}")
                dist.broadcast(data_tensor, 0)
                # Convert back to Python object
                import pickle
                import zlib
                data_bytes = bytes(data_tensor.cpu().numpy())
                all_data = pickle.loads(zlib.decompress(data_bytes))
            
            logger.info(f"Rank {rank}: Received {len(all_data)} examples from rank 0")
        
        # Split data for this rank
        data = split_data_for_rank(all_data, rank, world_size)
        
        # Apply start_from if specified
        if args.start_from > 0:
            start_idx = args.start_from
            if start_idx < len(data):
                data = data[start_idx:]
                logger.info(f"Rank {rank}: Starting from example {start_idx}, {len(data)} examples remaining")
            else:
                logger.error(f"Rank {rank}: Start index {start_idx} is greater than the number of examples {len(data)}")
                return
        
        # Initialize the distributed reasoning generator
        generator = ReasoningGeneratorDistributed(rank=rank)
        
        # Generate reasoning
        logger.info(f"Rank {rank}: Step 2: Generating reasoning")
        cleanup_delay = args.cleanup_delay or 2.0
        examples_with_reasoning = generator.generate_batch(
            data, 
            output_path=f"{Config.TEMP_DATA_DIR}/temp_reasoning_rank_{rank}.json",
            cleanup_delay=cleanup_delay,
            start_from=0
        )
        
        logger.info(f"Rank {rank}: Generated reasoning for {len(examples_with_reasoning)} examples")
        
        # Initialize evaluator and improver
        evaluator = SelfEvaluator()
        improver = IterativeImprover()
        
        # Evaluate reasoning
        logger.info(f"Rank {rank}: Step 3: Evaluating reasoning")
        evaluated_examples = evaluator.evaluate_batch(
            examples_with_reasoning,
            output_path=f"{Config.TEMP_DATA_DIR}/temp_evaluated_rank_{rank}.json"
        )
        logger.info(f"Rank {rank}: Evaluated {len(evaluated_examples)} examples")
        
        # Iteratively improve reasoning
        logger.info(f"Rank {rank}: Step 4: Iteratively improving reasoning")
        max_iterations = args.max_iterations or Config.MAX_ITERATIONS
        improved_examples = improver.run_improvement_loop(
            evaluated_examples,
            evaluator,
            max_iterations=max_iterations,
            output_path=f"{Config.TEMP_DATA_DIR}/temp_improved_rank_{rank}.json"
        )
        
        # Save results for this rank
        logger.info(f"Rank {rank}: Step 5: Saving results")
        from data_uploader import DataUploader
        uploader = DataUploader()
        uploader.save_to_local(improved_examples, rank_output_path)
        
        # Print statistics for this rank
        filtered = evaluator.filter_examples(improved_examples)
        good_examples = filtered["good_examples"]
        needs_improvement = filtered["needs_improvement"]
        
        logger.info(f"Rank {rank}: Pipeline completed successfully")
        logger.info(f"Rank {rank}: Final examples: {len(improved_examples)} total, {len(good_examples)} good, {len(needs_improvement)} need improvement")
        
        # Wait for all processes to complete
        dist.barrier()
        
        # Combine results from all ranks (only on rank 0)
        if rank == 0:
            logger.info("Combining results from all ranks")
            combined_results = []
            
            # Load results from each rank
            for r in range(world_size):
                rank_path = f"{output_path}.rank_{r}"
                if os.path.exists(rank_path):
                    from utils import load_json
                    rank_results = load_json(rank_path)
                    if rank_results:
                        combined_results.extend(rank_results)
                        logger.info(f"Loaded {len(rank_results)} examples from rank {r}")
            
            # Save combined results
            uploader.save_to_local(combined_results, output_path)
            csv_path = uploader.save_to_csv(combined_results)
            
            # Print final statistics
            logger.info(f"Combined results: {len(combined_results)} examples")
            logger.info(f"Data saved to {output_path} and {csv_path}")
            
            # Clean up rank-specific files
            for r in range(world_size):
                rank_path = f"{output_path}.rank_{r}"
                if os.path.exists(rank_path):
                    os.remove(rank_path)
                    logger.info(f"Removed temporary file: {rank_path}")
    
    except Exception as e:
        logger.error(f"Rank {rank}: Error in pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Clean up the distributed environment
        cleanup_distributed()

def run_distributed(args):
    """
    Run the pipeline in distributed mode.
    
    Args:
        args: The command line arguments
    """
    # Determine world size
    world_size = args.world_size or Config.WORLD_SIZE
    if world_size == -1:
        world_size = torch.cuda.device_count()
    
    # Ensure world size is valid
    if world_size <= 0:
        logger.error(f"Invalid world size: {world_size}")
        return
    
    if world_size > torch.cuda.device_count():
        logger.warning(f"Requested {world_size} GPUs but only {torch.cuda.device_count()} available")
        world_size = torch.cuda.device_count()
    
    logger.info(f"Running distributed pipeline with {world_size} GPUs")
    
    # Start processes
    if world_size == 1:
        # Single GPU mode
        run_pipeline_on_rank(0, 1, args)
    else:
        # Multi-GPU mode
        mp.spawn(
            run_pipeline_on_rank,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )

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
    
    # Ensure directories exist
    ensure_directories()
    
    # Log GPU information
    log_gpu_info()
    
    # Run the distributed pipeline
    run_distributed(args)
    
    logger.info("Distributed pipeline completed")

if __name__ == "__main__":
    main()
