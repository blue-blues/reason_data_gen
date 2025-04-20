"""
Module for generating mathematical reasoning using Chain-of-Thought in distributed mode.
"""

import os
import logging
import json
import time
from tqdm import tqdm
import torch
import gc
from config import Config
from utils import save_json, format_example, format_prompt
from model_utils_distributed import load_model_for_rank, generate_with_model_distributed

logger = logging.getLogger(__name__)

class ReasoningGeneratorDistributed:
    """Class for generating mathematical reasoning in distributed mode."""

    def __init__(self, rank=0):
        """Initialize the reasoning generator for a specific rank."""
        self.model_name = Config.REASONING_MODEL
        self.temperature = Config.TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS
        self.top_p = Config.TOP_P
        self.prompt_template = Config.REASONING_PROMPT_TEMPLATE
        self.rank = rank
        
        # Initialize model and tokenizer for this rank
        self.model, self.tokenizer = load_model_for_rank(
            self.model_name,
            rank=self.rank,
            dtype=Config.MODEL_DTYPE,
            quantization=Config.MODEL_QUANTIZATION,
            low_memory_mode=Config.LOW_MEMORY_MODE
        )

    def generate_reasoning(self, problem, cleanup_delay=2.0):
        """Generate reasoning for a single problem."""
        try:
            # Format the prompt
            formatted_prompt = format_prompt(self.prompt_template, problem=problem)

            # Generate using model for this rank
            reasoning = generate_with_model_distributed(
                formatted_prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                rank=self.rank
            )

            # Extract the final answer from the reasoning
            # Look for boxed answers first as recommended for DeepSeek-R1 models
            import re
            boxed_match = re.search(r'\\boxed{([^}]+)}', reasoning)
            if boxed_match:
                solution = boxed_match.group(0)  # Get the full \boxed{...} expression
            else:
                # Fall back to the previous heuristic
                lines = reasoning.split('\n')
                solution = lines[-1] if lines else reasoning

                # If the last line contains "answer is" or similar, use that as the solution
                for line in reversed(lines):
                    if any(phrase in line.lower() for phrase in ["answer is", "result is", "solution is", "therefore", "thus"]):
                        solution = line
                        break

            # Clean up the reasoning - remove the <think> tags if present
            reasoning = reasoning.replace('<think>', '').replace('</think>', '')

            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Add a delay to allow memory to be properly released
            time.sleep(cleanup_delay)

            return {
                "reasoning": reasoning,
                "solution": solution
            }

        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            # Clean up memory even on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Add a delay to allow memory to be properly released
            time.sleep(cleanup_delay)

            return {
                "reasoning": f"Error: {str(e)}",
                "solution": "Error occurred"
            }

    def generate_batch(self, problems, output_path=None, cleanup_delay=2.0, start_from=0):
        """Generate reasoning for a batch of problems."""
        results = []
        
        # Load existing results if available
        if output_path and os.path.exists(output_path) and start_from == 0:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"Loaded {len(results)} existing results from {output_path}")
                
                # If we have existing results, start from where we left off
                start_from = len(results)
            except Exception as e:
                logger.error(f"Error loading existing results: {e}")
                results = []

        # Process examples one by one
        for i, problem_data in enumerate(tqdm(problems[start_from:], desc=f"Generating reasoning (Rank {self.rank})")):
            problem = problem_data["problem"]
            
            # Log progress
            logger.info(f"Rank {self.rank}: Processing example {start_from + i + 1}/{len(problems)}")

            try:
                # Generate reasoning with enhanced memory management
                result = self.generate_reasoning(problem, cleanup_delay=cleanup_delay)

                # Format the example
                example = format_example(
                    problem=problem,
                    solution=result["solution"],
                    reasoning=result["reasoning"],
                    metadata={"original_data": problem_data}
                )

                results.append(example)

                # Save intermediate results after each example
                if output_path and (i + 1) % 5 == 0:
                    save_json(results, output_path)
                    logger.info(f"Saved {len(results)} generated examples to {output_path}")

                # Clean up memory between examples
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Add a delay to allow memory to be properly released
                time.sleep(cleanup_delay)

            except Exception as e:
                logger.error(f"Error generating reasoning for example {start_from + i}: {e}")
                
                # Save current progress even on error
                if output_path:
                    save_json(results, output_path)
                    logger.info(f"Saved {len(results)} generated examples to {output_path} after error")
                
                # Clean up memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Add a longer delay after an error
                time.sleep(cleanup_delay * 2)
                
                # Continue with the next example
                continue

        # Save final results if output path is provided
        if output_path:
            save_json(results, output_path)
            logger.info(f"Generated reasoning saved to {output_path}")

        return results
