"""
Module for generating mathematical reasoning using Chain-of-Thought.
"""

import os
import logging
import json
import time
from tqdm import tqdm
import torch
from config import Config
from utils import save_json, format_example, get_model_and_tokenizer, format_prompt, generate_with_model

logger = logging.getLogger(__name__)

class ReasoningGenerator:
    """Class for generating mathematical reasoning using Chain-of-Thought."""

    def __init__(self):
        """Initialize the reasoning generator."""
        self.model_name = Config.REASONING_MODEL
        self.temperature = Config.TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS
        self.top_p = Config.TOP_P
        self.prompt_template = Config.REASONING_PROMPT_TEMPLATE
        self.use_local_model = Config.USE_LOCAL_MODEL

        # Initialize model and tokenizer if using local model
        self.model = None
        self.tokenizer = None
        if self.use_local_model:
            self.model, self.tokenizer = get_model_and_tokenizer(self.model_name)

    def generate_reasoning(self, problem):
        """Generate reasoning for a single problem."""
        import gc
        import torch

        try:
            # Format the prompt
            formatted_prompt = format_prompt(self.prompt_template, problem=problem)

            if self.use_local_model:
                # Generate using local model
                reasoning = generate_with_model(
                    formatted_prompt,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            else:
                # This branch is kept for compatibility but not used with DeepSeek model
                logger.error("Non-local model generation is not supported for DeepSeek models")
                return {
                    "reasoning": "Error: Non-local model generation is not supported",
                    "solution": "Error occurred"
                }

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
            if Config.CLEANUP_BETWEEN_BATCHES:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return {
                "reasoning": reasoning,
                "solution": solution
            }

        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            # Clean up memory even on error
            if Config.CLEANUP_BETWEEN_BATCHES:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return {
                "reasoning": f"Error: {str(e)}",
                "solution": "Error occurred"
            }

    def generate_batch(self, problems, output_path=None):
        """Generate reasoning for a batch of problems."""
        import gc
        import torch

        results = []
        cleanup = Config.CLEANUP_BETWEEN_BATCHES

        for i, problem_data in enumerate(tqdm(problems, desc="Generating reasoning")):
            problem = problem_data["problem"]

            try:
                # Generate reasoning
                result = self.generate_reasoning(problem)

                # Format the example
                example = format_example(
                    problem=problem,
                    solution=result["solution"],
                    reasoning=result["reasoning"],
                    metadata={"original_data": problem_data}
                )

                results.append(example)

                # Save intermediate results periodically
                if output_path and (i + 1) % 5 == 0:
                    save_json(results, output_path)
                    logger.info(f"Saved {len(results)} generated examples to {output_path}")

                # Clean up memory between examples if enabled
                if cleanup:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Add a small delay to allow memory cleanup
                    time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error generating reasoning for example {i}: {e}")
                # Continue with the next example
                continue

        # Save final results if output path is provided
        if output_path:
            save_json(results, output_path)
            logger.info(f"Generated reasoning saved to {output_path}")

        return results

    # This method is now obsolete as we're using the generate_reasoning method with local models
    # Kept for backward compatibility
    def generate_with_local_model(self, problem):
        """Generate reasoning using a local model."""
        logger.info("Using generate_reasoning method instead of generate_with_local_model")
        return self.generate_reasoning(problem)


if __name__ == "__main__":
    # Test the reasoning generator
    from utils import setup_logging
    from data_downloader import DataDownloader

    logger = setup_logging()

    # Get some seed data
    downloader = DataDownloader()
    data = downloader.get_data(num_examples=2)

    # Generate reasoning
    generator = ReasoningGenerator()
    results = generator.generate_batch(data)

    logger.info(f"Generated reasoning for {len(results)} examples")
    if results:
        logger.info(f"First example reasoning: {results[0]['reasoning']}")
