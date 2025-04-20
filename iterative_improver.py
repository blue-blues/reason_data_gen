"""
Module for iteratively improving mathematical reasoning based on evaluation feedback.
"""

import os
import logging
import json
import time
import re
from tqdm import tqdm
import torch
from config import Config
from utils import save_json, format_example, get_model_and_tokenizer, format_prompt, generate_with_model

logger = logging.getLogger(__name__)

class IterativeImprover:
    """Class for iteratively improving mathematical reasoning based on evaluation feedback."""

    def __init__(self, model=None, tokenizer=None):
        """Initialize the iterative improver.

        Args:
            model: Optional pre-loaded model to reuse (to avoid loading twice)
            tokenizer: Optional pre-loaded tokenizer to reuse
        """
        self.model_name = Config.REASONING_MODEL
        self.temperature = Config.TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS
        self.top_p = Config.TOP_P
        self.prompt_template = Config.IMPROVEMENT_PROMPT_TEMPLATE
        self.max_iterations = Config.MAX_ITERATIONS
        self.use_local_model = Config.USE_LOCAL_MODEL

        # Initialize model and tokenizer if using local model
        self.model = model
        self.tokenizer = tokenizer

        # Only load a new model if one wasn't provided and we're using a local model
        if self.use_local_model and (self.model is None or self.tokenizer is None):
            logger.info("Loading improvement model (this could be avoided by passing an existing model)")
            self.model, self.tokenizer = get_model_and_tokenizer(self.model_name)

    def generate_feedback(self, example):
        """Generate feedback based on evaluation scores."""
        feedback = ["Here's feedback on your solution:"]

        # Add specific feedback based on scores
        scores = example.get("scores", {})
        for criterion, score in scores.items():
            if score < 0.7:
                if criterion == "correctness":
                    feedback.append(f"- The solution may not be correct. Please verify your calculations and logic.")
                elif criterion == "step_by_step_clarity":
                    feedback.append(f"- The step-by-step explanation could be clearer. Break down your reasoning into more explicit steps.")
                elif criterion == "logical_coherence":
                    feedback.append(f"- The logical flow could be improved. Make sure each step follows logically from the previous one.")
                elif criterion == "mathematical_precision":
                    feedback.append(f"- Be more precise with mathematical notation and terminology.")

        # Add general feedback if no specific issues were identified
        if len(feedback) == 1:
            feedback.append("- Your solution is good, but try to make the reasoning even more explicit and detailed.")

        # Add the evaluation text if available
        if "evaluation" in example:
            feedback.append("\nDetailed evaluation:")
            feedback.append(example["evaluation"])

        return "\n".join(feedback)

    def improve_reasoning(self, example):
        """Improve reasoning for a single example based on evaluation feedback."""
        import gc
        import torch

        problem = example["problem"]
        current_solution = example["reasoning"]

        # Generate feedback
        feedback = self.generate_feedback(example)

        try:
            # Format the prompt - use a clean string concatenation approach
            formatted_prompt = format_prompt(self.prompt_template,
                                           problem=problem,
                                           current_solution=current_solution,
                                           feedback=feedback)

            if self.use_local_model:
                # Generate using local model
                improved_reasoning = generate_with_model(
                    formatted_prompt,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            else:
                # This branch is kept for compatibility but not used with DeepSeek model
                logger.error("Non-local model improvement is not supported for DeepSeek models")
                return {
                    "reasoning": current_solution,
                    "solution": example["solution"]
                }

            # Extract the final answer from the improved reasoning
            # Look for boxed answers first as recommended for DeepSeek-R1 models
            boxed_match = re.search(r'\\boxed{([^}]+)}', improved_reasoning)
            if boxed_match:
                improved_solution = boxed_match.group(0)  # Get the full \boxed{...} expression
            else:
                # Fall back to the previous heuristic
                lines = improved_reasoning.split('\n')
                improved_solution = lines[-1] if lines else improved_reasoning

                # If the last line contains "answer is" or similar, use that as the solution
                for line in reversed(lines):
                    if any(phrase in line.lower() for phrase in ["answer is", "result is", "solution is", "therefore", "thus"]):
                        improved_solution = line
                        break

            # Clean up the reasoning - remove the <think> tags if present
            improved_reasoning = improved_reasoning.replace('<think>', '').replace('</think>', '')

            # Clean up memory
            if Config.CLEANUP_BETWEEN_BATCHES:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return {
                "reasoning": improved_reasoning,
                "solution": improved_solution
            }

        except Exception as e:
            logger.error(f"Error improving reasoning: {e}")
            # Clean up memory even on error
            if Config.CLEANUP_BETWEEN_BATCHES:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return {
                "reasoning": current_solution,
                "solution": example["solution"]
            }

    def improve_batch(self, examples, evaluator, output_path=None):
        """Improve a batch of examples that need improvement."""
        import gc
        import torch

        improved_examples = []
        cleanup = Config.CLEANUP_BETWEEN_BATCHES

        for i, example in enumerate(tqdm(examples, desc="Improving reasoning")):
            current_quality = example.get("quality_score", 0)
            current_iteration = example.get("improvement_iteration", 0)

            # Skip examples that are already good enough or have reached max iterations
            if current_quality >= Config.QUALITY_THRESHOLD or current_iteration >= self.max_iterations:
                improved_examples.append(example)
                continue

            try:
                # Improve reasoning
                improved = self.improve_reasoning(example)

                # Create a new example with improved reasoning
                improved_example = format_example(
                    problem=example["problem"],
                    solution=improved["solution"],
                    reasoning=improved["reasoning"],
                    metadata={
                        "original_data": example.get("metadata", {}).get("original_data", {}),
                        "previous_version": {
                            "reasoning": example["reasoning"],
                            "solution": example["solution"],
                            "quality_score": current_quality,
                            "evaluation": example.get("evaluation", "")
                        }
                    }
                )

                # Update improvement iteration
                improved_example["improvement_iteration"] = current_iteration + 1

                try:
                    # Re-evaluate the improved reasoning
                    evaluation = evaluator.evaluate_reasoning(
                        improved_example["problem"],
                        improved_example["reasoning"]
                    )

                    # Add evaluation results to the example
                    improved_example.update({
                        "evaluation": evaluation["evaluation_text"],
                        "scores": evaluation["scores"],
                        "quality_score": evaluation["quality_score"],
                        "needs_improvement": evaluation["needs_improvement"]
                    })
                except Exception as eval_error:
                    logger.error(f"Error evaluating improved reasoning: {eval_error}")
                    # Add default evaluation results if evaluation fails
                    improved_example.update({
                        "evaluation": "Error during evaluation",
                        "scores": {},
                        "quality_score": 0.0,
                        "needs_improvement": True
                    })

                improved_examples.append(improved_example)

                # Save intermediate results periodically
                if output_path and (i + 1) % 5 == 0:
                    save_json(improved_examples, output_path)
                    logger.info(f"Saved {len(improved_examples)} improved examples to {output_path}")

                # Clean up memory between examples if enabled
                if cleanup:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Add a small delay to allow memory cleanup
                    time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error improving example {i}: {e}")
                # Add the original example if improvement fails
                improved_examples.append(example)
                continue

        # Save final results if output path is provided
        if output_path:
            save_json(improved_examples, output_path)
            logger.info(f"Improved examples saved to {output_path}")

        return improved_examples

    def run_improvement_loop(self, examples, evaluator, max_iterations=None):
        """Run the full improvement loop until all examples are good enough or max iterations is reached."""
        if max_iterations is None:
            max_iterations = self.max_iterations

        current_examples = examples

        for iteration in range(max_iterations):
            logger.info(f"Starting improvement iteration {iteration + 1}/{max_iterations}")

            # Filter examples that need improvement
            filtered = evaluator.filter_examples(current_examples)
            good_examples = filtered["good_examples"]
            needs_improvement = filtered["needs_improvement"]

            # If all examples are good enough, we're done
            if not needs_improvement:
                logger.info(f"All examples meet quality threshold after {iteration + 1} iterations")
                break

            # Improve examples that need improvement
            improved_examples = self.improve_batch(needs_improvement, evaluator)

            # Combine good examples with improved examples for the next iteration
            current_examples = good_examples + improved_examples

            logger.info(f"Completed improvement iteration {iteration + 1}: {len(good_examples)} good, {len(needs_improvement)} improved")

        return current_examples


if __name__ == "__main__":
    # Test the iterative improver
    from utils import setup_logging
    from data_downloader import DataDownloader
    from reasoning_generator import ReasoningGenerator
    from self_evaluator import SelfEvaluator

    logger = setup_logging()

    # Get some seed data
    downloader = DataDownloader()
    data = downloader.get_data(num_examples=2)

    # Generate reasoning
    generator = ReasoningGenerator()
    examples = generator.generate_batch(data)

    # Evaluate reasoning
    evaluator = SelfEvaluator()
    evaluated_examples = evaluator.evaluate_batch(examples)

    # Improve reasoning
    improver = IterativeImprover()
    improved_examples = improver.run_improvement_loop(evaluated_examples, evaluator, max_iterations=1)

    logger.info(f"Final examples: {len(improved_examples)}")
    if improved_examples:
        logger.info(f"First example quality score: {improved_examples[0]['quality_score']}")
