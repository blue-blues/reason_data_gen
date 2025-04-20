"""
Module for evaluating the quality of generated mathematical reasoning.
"""

import os
import logging
import json
import time
import re
from tqdm import tqdm
import torch
from config import Config
from utils import save_json, calculate_quality_score, get_model_and_tokenizer, format_prompt, generate_with_model

logger = logging.getLogger(__name__)

class SelfEvaluator:
    """Class for evaluating the quality of generated mathematical reasoning."""

    def __init__(self):
        """Initialize the self-evaluator."""
        self.model_name = Config.EVALUATION_MODEL
        self.criteria = Config.EVALUATION_CRITERIA
        self.quality_threshold = Config.QUALITY_THRESHOLD
        self.temperature = 0.3  # Lower temperature for more consistent evaluations
        self.max_tokens = 1024
        self.top_p = Config.TOP_P
        self.use_local_model = Config.USE_LOCAL_MODEL

        # Initialize model and tokenizer if using local model
        self.model = None
        self.tokenizer = None
        if self.use_local_model:
            self.model, self.tokenizer = get_model_and_tokenizer(self.model_name)

    def evaluate_reasoning(self, problem, reasoning):
        """Evaluate the quality of reasoning for a single problem."""
        import gc

        # Construct the evaluation prompt
        criteria_str = "\n".join([f"- {criterion}" for criterion in self.criteria])

        # Use a clean, properly formatted prompt without indentation issues
        evaluation_prompt = ("Evaluate the following mathematical reasoning based on these criteria:\n"
                           f"{criteria_str}\n\n"
                           "For each criterion, provide a score from 0.0 to 1.0 and a brief explanation.\n\n"
                           f"Problem: {problem}\n\n"
                           f"Reasoning:\n{reasoning}\n\n"
                           "<think>\n"
                           "Carefully analyze the reasoning step by step. For each criterion, assign a score and explain your reasoning.\n"
                           "</think>\n\n"
                           "Evaluation:")

        try:
            # Format the prompt
            formatted_prompt = format_prompt(evaluation_prompt)

            if self.use_local_model:
                # Generate using local model
                evaluation_text = generate_with_model(
                    formatted_prompt,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            else:
                # This branch is kept for compatibility but not used with DeepSeek model
                logger.error("Non-local model evaluation is not supported for DeepSeek models")
                return {
                    "evaluation_text": "Error: Non-local model evaluation is not supported",
                    "scores": {},
                    "quality_score": 0.0,
                    "needs_improvement": True
                }

            # Clean up the evaluation text - remove the <think> tags if present
            evaluation_text = evaluation_text.replace('<think>', '').replace('</think>', '')

            # Parse the evaluation text to extract scores
            evaluation_results = {}
            for criterion in self.criteria:
                # Look for lines containing the criterion name and a score
                for line in evaluation_text.split('\n'):
                    if criterion.lower() in line.lower() and ":" in line:
                        # Try to extract a score (assuming format like "criterion: 0.8")
                        try:
                            score_part = line.split(":", 1)[1].strip()
                            # Extract the first number found in the score part
                            score_match = re.search(r'(\d+\.\d+|\d+)', score_part)
                            if score_match:
                                score = float(score_match.group(1))
                                # Ensure score is between 0 and 1
                                score = max(0.0, min(1.0, score))
                                evaluation_results[criterion] = score
                                break
                        except Exception as e:
                            logger.warning(f"Error parsing score for {criterion}: {e}")

            # Calculate overall quality score
            quality_score = calculate_quality_score(evaluation_results)

            return {
                "evaluation_text": evaluation_text,
                "scores": evaluation_results,
                "quality_score": quality_score,
                "needs_improvement": quality_score < self.quality_threshold
            }

        except Exception as e:
            logger.error(f"Error evaluating reasoning: {e}")
            return {
                "evaluation_text": f"Error: {str(e)}",
                "scores": {},
                "quality_score": 0.0,
                "needs_improvement": True
            }

    def evaluate_batch(self, examples, output_path=None):
        """Evaluate a batch of examples."""
        import gc
        import torch

        results = []
        batch_size = Config.BATCH_SIZE
        cleanup = Config.CLEANUP_BETWEEN_BATCHES

        for i, example in enumerate(tqdm(examples, desc="Evaluating reasoning")):
            problem = example["problem"]
            reasoning = example["reasoning"]

            try:
                # Evaluate reasoning
                evaluation = self.evaluate_reasoning(problem, reasoning)

                # Add evaluation results to the example
                example_with_eval = example.copy()
                example_with_eval.update({
                    "evaluation": evaluation["evaluation_text"],
                    "scores": evaluation["scores"],
                    "quality_score": evaluation["quality_score"],
                    "needs_improvement": evaluation["needs_improvement"]
                })

                results.append(example_with_eval)

                # Save intermediate results periodically
                if output_path and (i + 1) % 5 == 0:
                    save_json(results, output_path)
                    logger.info(f"Saved {len(results)} evaluation results to {output_path}")

                # Clean up memory between examples if enabled
                if cleanup:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Add a small delay to allow memory cleanup
                    time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error evaluating reasoning for example {i}: {e}")
                # Continue with the next example
                continue

        # Save final results if output path is provided
        if output_path:
            save_json(results, output_path)
            logger.info(f"Evaluation results saved to {output_path}")

        return results

    def filter_examples(self, examples, threshold=None):
        """Filter examples based on quality score."""
        if threshold is None:
            threshold = self.quality_threshold

        good_examples = [ex for ex in examples if ex.get("quality_score", 0) >= threshold]
        needs_improvement = [ex for ex in examples if ex.get("quality_score", 0) < threshold]

        logger.info(f"Filtered examples: {len(good_examples)} good, {len(needs_improvement)} need improvement")

        return {
            "good_examples": good_examples,
            "needs_improvement": needs_improvement
        }


if __name__ == "__main__":
    # Test the self-evaluator
    from utils import setup_logging
    from data_downloader import DataDownloader
    from reasoning_generator import ReasoningGenerator

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

    logger.info(f"Evaluated {len(evaluated_examples)} examples")
    if evaluated_examples:
        logger.info(f"First example quality score: {evaluated_examples[0]['quality_score']}")
