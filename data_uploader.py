"""
Module for uploading and storing the final high-quality data.
"""

import os
import logging
import json
import pandas as pd
from datetime import datetime
from config import Config
from utils import save_json

logger = logging.getLogger(__name__)

class DataUploader:
    """Class for uploading and storing the final high-quality data."""
    
    def __init__(self):
        """Initialize the data uploader."""
        self.output_data_path = Config.OUTPUT_DATA_PATH
    
    def prepare_data_for_upload(self, examples):
        """Prepare data for upload by cleaning and formatting."""
        prepared_data = []
        
        for example in examples:
            # Create a clean version of the example with only the necessary fields
            clean_example = {
                "problem": example["problem"],
                "solution": example["solution"],
                "reasoning": example["reasoning"],
                "quality_score": example.get("quality_score", 0.0),
                "timestamp": example.get("timestamp", datetime.now().isoformat())
            }
            
            # Add metadata if available
            if "metadata" in example:
                clean_example["metadata"] = example["metadata"]
            
            prepared_data.append(clean_example)
        
        return prepared_data
    
    def save_to_local(self, examples, output_path=None):
        """Save data to a local file."""
        if output_path is None:
            output_path = self.output_data_path
        
        # Prepare data
        prepared_data = self.prepare_data_for_upload(examples)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        save_json(prepared_data, output_path)
        logger.info(f"Saved {len(prepared_data)} examples to {output_path}")
        
        return output_path
    
    def save_to_csv(self, examples, output_path=None):
        """Save data to a CSV file."""
        if output_path is None:
            output_path = self.output_data_path.replace(".json", ".csv")
        
        # Prepare data
        prepared_data = self.prepare_data_for_upload(examples)
        
        # Convert to DataFrame
        df = pd.DataFrame(prepared_data)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(prepared_data)} examples to {output_path}")
        
        return output_path
    
    def upload_to_huggingface(self, examples, dataset_name, token=None):
        """Upload data to Hugging Face (placeholder for future implementation)."""
        # This is a placeholder for uploading to Hugging Face
        # Implementation would depend on the specific requirements
        logger.warning("Upload to Hugging Face not implemented yet")
        
        # Mock implementation
        logger.info(f"Would upload {len(examples)} examples to Hugging Face dataset: {dataset_name}")
        
        return f"https://huggingface.co/datasets/{dataset_name}"
    
    def upload_to_s3(self, examples, bucket_name, key_prefix=None):
        """Upload data to AWS S3 (placeholder for future implementation)."""
        # This is a placeholder for uploading to S3
        # Implementation would depend on the specific requirements
        logger.warning("Upload to S3 not implemented yet")
        
        # Mock implementation
        if key_prefix is None:
            key_prefix = f"math_reasoning_data/{datetime.now().strftime('%Y%m%d')}"
        
        logger.info(f"Would upload {len(examples)} examples to S3 bucket: {bucket_name}/{key_prefix}")
        
        return f"s3://{bucket_name}/{key_prefix}/data.json"


if __name__ == "__main__":
    # Test the data uploader
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
    
    # Upload data
    uploader = DataUploader()
    output_path = uploader.save_to_local(examples)
    
    logger.info(f"Data saved to {output_path}")
