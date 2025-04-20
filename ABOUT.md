# About Mathematical Reasoning Data Pipeline

[![GitHub stars](https://img.shields.io/github/stars/blue-blues/reason_data_gen?style=social)](https://github.com/blue-blues/reason_data_gen/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/blue-blues/reason_data_gen?style=social)](https://github.com/blue-blues/reason_data_gen/network/members)
[![GitHub issues](https://img.shields.io/github/issues/blue-blues/reason_data_gen)](https://github.com/blue-blues/reason_data_gen/issues)
[![GitHub license](https://img.shields.io/github/license/blue-blues/reason_data_gen)](https://github.com/blue-blues/reason_data_gen/blob/master/LICENSE)

**Website**: [Project Documentation](https://github.com/blue-blues/reason_data_gen/wiki)

**Topics**: mathematical-reasoning, chain-of-thought, data-generation, language-models, machine-learning, distributed-processing, pytorch, huggingface

## Project Purpose

The Mathematical Reasoning Data Pipeline is designed to address the challenge of generating high-quality mathematical reasoning data for training and fine-tuning language models. While large language models (LLMs) have shown impressive capabilities across many domains, mathematical reasoning remains a challenging area that requires specialized training data with detailed step-by-step reasoning.

This project provides a complete solution for:

1. **Generating high-quality mathematical reasoning data** that includes detailed step-by-step solutions
2. **Improving the quality of reasoning** through iterative self-evaluation and refinement
3. **Scaling the data generation process** across multiple GPUs for efficiency
4. **Optimizing memory usage** to handle large datasets like NuminaMath-1.5

## Technical Approach

The pipeline implements several key technical approaches:

### Chain-of-Thought (CoT) Generation
- Uses advanced prompting techniques to generate detailed step-by-step reasoning
- Structures the reasoning in a way that models can learn from and replicate

### Self-Improvement Loop
- Evaluates the quality of generated reasoning using predefined criteria
- Identifies areas for improvement in the reasoning
- Iteratively refines the reasoning based on evaluation feedback

### Distributed Processing
- Implements PyTorch Distributed Data Parallel (DDP) for multi-GPU processing
- Efficiently distributes workload across available GPUs
- Includes synchronization mechanisms to ensure consistent results

### Memory Optimization
- Implements gradient checkpointing and other memory-saving techniques
- Provides low-memory processing modes for large datasets
- Includes GPU memory monitoring and optimization

## Model Support

The pipeline is designed to work with various language models, with a preference for:
- **DeepSeek-R1-Distill-Qwen-7B**: Optimized for mathematical reasoning tasks
- Other models can be configured in the `config.py` file

## Dataset Support

The pipeline supports multiple mathematical datasets, including:
- **NuminaMath-1.5**: A comprehensive dataset with 896,215 examples of high-quality mathematical problems
- **GSM8K**: Grade School Math problems
- **MathQA**: Mathematics question answering dataset
- **MATH**: Challenging math problems across various domains
- And several others listed in the main README

## Use Cases

This pipeline is particularly useful for:
1. **Researchers** working on improving mathematical reasoning capabilities in language models
2. **ML Engineers** preparing training data for fine-tuning specialized mathematical models
3. **Educators** interested in generating step-by-step solutions for educational content
4. **Data Scientists** exploring how to improve reasoning capabilities through better training data

## Future Directions

Planned enhancements include:
- Support for more diverse mathematical datasets
- Integration with additional language models
- Enhanced evaluation metrics for reasoning quality
- Improved parallelization and distributed processing
- Web interface for easier interaction with the pipeline
