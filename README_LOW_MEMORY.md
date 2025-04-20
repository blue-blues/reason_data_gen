# Low Memory Mode for Mathematical Reasoning Pipeline

## Overview

This document explains how to use the low memory mode for processing large datasets like NuminaMath with the DeepSeek-R1-Distill-Qwen-7B model.

## Why Use Low Memory Mode?

The standard pipeline can get stuck when processing large datasets with complex models due to:
- Memory leaks in the model generation process
- Insufficient memory cleanup between examples
- Batch processing that consumes too much memory

Low memory mode addresses these issues by:
1. Processing examples one at a time (batch size = 1)
2. Reducing the maximum token length for generation
3. Adding longer delays between examples for better memory cleanup
4. Enabling all memory optimization features
5. Supporting resuming from a specific example if the process is interrupted

## How to Use

### Windows

#### Option 1: Using the Batch File (Recommended)

Simply run:
```
run_low_memory.bat
```

This will run the pipeline with optimal low memory settings.

#### Option 2: Command Line

```
python run_optimized_dataset_pipeline_low_memory.py --dataset numina_math --force-download --batch-size 1 --max-tokens 4096 --cleanup-delay 2.0 --optimize-gpu --no-warnings
```

### Linux

#### Option 1: Using the Shell Script (Recommended)

First, make the script executable:
```
chmod +x run_low_memory.sh
```
Or use the helper script:
```
./make_scripts_executable.sh
```

Then run:
```
./run_low_memory.sh
```

This will run the pipeline with optimal low memory settings.

#### Option 2: Command Line

```
python run_optimized_dataset_pipeline_low_memory.py --dataset numina_math --force-download --batch-size 1 --max-tokens 4096 --cleanup-delay 2.0 --optimize-gpu --no-warnings
```

## Additional Parameters

The low memory script supports all the parameters of the standard script, plus:

- `--max-tokens`: Maximum number of tokens for generation (default: 4096)
- `--cleanup-delay`: Delay between examples for memory cleanup in seconds (default: 2.0)
- `--start-from`: Start processing from this example index (useful for resuming)

## Resuming Interrupted Processing

If the process is interrupted, you can resume from where it left off:

1. Check how many examples were processed by looking at the temporary output file:
   ```
   ./data/temp/temp_reasoning_output.json
   ```

2. Count the number of examples in this file (N)

3. Resume processing from that point:
   ```
   python run_optimized_dataset_pipeline_low_memory.py --dataset numina_math --start-from N --optimize-gpu --no-warnings
   ```

## Performance Considerations

Low memory mode is significantly slower than the standard mode but much more stable for large datasets. Expect processing to take longer, but with fewer memory-related errors.

## Troubleshooting

If you still encounter memory issues:
1. Try reducing `--max-tokens` further (e.g., to 2048)
2. Increase `--cleanup-delay` (e.g., to 5.0)
3. Make sure no other memory-intensive applications are running
4. Restart your system to clear any lingering memory issues
