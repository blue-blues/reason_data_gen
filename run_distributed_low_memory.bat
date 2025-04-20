@echo off
echo ===================================================
echo Mathematical Reasoning Pipeline - Distributed Low Memory Mode
echo ===================================================
echo.
echo This script runs the pipeline in low memory mode using all available GPUs.
echo.

REM Check CUDA availability and count GPUs
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Number of GPUs:', torch.cuda.device_count())"
if %ERRORLEVEL% NEQ 0 (
    echo Error checking CUDA availability
    goto error
)

REM Run the distributed pipeline in low memory mode
python run_distributed.py --dataset numina_math --force-download --batch-size 1 --max-tokens 4096 --cleanup-delay 2.0 --optimize-gpu --no-warnings --low-memory
if %ERRORLEVEL% NEQ 0 (
    echo Error running distributed pipeline in low memory mode
    goto error
)

echo Pipeline completed successfully!
goto end

:error
echo Pipeline encountered an error.

:end
echo.
echo Press any key to exit...
pause > nul
