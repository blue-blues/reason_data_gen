@echo off
echo ===================================================
echo Mathematical Reasoning Pipeline
echo ===================================================
echo.
echo Available options:
echo 1. Run optimized pipeline (NuminaMath dataset) - RECOMMENDED
echo    * Fastest and most efficient option
echo    * Uses GPU optimizations and warning suppression
echo    * Processes the NuminaMath dataset
echo.
echo 2. Run with GPU monitoring (NuminaMath dataset)
echo    * Same as option 1 but with GPU usage monitoring
echo    * Useful for performance analysis
echo.
echo 3. Run all datasets (may take a VERY long time)
echo    * Processes all available datasets sequentially
echo    * Can take many hours to complete
echo.
echo 4. Exit
echo.

:menu
set /p choice=Enter your choice (1-4):

if "%choice%"=="1" goto run_optimized
if "%choice%"=="2" goto run_monitoring
if "%choice%"=="3" goto run_all
if "%choice%"=="4" goto end

echo Invalid choice. Please try again.
goto menu

:run_optimized
echo.
echo Running optimized pipeline with NuminaMath dataset...
echo.

REM Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
if %ERRORLEVEL% NEQ 0 (
    echo Error checking CUDA availability
    goto error
)

REM Run the optimized pipeline with warning suppression
python run_optimized_dataset_pipeline.py --dataset numina_math --force-download --batch-size 4 --num-examples -1 --optimize-gpu --no-warnings
if %ERRORLEVEL% NEQ 0 (
    echo Error running optimized pipeline
    goto error
)

echo Pipeline completed successfully!
goto end

:run_monitoring
echo.
echo Running pipeline with GPU monitoring...
echo.

REM Start GPU monitoring in a separate window
start cmd /k "python monitor_gpu.py --interval 2 --output gpu_usage_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.csv"

REM Wait a moment for monitoring to start
timeout /t 2 > nul

REM Run the optimized pipeline with warning suppression
python run_optimized_dataset_pipeline.py --dataset numina_math --force-download --batch-size 4 --num-examples -1 --optimize-gpu --no-warnings

REM Display completion message
echo Pipeline execution completed.
echo GPU monitoring is still running in the other window.
echo You can close it when you're done analyzing the results.
goto end

:run_all
echo.
echo Running pipeline for all available datasets...
echo This may take a long time to complete.
echo.

echo Processing GSM8K dataset...
python run_dataset.py --dataset gsm8k --force-download
if %ERRORLEVEL% NEQ 0 (
    echo Error processing GSM8K dataset
    goto error
)

echo Processing MathQA dataset...
python run_dataset.py --dataset math_qa --force-download
if %ERRORLEVEL% NEQ 0 (
    echo Error processing MathQA dataset
    goto error
)

echo Processing MATH dataset...
python run_dataset.py --dataset math --force-download
if %ERRORLEVEL% NEQ 0 (
    echo Error processing MATH dataset
    goto error
)

echo Processing ASDiv dataset...
python run_dataset.py --dataset asdiv --force-download
if %ERRORLEVEL% NEQ 0 (
    echo Error processing ASDiv dataset
    goto error
)

echo Processing AQuA dataset...
python run_dataset.py --dataset aqua --force-download
if %ERRORLEVEL% NEQ 0 (
    echo Error processing AQuA dataset
    goto error
)

echo Processing Maths-College dataset...
python run_dataset.py --dataset maths_college --force-download
if %ERRORLEVEL% NEQ 0 (
    echo Error processing Maths-College dataset
    goto error
)

echo Processing NuminaMath dataset...
python run_dataset.py --dataset numina_math --force-download
if %ERRORLEVEL% NEQ 0 (
    echo Error processing NuminaMath dataset
    goto error
)

echo All datasets processed successfully!
goto end

:error
echo Pipeline encountered an error.

:end
echo.
echo Press any key to exit...
pause > nul
