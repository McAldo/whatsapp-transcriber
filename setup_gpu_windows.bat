@echo off
REM WhatsApp Transcriber - GPU Setup Helper for Windows
REM This script helps setup GPU acceleration by installing cuDNN and updating PATH

echo ========================================
echo WhatsApp Transcriber - GPU Setup
echo ========================================
echo.

echo Step 1: Checking CUDA installation...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CUDA not found! Please install CUDA Toolkit first.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

echo [OK] CUDA is installed
nvidia-smi | findstr "CUDA Version"
echo.

echo Step 2: Detecting CUDA version...
for /f "tokens=*" %%a in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_LINE=%%a
echo %CUDA_LINE% | findstr "12." >nul
if not errorlevel 1 (
    set CUDNN_PACKAGE=nvidia-cudnn-cu12
    echo [OK] Detected CUDA 12.x - will install nvidia-cudnn-cu12
) else (
    set CUDNN_PACKAGE=nvidia-cudnn-cu11
    echo [OK] Detected CUDA 11.x - will install nvidia-cudnn-cu11
)
echo.

echo Step 3: Installing cuDNN via pip...
echo Running: pip install %CUDNN_PACKAGE%
pip install %CUDNN_PACKAGE%
if errorlevel 1 (
    echo [ERROR] cuDNN installation failed!
    echo Try manually: pip install %CUDNN_PACKAGE%
    pause
    exit /b 1
)
echo [OK] cuDNN installed
echo.

echo Step 4: Finding cuDNN installation path...
for /f "delims=" %%i in ('python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))"') do set CUDNN_PATH=%%i
echo Found at: %CUDNN_PATH%
echo.

echo Step 5: Checking PATH...
echo The following path needs to be in your System PATH:
echo %CUDNN_PATH%\bin
echo.

echo Checking if already in PATH...
echo %PATH% | findstr /C:"%CUDNN_PATH%" >nul
if not errorlevel 1 (
    echo [OK] cuDNN is already in PATH!
) else (
    echo [MANUAL ACTION REQUIRED]
    echo Please add the following to your System PATH:
    echo %CUDNN_PATH%\bin
    echo.
    echo Instructions:
    echo 1. Press Windows key and search for "Environment Variables"
    echo 2. Click "Environment Variables..." button
    echo 3. Under "System variables", select "Path" and click "Edit"
    echo 4. Click "New" and paste: %CUDNN_PATH%\bin
    echo 5. Click OK on all windows
    echo 6. Restart your terminal/command prompt
    echo.
    echo Would you like to copy the path to clipboard?
    set /p COPY_CLIP="Enter Y to copy, or press Enter to skip: "
    if /i "%COPY_CLIP%"=="Y" (
        echo %CUDNN_PATH%\bin | clip
        echo [OK] Path copied to clipboard! Now follow the instructions above.
    )
)
echo.

echo Step 6: Installing GPU-enabled CTranslate2...
echo Running: pip install ctranslate2 --force-reinstall --extra-index-url https://pypi.nvidia.com
pip install ctranslate2 --force-reinstall --extra-index-url https://pypi.nvidia.com
if errorlevel 1 (
    echo [WARNING] CTranslate2 GPU installation failed, but you can try running the app anyway
) else (
    echo [OK] CTranslate2 installed
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. If you had to update PATH, restart your terminal
echo 2. Run: streamlit run app.py
echo 3. The app should show "GPU acceleration available"
echo.
echo If GPU still doesn't work, see README.md troubleshooting section
echo.
pause
