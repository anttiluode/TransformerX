@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create .venv and install dependencies using the README first.
  exit /b 1
)
".venv\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: install a supported CUDA PyTorch wheel'"
if errorlevel 1 exit /b 1
if exist "runs\language-gpu\latest.pt" (
  echo Language checkpoint exists. Use the README resume command or choose another output folder.
  exit /b 1
)
if exist "runs\vision-gpu\latest.pt" (
  echo Vision checkpoint exists. Use the README resume command or choose another output folder.
  exit /b 1
)
".venv\Scripts\python.exe" -m transformerx.run --config configs/language_gpu.json
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m transformerx.run --config configs/vision_gpu.json
if errorlevel 1 exit /b 1
echo Both runs completed. Results are in runs\language-gpu and runs\vision-gpu.
