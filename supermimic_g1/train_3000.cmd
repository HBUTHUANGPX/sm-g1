@echo off
setlocal
if /i "%~1"=="newton" goto run
if /i "%~1"=="ovphysx" goto run
echo Usage: train_3000.cmd newton ^| ovphysx
exit /b 2

:run
pushd "%~dp0"
if not defined ISAACLAB_PYTHON set "ISAACLAB_PYTHON=%~dp0..\IsaacLab-3.0.0\.venv\Scripts\python.exe"
if not exist "%ISAACLAB_PYTHON%" set "ISAACLAB_PYTHON=%~dp0..\IsaacLab\.venv\Scripts\python.exe"
if exist "%~dp0..\g1-deps" set "PYTHONPATH=%~dp0..\g1-deps;%PYTHONPATH%"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
if not exist logs mkdir logs
if exist "logs\%~1-3000\last.pt" (
  echo Output exists. Use the explicit resume command or choose a new run directory.
  exit /b 3
)
"%ISAACLAB_PYTHON%" supermimic\train.py --backend %~1 --device cuda:0 --viz none --iterations 3000 --num_envs 64 --buffer_steps 256 --warmup_steps 64 --wm_batches 2 --wm_batch 32 --pi_batch 32 --horizon 8 --wm_only_iters 100 --critic_epochs 1 --ckpt_every 500 --log_every 25 --lr_decay_iters 3000 --log_dir logs/%~1-3000 > "logs\%~1-3000.stdout.log" 2>&1
set "rc=%ERRORLEVEL%"
> "logs\%~1-3000.exit" echo %rc%
popd
exit /b %rc%
