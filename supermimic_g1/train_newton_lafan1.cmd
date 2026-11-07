@echo off
setlocal
pushd "%~dp0"
if not defined ISAACLAB_PYTHON set "ISAACLAB_PYTHON=%~dp0..\IsaacLab-3.0.0\.venv\Scripts\python.exe"
if not exist "%ISAACLAB_PYTHON%" set "ISAACLAB_PYTHON=%~dp0..\IsaacLab\.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "RUN=logs\newton-lafan1-4096"
if not exist logs mkdir logs
if exist "%RUN%\train_log.csv" (
  echo Training already exists at %RUN%.
  popd
  exit /b 3
)
"%ISAACLAB_PYTHON%" supermimic\train.py --backend newton --device cuda:0 --viz none --iterations 20000 --num_envs 4096 --buffer_steps 512 --warmup_steps 64 --wm_batches 8 --wm_batch 1024 --pi_batch 1024 --horizon 32 --wm_only_iters 500 --critic_epochs 5 --ckpt_every 500 --log_every 25 --lr_decay_iters 250000 --log_dir "%RUN%" > "%RUN%.stdout.log" 2>&1
set "rc=%ERRORLEVEL%"
> "%RUN%.exit" echo %rc%
popd
exit /b %rc%
