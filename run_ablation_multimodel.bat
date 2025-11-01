@echo off
setlocal enabledelayedexpansion

:: Activate virtual environment
call .venv\Scripts\activate

:: MLflow setup
set MLFLOW_ARTIFACTS=D:/mlflow_data/artifacts
set MLFLOW_DB=D:/mlflow_data/db
set MLFLOW_PORT=5000

:: Data and script paths
set DATA_PATH=./data/stackoverflow_qna+context+time.csv
set SCRIPT=run_ablation

:: Start MLflow server (in background)
start "MLflow Server" cmd /k mlflow server ^
    --backend-store-uri sqlite:///%MLFLOW_DB%/mlflow.db ^
    --default-artifact-root file:///%MLFLOW_ARTIFACTS% ^
    --host 127.0.0.1 --port %MLFLOW_PORT%
timeout /t 10 >nul

:: Define common args
set COMMON_ARGS=--eval_data_path %DATA_PATH% --question-limit 300 --mlflow_uri http://127.0.0.1:%MLFLOW_PORT% --mlflow_experiment rag_ablation_study

:: Run ablations for each model
echo ===== Starting ablation: mistral:7b =====
python -m %SCRIPT% --llm_model mistral:7b %COMMON_ARGS%

echo ===== Starting ablation: deepseek-coder:6.7b =====
python -m %SCRIPT% --llm_model deepseek-coder:6.7b %COMMON_ARGS%

echo ===== Starting ablation: llama3-chatqa:8b =====
python -m %SCRIPT% --llm_model llama3-chatqa:8b %COMMON_ARGS%

echo ===== Starting ablation: llama3.1:8b =====
python -m %SCRIPT% --llm_model llama3.1:8b %COMMON_ARGS%

echo All ablation studies completed.
endlocal