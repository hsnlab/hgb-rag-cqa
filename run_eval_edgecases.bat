@echo off
setlocal enabledelayedexpansion

:: Activate virtual environment
call .venv\Scripts\activate

:: MLflow setup
set MLFLOW_ARTIFACTS=D:/mlflow_data/artifacts
set MLFLOW_DB=D:/mlflow_data/db
set MLFLOW_PORT=5000

:: Data and script paths
set DATA_PATH=./data/eval_easy.csv
set SCRIPT=src.eval.kg_rag_eval

:: Start MLflow server
start "MLflow Server" cmd /k mlflow server --backend-store-uri sqlite:///%MLFLOW_DB%/mlflow.db --default-artifact-root file:///%MLFLOW_ARTIFACTS% --host 127.0.0.1 --port %MLFLOW_PORT%
timeout /t 8 >nul

:: Models to run
set MODELS_TO_RUN=^
 "mistral:7b"

set CONFIGS=^
 "rag_config_no_dedup.json" ^
 "rag_config_no_rerank.json" ^
 "rag_config_no_rerankgraph.json" ^
 "rag_config_no_rerankpop.json" ^
 "rag_config_no_overretrieve.json" ^
 "rag_config_full_context.json"
echo Starting all evaluations...

:: Loop through each model in the MODELS_TO_RUN list
set COMMON_ARGS=--eval-path %DATA_PATH% -ql 50 --mlflow-uri http://127.0.0.1:%MLFLOW_PORT% --mlflow-exp "ablation_studies"
:: Loop through each model in the MODELS_TO_RUN list
for %%M in (%MODELS_TO_RUN%) do (
    echo.
    echo ======================================================
    echo ===== Starting ablation for model: %%~M =====
    echo ======================================================
    
    :: NESTED LOOP: Loop through each config for the current model
    for %%C in (%CONFIGS%) do (
        echo.
        echo --- Running config %%~C for model %%~M ---
        python -m %SCRIPT% --model-name %%~M %COMMON_ARGS% --rag-config %%~C
    )
    
    echo.
    echo ===== Finished ablation for model: %%~M =====
    echo ======================================================
)

echo.
echo ======================================================
echo ALL EVALUATIONS COMPLETED.
echo ======================================================

endlocal