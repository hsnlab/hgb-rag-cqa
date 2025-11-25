#!/usr/bin/env bash
set -e

EVAL_PATH="./data/agentic_answercontext_reviewed.csv"
VENV_PATH=".venv"
QUESTION_LIMIT=10
SCRIPT="src.eval.kg_rag_eval"

MLFLOW_ARTIFACTS="$HOME/Documents/git/hgb-rag-cqa/mlflow_data/artifacts"
MLFLOW_DB="$HOME/Documents/git/hgb-rag-cqa/mlflow_data/db"
MLFLOW_PORT=5000
MLFLOW_URI="http://127.0.0.1:${MLFLOW_PORT}"
MLFLOW_EXPERIMENT="edgecases_mid_p_questions"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi
CONFIGS=(
    "rag_config_no_dedup_rerank.json"
    "rag_config_no_rerank.json"
    "rag_config_no_overretrieve.json"
    "rag_config_no_overretrieve_rerank.json"
    "rag_config_full_nocontext.json"
    "rag_config_full_context.json"
)
MODELS=(
    "mistral:7b"
    "deepseek-coder:6.7b"
    "llama3.1:8b"
    "llama3-chatqa:8b"
    "deepseek-r1:8b"
    
)
COMMON_ARGS="--eval-path ${EVAL_PATH} --question-limit ${QUESTION_LIMIT} --mlflow-uri ${MLFLOW_URI} --mlflow-exp ${MLFLOW_EXPERIMENT}"

for MODEL in "${MODELS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        echo "===== Starting evaluation: $MODEL, $CONFIG ====="
        if ! python -m "${SCRIPT}" --model-name ${MODEL} -rc ${CONFIG} ${COMMON_ARGS}; then
            echo "Evaluation failed for model=$MODEL, config=$CONFIG � continuing..."
        fi
    done
done

echo "All evals completed."
