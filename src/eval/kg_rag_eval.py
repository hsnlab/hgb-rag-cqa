import os, html, re, httpx, sys, traceback, time, mlflow
import pandas as pd
from ast import literal_eval
from transformers import pipeline
from huggingface_hub import login
from typing import Tuple

from src.utils.config_loader import load_all_configs
from src.utils.qdrant_store import QdrantStore
from src.utils.retrieval import KnowledgeGraphRetriever
from src.utils.deduplicator import Deduplicator
from src.utils.reranker import Reranker
from src.rag.repo_rag import RepositoryRAG
from src.rag.config import PipelineConfig
from src.eval.evaluation import RAGEvaluator
from langchain_ollama import ChatOllama

def remove_html_tags(text):
  """
  Removes HTML tags from a string and unescapes HTML entities.

  Args:
    text: The input string containing HTML.

  Returns:
    The cleaned string without HTML tags or entities.
  """
  tag_re = re.compile('<[^>]+>')
  
  no_tags = tag_re.sub('', text)
  cleaned_text = html.unescape(no_tags)
  
  return cleaned_text.strip()

def build_llm(llm_model: str):
    """
    Build an Ollama-based LLM client for generation.
    Supports models served by the local Ollama daemon, e.g. mistral:7b-instruct.
    """
    if not llm_model:
        raise ValueError("llm_model must be provided (e.g., mistral:7b-instruct)")

    os.environ["OLLAMA_KEEP_ALIVE"] = "0"
    llm = ChatOllama(
        model=llm_model, 
        base_url="http://localhost:11434", 
        async_client_kwargs={
            "headers": {"Connection": "close"},
            "timeout": 120,
            "limits": httpx.Limits(
                max_keepalive_connections=0, 
                max_connections=10,           
            ),
        },
    )
    return llm

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", required=True, help="Path to eval dataset CSV")
    parser.add_argument(
        "--model-name", "-ml",
        default="mistral:7b",
        type=str,
        help="Name of Ollama model to use for agents.",
        required=False
    )
    parser.add_argument(
        "--question-limit", "-ql",
        default=None,
        type=int,
        help="Limit the evaluation dataset to this number of questions",
        required=False
    )
    parser.add_argument(
        "--rag-config", "-rc",
        default="rag_config.json",
        type=str,
        help="Name of the rag config file in the './_/' directory (default: rag_config.json)",
        required=False
    )
    parser.add_argument(
        "--mlflow-uri", "-mu", 
        default="http://127.0.0.1:5000", 
        type=str,
        required=False
    )
    parser.add_argument(
        "--mlflow-exp", "-me", 
        default="rag_eval_experiments", 
        type=str,
        required=False
    )

    args = parser.parse_args()
    eval_df_path = args.eval_path
    q_limit = args.question_limit
    model_name = args.model_name
    rag_config_name = args.rag_config
    mlflow_uri = args.mlflow_uri
    mlflow_exp = args.mlflow_exp

    # --- Load dataset
    if not os.path.isfile(eval_df_path):
        print(f"Error: File '{eval_df_path}' does not exist.")
        sys.exit(1)
        
    print(f"Loading dataset from: {eval_df_path}")
    try:
        dataset = pd.read_csv(eval_df_path)
    except:
        dataset = pd.read_csv(eval_df_path, sep="\t")
    dataset = dataset.rename(columns={"LLM_questions": "question", "LLM_answers": "answer", "filtered_classes": "class_context", "filtered_combined_names": "function_context",
                                      "questions":"question", "answers":"answer", "contexts":"function_context","answer_contexts":"function_context","edit_functions":"function_context"})
    dataset = dataset.dropna(subset=["question", "answer", "function_context"])
    dataset["function_context"] = dataset["function_context"].apply(literal_eval)
    if "class_context" in dataset.columns.tolist():
        dataset["class_context"] = dataset["class_context"].apply(literal_eval)    
    
    dataset = dataset.loc[(dataset["function_context"].str.len() > 0)]

    if q_limit:
        dataset = dataset.iloc[:min(q_limit,len(dataset))]
    dataset["question"] = dataset["question"].apply(remove_html_tags)
    dataset["answer"] = dataset["answer"].apply(remove_html_tags)

    # --- Load configs
    qdrant_cfg, neo4j_cfg, _, rag_config = load_all_configs(rag_config_name=rag_config_name)

    with open("./_/hf_token.txt") as f:
        hf_token = f.read().strip()
    login(hf_token)
    
    # --- Build components
    vectorstore = QdrantStore(
        model_name=qdrant_cfg["model_name"],
        qdrant_url=qdrant_cfg["url"],
        collection_name=qdrant_cfg["collection"],
        api_key=qdrant_cfg["api_key"],
        distance_type=qdrant_cfg["distance"],
        kwargs = qdrant_cfg["kwargs"]
    )
    kg_retriever = KnowledgeGraphRetriever(
        vector_store=vectorstore,
        neo4j_url=neo4j_cfg["url"],
        neo4j_username=neo4j_cfg["user"],
        neo4j_password=neo4j_cfg["password"],
        database="neo4j"
    )
    deduplicator = Deduplicator(embedder=vectorstore.embeddings)
    reranker = Reranker(neo4j_config=neo4j_cfg)

    gen = build_llm(model_name)

    rag = RepositoryRAG(
        vectorstore,
        retriever=kg_retriever,
        deduplicator=deduplicator,
        reranker=reranker,
        llm=gen,
        neo4j_uri=neo4j_cfg["url"],
        neo4j_auth=(neo4j_cfg["user"], neo4j_cfg["password"]),
        classifier_model = "facebook/bart-large-mnli"
    )

    # MLflow setup
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_exp)

    evaluator = RAGEvaluator(dataset, rag, k_values=[3, 5, 10],class_context_column="class_context",function_context_column="function_context")
    try:
        config_base = os.path.splitext(os.path.basename(rag_config_name))[0]
        run_name = f"{config_base}_{model_name.replace(':', '-')}"
        evaluator.evaluate(rag_config, run_name=run_name, verbose=True)
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    main()