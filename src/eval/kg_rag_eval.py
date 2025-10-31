import os, html, re, torch, sys, traceback, time, mlflow
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

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

def build_llm(
    llm_model: str,
    quantize: bool = False,
    use_4bit: bool = True,
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    use_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, pipeline]:
    """
    Build (optionally quantized) LLM + tokenizer + generation pipeline.

    Args:
      llm_model: HF model id
      quantize: whether to load a quantized model
      use_4bit: if quantize=True, prefer 4-bit (nf4) quantization. If False and quantize=True, will try 8-bit
      bnb_4bit_*: bitsandbytes config options for 4-bit
      use_8bit: explicit 8-bit flag (overrides use_4bit when quantize=True)

    Returns:
      (model, tokenizer, gen_pipeline)
    """

    tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    if quantize and not has_cuda:
        raise RuntimeError("Quantization (bitsandbytes) requires CUDA. Set quantize=False or run on GPU.")

    model = None

    if quantize:
        if use_4bit and not use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            )
        
        else:
            # fallback to 8-bit (older but widely supported)
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            device_map="auto",
            quantization_config=bnb_config,
            dtype=torch.float16
        )

    else:
        if has_cuda:
            model = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto", torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto")

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"  
    )

    return model, tokenizer, gen

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", required=True, help="Path to eval dataset CSV")
    parser.add_argument(
        "--model-name", "-ml",
        default="mistralai/mistral-7b-instruct-v0.3",
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
    dataset = dataset.rename(columns={"LLM_questions": "question", "LLM_answers": "answer",
                                      "questions":"question", "answers":"answer", "contexts":"golden_context","answer_contexts":"golden_context"})
    dataset = dataset.dropna(subset=["question", "answer", "golden_context"])
    dataset["golden_context"] = dataset["golden_context"].apply(literal_eval)
    
    dataset = dataset.loc[dataset["golden_context"].str.len() > 0]

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
    )
    kg_retriever = KnowledgeGraphRetriever(
        vector_store=vectorstore,
        neo4j_url=neo4j_cfg["url"],
        neo4j_username=neo4j_cfg["user"],
        neo4j_password=neo4j_cfg["password"],
        database="neo4j"
    )
    deduplicator = Deduplicator(embedder=vectorstore.embeddings)
    reranker = Reranker()

    _, _, gen = build_llm(model_name, quantize=True, use_8bit=True)

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

    evaluator = RAGEvaluator(dataset, rag, k_values=[3, 5, 10])
    try:
        evaluator.evaluate(rag_config, run_name=os.path.splitext(rag_config_name)[0], verbose=True)
    except Exception as e:
        traceback.print_exc()
    df_with_eval = evaluator.df
    base_path, _ = os.path.splitext(eval_df_path)
    finish_time = time.time()
    model_id = model_name.replace(":", "_")
    model_id = model_id.split("/")[-1]  # safely get last part even if no slash
    output_path = f"{base_path}_w_metrics_{os.path.splitext(rag_config_name)[0]}_{model_id}_{int(finish_time)}.csv"
    df_with_eval.to_csv(output_path, index=False)
    print(f"Saved evaluation results to {output_path}")    

if __name__ == "__main__":
    main()