# Data
import pandas as pd
import argparse
from ast import literal_eval
import os
import sys
sys.path.append("..")
from rag.repo_rag import RepositoryRAG
from rag.config import PipelineConfig
from utils.deduplicator import Deduplicator
from utils.reranker import Reranker
from utils.qdrant_store import QdrantStore
from utils.retrieval import KnowledgeGraphRetriever

from evaluation import RAGEvaluator
from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import login


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

    # tokenizer (safe to always load)
    tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # decide device availability
    has_cuda = torch.cuda.is_available()
    if quantize and not has_cuda:
        raise RuntimeError("Quantization (bitsandbytes) requires CUDA. Set quantize=False or run on GPU.")

    model = None

    if quantize:
        # prefer explicit 4-bit if use_4bit True and use_8bit False
        if use_4bit and not use_8bit:
            # 4-bit config using BitsAndBytesConfig
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
            torch_dtype=torch.float16
        )

    else:
        # Full precision or mixed precision (let transformers pick optimal device_map)
        # If CUDA present, we request float16 for speed; otherwise default dtype.
        if has_cuda:
            model = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto", torch_dtype=torch.float16)
        else:
            # CPU fallback (may be slow)
            model = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto")

    # Build generation pipeline. We pass the already loaded model + tokenizer.
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"  # keep this consistent with model device_map
    )

    return model, tokenizer, gen

def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG model using a test DataFrame.")
    parser.add_argument(
            "eval_df_path",
            default="../data/eval_df.csv",
            type=str,
            help="Path to the CSV file containing the evaluation dataframe."
        )
    args = parser.parse_args()
    
    eval_df_path = args.eval_df_path

    if not os.path.isfile(eval_df_path):
        print(f"Error: File '{eval_df_path}' does not exist.")
        sys.exit(1)
        
    print(f"Loading dataset from: {eval_df_path}")
    try:
        dataset = pd.read_csv(eval_df_path)
    except:
        dataset = pd.read_csv(eval_df_path, sep="\t")
    dataset = dataset.rename(columns={"LLM_questions": "question", "LLM_answers": "answer", "edit_functions": "golden_context",
                                      "questions": "question", "answers": "answer", "contexts": "golden_context"})
    dataset = dataset.dropna(subset=["question", "answer", "golden_context"])
    dataset["golden_context"] = dataset["golden_context"].apply(literal_eval)
    dataset = dataset.loc[dataset["golden_context"].str.len() > 0]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.abspath(os.path.join(script_dir, "..","..", "_", "hf_token.txt"))
    with open(token_path, "r") as f:
        huggingface_apikey = f.read().strip()

    login(huggingface_apikey)
    token_path = os.path.abspath(os.path.join(script_dir, "..","..", "_", "drant_api_key.txt"))
    with open(token_path, "r") as f:
        qdrant_api_key = f.read().strip()
    vectorstore = QdrantStore(
            model_name="microsoft/codebert-base",
            qdrant_url="http://localhost:6333",
            collection_name="rag_collection_codebert-base_cosine",
            api_key=qdrant_api_key,
            distance_type="cosine"
        )
    neo4j_user = "neo4j"
    neo4j_password = "password"
    neo4j_uri = "bolt://localhost:7687"
    kg_retriever = KnowledgeGraphRetriever(vector_store=vectorstore, neo4j_url =neo4j_uri, neo4j_username=neo4j_user, neo4j_password =neo4j_password, database= "neo4j")
    deduplicator = Deduplicator(embedder=vectorstore.embeddings)
    reranker = Reranker()

    # Build LLM
    llm, tokenizer, gen = build_llm("mistralai/Mistral-7B-Instruct-v0.3",quantize=True,use_8bit=True)
    tool =  RepositoryRAG(
        vectorstore,
        retriever=kg_retriever,
        deduplicator=deduplicator,
        reranker=reranker,
        llm=gen,
        neo4j_auth=(neo4j_user, neo4j_password),
        neo4j_uri=neo4j_uri,
    )

    cfg = PipelineConfig(
            retriever="kg",
            deduplicate=True,
            dedup_use_minhash=False,
            dedup_use_semantic=True,
            rerank=True,
            rerank_use_graph=True,
            rerank_use_popularity=True,
            top_k=10,
            llm_max_tokens=150,
            over_retrieve=True,
        )
    
    evaluator = RAGEvaluator(df=dataset, rag_model=tool, k_values=[3, 5, 10])
    evaluator.evaluate(cfg, verbose=False)
    #evaluator.print_summary()
    
    df_with_eval = evaluator.df
    # Save to same folder with _w_metrics.csv
    base_path, _ = os.path.splitext(eval_df_path)
    output_path = f"{base_path}_w_metrics_kgrag.csv"
    df_with_eval.to_csv(output_path, index=False)
if __name__ == "__main__":
    main()