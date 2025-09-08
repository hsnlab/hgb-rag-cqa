import itertools
import os
import pandas as pd
from ast import literal_eval
import mlflow
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import torch
from typing import Tuple
from transformers import BitsAndBytesConfig
import warnings

from rag.config import PipelineConfig
from rag.simple_rag import SimpleRAG
from rag.repo_rag import RepositoryRAG
from eval.evaluation import RAGEvaluator
from utils.deduplicator import Deduplicator
from utils.reranker import Reranker
from utils.qdrant_store import QdrantStore
from utils.retrieval import KnowledgeGraphRetriever

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

def safe_run_name(base: str) -> str:
    """Ensure run_name is filesystem/MLflow safe (no spaces)."""
    return base.replace(" ", "_").replace("/", "_")

def main():
    parser = argparse.ArgumentParser(description="Run RAG ablation study.")
    parser.add_argument("--eval_data_path",type=str,required=True,
        help="Path to evaluation dataset (CSV file)."
    )
    parser.add_argument("--qdrant_url",type=str,default="http://localhost:6333",
        help="Qdrant URL for vector store."
    )
    parser.add_argument("--qdrant_collection",type=str,required=True,
        help="Qdrant collection name."
    )
    parser.add_argument("--qdrant_embedding_model",type=str,default="microsoft/codebert-base",
        help="Embedding model for Qdrant."
    )
    parser.add_argument("--qdrant_dist_type",type=str,default="cosine",
        help="Distance type for Qdrant (cosine, euclidean, or dot)."
    )
    parser.add_argument("--qdrant_api_key",type=str,required=True,default=None,
        help="Qdrant API key if needed."
    )
    parser.add_argument("--neo4j_uri",type=str,default="bolt://localhost:7687",
        help="Neo4j URI for knowledge graph."
    )
    parser.add_argument("--neo4j_user",type=str,default="neo4j",
        help="Neo4j username."
    )
    parser.add_argument("--neo4j_password",type=str,default="password",
        help="Neo4j password."
    )
    parser.add_argument("--llm_model",type=str,default="mistralai/mistral-7b-instruct-v0.3",
        help="LLM model for RAG."
    )
    parser.add_argument("--quantize_llm", action="store_true",
                        help="If set, load the LLM in quantized mode (requires GPU).")
    parser.add_argument("--mlflow_uri", type=str, default="http://127.0.0.1:5000",
                    help="MLflow tracking URI (e.g., http://mlflow-server:5000 or file:///path/to/mlruns)."
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="If set, run only a single quick experiment on a small slice (for smoke test).")


    args = parser.parse_args()
    eval_data_apth = args.eval_data_path
    qdrant_url = args.qdrant_url
    qdrant_collection = args.qdrant_collection
    qdrant_dist_type = args.qdrant_dist_type
    qdrant_api_key = args.qdrant_api_key
    model_name = args.qdrant_embedding_model
    neo4j_uri = args.neo4j_uri
    neo4j_user = args.neo4j_user
    neo4j_password = args.neo4j_password
    neo4j_auth = (neo4j_user, neo4j_password)
    llm_model = args.llm_model
    quantize = args.quantize_llm
    mlflow_uri = args.mlflow_uri

    # Load huggingface API key
    script_dir = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.abspath(os.path.join(script_dir, "..", "_", "hf_token.txt"))
    with open(token_path, "r") as f:
        huggingface_apikey = f.read().strip()

    login(huggingface_apikey)

    # Load dataset
    df = pd.read_csv(eval_data_apth)  
    df["edit_functions"] = df["edit_functions"].apply(literal_eval)

    # Instantiate backend components
    vectorstore = QdrantStore(
            model_name=model_name,
            qdrant_url=qdrant_url,
            collection_name=qdrant_collection,
            api_key=qdrant_api_key,
            distance_type=qdrant_dist_type
        )
    kg_retriever = KnowledgeGraphRetriever(vector_store=vectorstore, neo4j_url =neo4j_uri, neo4j_username=neo4j_user, neo4j_password =neo4j_password, database= "neo4j")
    deduplicator = Deduplicator(embedder=vectorstore.embeddings)
    reranker = Reranker()

    # Build LLM
    llm, tokenizer, gen = build_llm(llm_model,quantize=quantize,use_8bit=True)
    # Instantiate both RAG variants
    simple_rag = SimpleRAG(
        vectorstore,
        neo4j_uri=neo4j_uri,
        neo4j_auth=neo4j_auth,
        llm=gen
    )
    repo_rag = RepositoryRAG(
        vectorstore,
        retriever=kg_retriever,
        deduplicator=deduplicator,
        reranker=reranker,
        llm=gen,
        neo4j_auth=neo4j_auth,
        neo4j_uri=neo4j_uri,
    )

    # Hyperparameter sweeps
    retrievers = ["simple","kg"]
    dedups = [True, False]
    reranks = [True, False]
    over_retrieve_factor = [10, 15]            
    over_retrieve_cap=[200, 300]              
    rerank_candidate_cap = [150, 200]

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "rag_ablation_study"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    done_runs = list(set(runs["tags.mlflow.runName"]))
    run_counter = 0

    # dry run too test everything works
    if args.dry_run:
        # run only a single small job on the first row for quick smoke testing
        sample_df = df.head(2).copy()
        cfg = PipelineConfig(
            retriever="simple",
            deduplicate=False,
            dedup_use_minhash=False,
            dedup_use_semantic=False,
            rerank=False,
            rerank_use_graph=False,
            top_k=3,
            llm_max_tokens=64,
            over_retrieve=False
        )
        print("Dry run: running a single quick job using 'simple' retriever on 2 rows")
        evaluator = RAGEvaluator(sample_df, simple_rag, k_values=[1, 3])
        evaluator.evaluate(cfg, run_name="dry_run_simple", verbose=True)
        return

    # Iterate over all retrievers and run evaluations
    for retr in retrievers:
        if retr == "simple":
            # Single, minimal config for simple retriever
            cfg = PipelineConfig(
                retriever="simple",
                deduplicate=False,
                dedup_use_minhash=False,
                dedup_use_semantic=False,
                rerank=False,
                rerank_use_graph=False,
                top_k=10,
                llm_max_tokens=200,
                over_retrieve=False
            )
            rag_impl = simple_rag
            run_name = safe_run_name(f"exp_{run_counter}_retr-simple_minimal")
            if run_name in done_runs:
                print(f"[{run_counter}] Skipping {run_name} (already done)")
                run_counter += 1
                continue
            print(f"[{run_counter}] Starting {run_name}")
            evaluator = RAGEvaluator(df.copy(), rag_impl, k_values=[3, 5, 10])
            evaluator.evaluate(cfg, run_name=run_name, verbose=False)
            print(f"[{run_counter}] Finished {run_name}")
            run_counter += 1

        else:
            # For "kg" retriever, only iterate flags if the parent feature is enabled
            for dedup, rr in itertools.product(dedups, reranks):
                # dependent flags
                minhash_iter = [True, False] if dedup else [False]
                semd_iter = [True, False] if dedup else [False]
                rrgraph_iter = [True, False] if rr else [False]
                rrpop_iter = [True, False] if rr else [False]

                for mh, semd, rrgraph, rrpop, orf, orc, rcc in itertools.product(
                    minhash_iter,
                    semd_iter,
                    rrgraph_iter,
                    rrpop_iter,
                    over_retrieve_factor,
                    over_retrieve_cap,
                    rerank_candidate_cap
                ):
                    cfg = PipelineConfig(
                        retriever="kg",
                        deduplicate=dedup,
                        dedup_use_minhash=mh,
                        dedup_use_semantic=semd,
                        rerank=rr,
                        rerank_use_graph=rrgraph,
                        rerank_use_popularity=rrpop,
                        top_k=10,
                        llm_max_tokens=150,
                        over_retrieve=True,
                        over_retrieve_factor=orf,
                        over_retrieve_cap=orc,
                        rerank_candidate_cap=rcc
                    )
                    rag_impl = repo_rag
                    run_name = safe_run_name(
                        f"exp_{run_counter}_retr-kg_dedup-{dedup}_minhash-{mh}_semdep-{semd}_rr-{rr}_rrgraph-{rrgraph}_rrpop-{rrpop}_orf-{orf}_orc-{orc}_rcc-{rcc}"
                    )
                    if run_name in done_runs:
                        print(f"[{run_counter}] Skipping {run_name} (already done)")
                        run_counter += 1
                        continue
                    print(f"[{run_counter}] Starting {run_name}")
                    evaluator = RAGEvaluator(df.copy(), rag_impl, k_values=[3, 5, 10])
                    evaluator.evaluate(cfg, run_name=run_name, verbose=False)
                    print(f"[{run_counter}] Finished {run_name}")
                    run_counter += 1


    print(f"All done — total runs: {run_counter}")


if __name__ == "__main__":
    main()