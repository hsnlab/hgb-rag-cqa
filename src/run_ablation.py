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

from src.rag.config import PipelineConfig
from src.utils.config_loader import load_all_configs
from src.rag.simple_rag import SimpleRAG
from src.rag.repo_rag import RepositoryRAG
from src.eval.evaluation import RAGEvaluator
from src.utils.deduplicator import Deduplicator
from src.utils.reranker import Reranker
from src.utils.qdrant_store import QdrantStore
from src.utils.retrieval import KnowledgeGraphRetriever

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
    parser.add_argument("--llm_model",type=str,default="mistralai/mistral-7b-instruct-v0.3",
        help="LLM model for RAG."
    )
    parser.add_argument("--quantize_llm", action="store_true",
                        help="If set, load the LLM in quantized mode (requires GPU).")
    parser.add_argument("--mlflow_uri", type=str, default="http://127.0.0.1:5000",
                    help="MLflow tracking URI (e.g., http://mlflow-server:5000 or file:///path/to/mlruns)."
    )
    parser.add_argument("--mlflow_experiment", type=str, default="rag_ablation_study",
                        help="MLflow experiment name.")
    parser.add_argument("--dry_run", action="store_true",
                        help="If set, run only a single quick experiment on a small slice (for smoke test).")

    import os 
    print(os.getenv("HF_HOME"))
    args = parser.parse_args()
    # -----------------------------------------------------
    # Load configs (HuggingFace, Qdrant, Neo4j, RAG)
    # -----------------------------------------------------
    qdrant_cfg, neo4j_cfg, _, base_rag_cfg = load_all_configs()
    login(open("./_/hf_token.txt").read().strip())

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    df = pd.read_csv(args.eval_data_path)
    df["edit_functions"] = df["edit_functions"].apply(literal_eval)

    # -----------------------------------------------------
    # Build backend components
    # -----------------------------------------------------
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
        database="neo4j",
    )
    deduplicator = Deduplicator(embedder=vectorstore.embeddings)
    reranker = Reranker()

    # -----------------------------------------------------
    # Build LLM (respects CLI args)
    # -----------------------------------------------------
    print(f"Loading LLM: {args.llm_model} - Quantized={args.quantize_llm}")
    _, _, gen = build_llm(args.llm_model, quantize=args.quantize_llm, use_8bit=True)

    # -----------------------------------------------------
    # Instantiate both RAG variants
    # -----------------------------------------------------
    simple_rag = SimpleRAG(
        vectorstore,
        neo4j_uri=neo4j_cfg["url"],
        neo4j_auth=(neo4j_cfg["user"], neo4j_cfg["password"]),
        llm=gen,
    )
    repo_rag = RepositoryRAG(
        vectorstore,
        retriever=kg_retriever,
        deduplicator=deduplicator,
        reranker=reranker,
        llm=gen,
        neo4j_uri=neo4j_cfg["url"],
        neo4j_auth=(neo4j_cfg["user"], neo4j_cfg["password"]),
    )

    # -----------------------------------------------------
    # MLflow setup
    # -----------------------------------------------------
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    experiment = mlflow.get_experiment_by_name(args.mlflow_experiment)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id]) if experiment else pd.DataFrame()
    done_runs = set(runs["tags.mlflow.runName"]) if not runs.empty else set()

    # -----------------------------------------------------
    # Dry run (quick smoke test)
    # -----------------------------------------------------
    if args.dry_run:
        print("[DRY RUN] Running quick 2-row test with simple retriever...")
        sample_df = df.head(2).copy()
        cfg = PipelineConfig(retriever="simple", top_k=3, llm_max_tokens=64, deduplicate=False, rerank=False)
        evaluator = RAGEvaluator(sample_df, simple_rag, k_values=[1, 3])
        evaluator.evaluate(cfg, run_name="dry_run_simple", verbose=True)
        return

    # -----------------------------------------------------
    # Define ablation grid (smart dependency control)
    # -----------------------------------------------------
    retrievers = ["simple", "kg"]
    dedup_flags = [True, False]
    rerank_flags = [True, False]
    over_retrieve_factor = [15]
    over_retrieve_cap = [300]
    rerank_candidate_cap = [300]

    run_counter = 0

    for retr in retrievers:
        if retr == "simple":
            cfg = PipelineConfig(
                retriever="simple",
                deduplicate=False,
                dedup_use_minhash=False,
                dedup_use_semantic=False,
                rerank=False,
                rerank_use_graph=False,
                top_k=10,
                llm_max_tokens=150,
                over_retrieve=False,
            )
            run_name = safe_run_name(f"exp_{run_counter}_retr-simple_minimal")
            if run_name in done_runs:
                print(f"[{run_counter}] Skipping {run_name} (already done)")
                run_counter += 1
                continue
            print(f"[{run_counter}] Running {run_name}")
            evaluator = RAGEvaluator(df.copy(), simple_rag, k_values=[3, 5, 10])
            evaluator.evaluate(cfg, run_name=run_name, verbose=False)
            run_counter += 1

        elif retr == "kg":
            for dedup, rerank in itertools.product(dedup_flags, rerank_flags):
                minhash_opts = [True, False] if dedup else [False]
                semantic_opts = [True, False] if dedup else [False]
                graph_opts = [True, False] if rerank else [False]
                pop_opts = [True, False] if rerank else [False]

                for mh, semd, rrgraph, rrpop, orf, orc, rcc in itertools.product(
                    minhash_opts,
                    semantic_opts,
                    graph_opts,
                    pop_opts,
                    over_retrieve_factor,
                    over_retrieve_cap,
                    rerank_candidate_cap,
                ):
                    if not dedup and (mh or semd):
                        continue
                    if not rerank and (rrgraph or rrpop):
                        continue

                    cfg = PipelineConfig(
                        retriever="kg",
                        deduplicate=dedup,
                        dedup_use_minhash=mh,
                        dedup_use_semantic=semd,
                        rerank=rerank,
                        rerank_use_graph=rrgraph,
                        rerank_use_popularity=rrpop,
                        top_k=10,
                        llm_max_tokens=150,
                        over_retrieve=True,
                        over_retrieve_factor=orf,
                        over_retrieve_cap=orc,
                        rerank_candidate_cap=rcc,
                    )

                    run_name = safe_run_name(
                        f"exp_{run_counter}_retr-kg_dedup-{dedup}_mh-{mh}_sem-{semd}_rr-{rerank}_graph-{rrgraph}_pop-{rrpop}_orf-{orf}_orc-{orc}_rcc-{rcc}"
                    )
                    if run_name in done_runs:
                        print(f"[{run_counter}] Skipping {run_name} (already done)")
                        run_counter += 1
                        continue

                    print(f"[{run_counter}] Running {run_name}")
                    evaluator = RAGEvaluator(df.copy(), repo_rag, k_values=[3, 5, 10])
                    evaluator.evaluate(cfg, run_name=run_name, verbose=False)
                    run_counter += 1

    print(f"All done - total executed runs: {run_counter}")


if __name__ == "__main__":
    main()