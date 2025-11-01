import itertools
import os, re, html, httpx, gc, torch
import pandas as pd
from ast import literal_eval
import mlflow
import argparse
from huggingface_hub import login
from langchain_ollama import ChatOllama

from src.rag.config import PipelineConfig
from src.utils.config_loader import load_all_configs
from src.rag.simple_rag import SimpleRAG
from src.rag.repo_rag import RepositoryRAG
from src.eval.evaluation import RAGEvaluator
from src.utils.deduplicator import Deduplicator
from src.utils.reranker import Reranker
from src.utils.qdrant_store import QdrantStore
from src.utils.retrieval import KnowledgeGraphRetriever

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

def safe_run_name(base: str) -> str:
    """Ensure run_name is filesystem/MLflow safe (no spaces)."""
    return base.replace(" ", "_").replace("/", "_")

def main():
    parser = argparse.ArgumentParser(description="Run RAG ablation study.")
    parser.add_argument("--eval_data_path",type=str,required=True,
        help="Path to evaluation dataset (CSV file)."
    )
    parser.add_argument("--llm_model",type=str,default="mistral:7b",
        help="LLM model for RAG."
    )
    parser.add_argument(
        "--question-limit", "-ql",
        default=None,
        type=int,
        help="Limit the evaluation dataset to this number of questions",
        required=False
    )
    parser.add_argument("--mlflow_uri", type=str, default="http://127.0.0.1:5000",
                    help="MLflow tracking URI (e.g., http://mlflow-server:5000 or file:///path/to/mlruns)."
    )
    parser.add_argument("--mlflow_experiment", type=str, default="rag_ablation_study",
                        help="MLflow experiment name.")
    parser.add_argument("--dry_run", action="store_true",
                        help="If set, run only a single quick experiment on a small slice (for smoke test).")

    args = parser.parse_args()
    # -----------------------------------------------------
    # Load configs (HuggingFace, Qdrant, Neo4j, RAG)
    # -----------------------------------------------------
    qdrant_cfg, neo4j_cfg, _, base_rag_cfg = load_all_configs()
    login(open("./_/hf_token.txt").read().strip())

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    eval_df_path = args.eval_data_path
    q_limit = args.question_limit
    print(f"Loading dataset from: {eval_df_path}")
    try:
        dataset = pd.read_csv(eval_df_path)
    except:
        dataset = pd.read_csv(eval_df_path, sep="\t")
    dataset = dataset.rename(columns={"LLM_questions": "question", "LLM_answers": "answer",
                                      "questions":"question", "answers":"answer", "contexts":"golden_context","answer_contexts":"golden_context"})
    dataset = dataset.dropna(subset=["question", "answer", "golden_context"])
    dataset["golden_context"] = dataset["golden_context"].apply(literal_eval)
    
    dataset = dataset.loc[(dataset["golden_context"].str.len() > 0) & (dataset["question"].str.len() <= 1000)]

    if q_limit:
        dataset = dataset.iloc[:min(q_limit,len(dataset))]
    dataset["question"] = dataset["question"].apply(remove_html_tags)
    dataset["answer"] = dataset["answer"].apply(remove_html_tags)

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
    print(f"Loading Ollama model: {args.llm_model}")
    gen = build_llm(args.llm_model)

    # -----------------------------------------------------
    # Instantiate both RAG variants
    # -----------------------------------------------------
    simple_rag = SimpleRAG(
        vectorstore,
        neo4j_uri=neo4j_cfg["url"],
        neo4j_auth=(neo4j_cfg["user"], neo4j_cfg["password"]),
        llm=gen,
        classifier_model = "facebook/bart-large-mnli"
    )
    repo_rag = RepositoryRAG(
        vectorstore,
        retriever=kg_retriever,
        deduplicator=deduplicator,
        reranker=reranker,
        llm=gen,
        neo4j_uri=neo4j_cfg["url"],
        neo4j_auth=(neo4j_cfg["user"], neo4j_cfg["password"]),
        classifier_model = "facebook/bart-large-mnli"
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
        sample_df = dataset.head(2).copy()
        cfg = PipelineConfig(retriever="simple", top_k=3, llm_max_tokens=64, deduplicate=False, rerank=False)
        evaluator = RAGEvaluator(sample_df, simple_rag, k_values=[1, 3])
        evaluator.evaluate(cfg, run_name="dry_run_simple", verbose=True)
        return

    # -----------------------------------------------------
    # Define ablation grid (smart dependency control)
    # -----------------------------------------------------
    retrievers = ["simple", "kg"]
    dedup_flags = [False, True]
    rerank_flags = [False, True]
    shortest_path_flags = [False, True]  
    over_retrieve_flags = [False, True]

    run_counter = 0
    seen_configs = set()

    for retr in retrievers:
        if retr == "simple":
            cfg = PipelineConfig(
                retriever="simple",
                deduplicate=False,
                rerank=False,
                top_k=10,
                llm_max_tokens=200,
                over_retrieve=False,
                use_query_expansion=True,
                use_shortest_path_context=False,
            )
            run_name = safe_run_name(f"exp_{run_counter}_retr-simple_base")
            if run_name in done_runs:
                print(f"[{run_counter}] Skipping {run_name} (already done)")
                run_counter += 1
                continue

            print(f"[{run_counter}] Running {run_name}")
            evaluator = RAGEvaluator(dataset.copy(), simple_rag, k_values=[3, 5, 10])
            evaluator.evaluate(cfg, run_name=run_name, verbose=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            run_counter += 1

        elif retr == "kg":
            for dedup, rerank, overr, sp_flag in itertools.product(dedup_flags, rerank_flags, over_retrieve_flags, shortest_path_flags):
                # handle sub-options only when relevant
                minhash_opts = [True, False] if dedup else [False]
                semantic_opts = [True, False] if dedup else [False]
                graph_opts = [True, False] if rerank else [False]
                pop_opts = [True, False] if rerank else [False]
                rerank_candidate_caps = [50, 100] if rerank else [0]
                over_retrieve_factors = [5, 10] if overr else [0]
                over_retrieve_caps = [100, 200] if overr else [0]
                for mh, semd, rrgraph, rrpop, orf, orc, rcc in itertools.product(
                    minhash_opts,
                    semantic_opts,
                    graph_opts,
                    pop_opts,
                    over_retrieve_factors,
                    over_retrieve_caps,
                    rerank_candidate_caps,
                ):
                    # prune irrelevant combos
                    if not dedup and (mh or semd):
                        continue
                    if not rerank and (rrgraph or rrpop):
                        continue

                    config_signature = (
                        dedup, mh, semd, rerank, rrgraph, rrpop,
                        overr, orf, orc, rcc, sp_flag
                    )
                    if config_signature in seen_configs:
                        continue
                    seen_configs.add(config_signature)

                    cfg = PipelineConfig(
                        retriever="kg",
                        deduplicate=dedup,
                        dedup_use_minhash=mh,
                        dedup_use_semantic=semd,
                        rerank=rerank,
                        rerank_use_graph=rrgraph,
                        rerank_use_popularity=rrpop,
                        top_k=10,
                        llm_max_tokens=200,
                        over_retrieve=True,
                        over_retrieve_factor=orf,
                        over_retrieve_cap=orc,
                        rerank_candidate_cap=rcc,
                        use_shortest_path_context=sp_flag,
                        use_query_expansion=True,
                    )

                    run_name = safe_run_name(
                        f"exp_{run_counter}_retr-kg_dedup-{dedup}_mh-{mh}_sem-{semd}_"
                        f"rr-{rerank}_graph-{rrgraph}_pop-{rrpop}_sp-{sp_flag}_orf-{orf}_orc-{orc}_rcc-{rcc}"
                    )
                    if run_name in done_runs:
                        print(f"[{run_counter}] Skipping {run_name} (already done)")
                        run_counter += 1
                        continue

                    print(f"[{run_counter}] Running {run_name}")
                    evaluator = RAGEvaluator(dataset.copy(), repo_rag, k_values=[3, 5, 10])
                    evaluator.evaluate(cfg, run_name=run_name, verbose=False)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    run_counter += 1

    print(f"All done - total executed runs: {run_counter}")


if __name__ == "__main__":
    main()