import itertools
import pandas as pd
import mlflow
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

from src.rag.config import PipelineConfig
from src.rag.simple_rag import SimpleRAG
from src.rag.repository_rag import RepositoryRAG
from src.eval.evaluator import RAGEvaluator
from src.utils.dedup import Deduplicator
from src.utils.rerank import Reranker
from src.utils.qdrant_store import QdrantStore
from src.utils.kg_retriever import KnowledgeGraphRetriever

def build_llm(llm_model: str):
    tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto")
    gen = pipeline("text-generation", model=llm, tokenizer=tokenizer, device_map="auto")
    return llm, tokenizer, gen

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
    parser.add_argument("--mlflow_uri", type=str, default=None,
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
    mlflow_uri = args.mlflow_uri

    # Load huggingface API key
    with open("../_/hf_token.txt", "r") as f:
        huggingface_apikey = f.read().strip()

    login(huggingface_apikey)

    # Load dataset
    df = pd.read_csv(eval_data_apth)  

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
    llm, tokenizer, gen = build_llm(llm_model)
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
        llm=gen
    )

    # Hyperparameter sweeps
    retrievers = ["simple", "kg"]
    dedups = [False, True]
    minhash_flags = [False, True]
    semantic_dedup_flags = [False, True]
    reranks = [False, True]
    rerank_graph_flags = [False, True]
    over_retrieve_factor = [10]            
    over_retrieve_cap=[100,  200]              
    rerank_candidate_cap = [50, 100]

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("rag_ablation_study")
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
        )
        print("Dry run: running a single quick job using 'simple' retriever on 2 rows")
        evaluator = RAGEvaluator(sample_df, simple_rag, k_values=[1, 3])
        evaluator.evaluate(cfg, run_name="dry_run_simple", verbose=True)
        return

    # Iterate over all combinations and run evaluations
    for retr in retrievers:
        if retr == "simple":
            # single, minimal config for simple retriever
            cfg = PipelineConfig(
                retriever="simple",
                deduplicate=False,
                dedup_use_minhash=False,
                dedup_use_semantic=False,
                rerank=False,
                rerank_use_graph=False,
                top_k=10,
                llm_max_tokens=200,
            )
            rag_impl = simple_rag
            run_name = safe_run_name(f"exp_{run_counter}_retr-simple_minimal")
            print(f"[{run_counter}] Starting {run_name}")
            evaluator = RAGEvaluator(df.copy(), rag_impl, k_values=[3, 5, 10])
            evaluator.evaluate(cfg, run_name=run_name, verbose=False)
            print(f"[{run_counter}] Finished {run_name}")
            run_counter += 1

        else:
            for dedup, mh, semd, rr, rrgraph,orf,orc,rcc in itertools.product(
                dedups, minhash_flags, semantic_dedup_flags, reranks, rerank_graph_flags, over_retrieve_factor, over_retrieve_cap, rerank_candidate_cap
            ):
                cfg = PipelineConfig(
                    retriever="kg",
                    deduplicate=dedup,
                    dedup_use_minhash=mh,
                    dedup_use_semantic=semd,
                    rerank=rr,
                    rerank_use_graph=rrgraph,
                    top_k=10,
                    llm_max_tokens=200,
                    over_retrieve=True,
                    over_retrieve_factor=orf,
                    over_retrieve_cap=orc,
                    rerank_candidate_cap=rcc
                )
                rag_impl = repo_rag
                run_name = safe_run_name(
                    f"exp_{run_counter}_retr-kg_dedup-{dedup}_minhash-{mh}_semdep-{semd}_rr-{rr}_rrgraph-{rrgraph}"
                )
                print(f"[{run_counter}] Starting {run_name}")
                evaluator = RAGEvaluator(df.copy(), rag_impl, k_values=[3, 5, 10])
                evaluator.evaluate(cfg, run_name=run_name, verbose=False)
                print(f"[{run_counter}] Finished {run_name}")
                run_counter += 1

    print(f"All done — total runs: {run_counter}")


if __name__ == "__main__":
    main()