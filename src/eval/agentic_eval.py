from dataclasses import asdict
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import pipeline + evaluator
from src.eval.evaluation import RAGEvaluator
from src.rag.agent_generate import run_agents
from src.rag.config import PipelineConfig


class AgenticEvaluator(RAGEvaluator):
    """
    AgenticEvaluator:
    A CrewAI-alapú (agentes) RAG pipeline-t értékeli az offline .csv kérdés–válasz párokon.
    Ugyanazokat a metrikákat számolja (BLEU, METEOR, BERTScore, SemSim, Recall@k, MRR),
    mint a klasszikus RAGEvaluator, de a RAG helyett a CrewAI agent pipeline-t hívja.
    """

    def _run_rag(self, question: str, config_dict):
        """
        Overriding _run_rag: ahelyett, hogy a RepositoryRAG-ot hívnánk,
        a CrewAI agent pipeline-t futtatjuk (run_agents).
        """
        try:
            result = run_agents(
                question,
                context=config_dict,  # a config paramétereket átadjuk contextként
                huggingface_apikey=config_dict.get("huggingface_apikey")
            )
            print(f"[DEBUG] Agentic result keys: {list(result.keys()) if result else 'None'}")
            print(f"[DEBUG] Agentic result raw: {result}")
        except Exception as e:
            print(f"[⚠️] Agentic pipeline error for question '{question[:60]}...': {e}")
            return [], "", [], "error"

        # Igazodunk a run_agents visszatérési szerkezetéhez
        return (
            result.get("retrieved_functions", []),
            result.get("final_answer") or result.get("gen") or result.get("answer", "") or result.get("output", ""),
            result.get("retrieved_docs", []),
            result.get("query_type", "default"),
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run agentic evaluation on a QnA CSV file")
    parser.add_argument("eval_df_path", type=str, help="Path to CSV file with question/answer pairs")
    args = parser.parse_args()

    eval_df_path = args.eval_df_path
    print(f"📂 Loading dataset from: {eval_df_path}")

    # Load dataset
    dataset = pd.read_csv(eval_df_path, sep=None, engine="python", quoting=3, on_bad_lines="skip")
    dataset = dataset.head(30)

    print("⚠️ Empty question rows:", dataset["questions"].isna().sum())
    dataset = dataset.dropna(subset=["questions"])

    if "contexts" not in dataset.columns:
        print("⚠️  No 'contexts' column found — using empty lists for evaluation context.")
        dataset["contexts"] = [[] for _ in range(len(dataset))]

    with open(os.path.join(os.path.dirname(__file__), "D:/EricssonCodeGraph/hgb-rag-cqa/_/hf_token.txt"), "r") as f:
        hf_token = f.read().strip()

    # Define config
    config = PipelineConfig(
        verbose=True,
        retriever="hybrid",
        deduplicate=True,
        dedup_use_minhash=True,
        rerank=True,
        rerank_use_graph=True,
        top_k=10,
        llm_max_tokens=300,
    )

    config.huggingface_apikey = hf_token

    # Init agentic evaluator
    evaluator = AgenticEvaluator(
        df=dataset,
        rag_model=None,  # agentek futtatják helyette
        huggingface_apikey=config.huggingface_apikey,
    )

    # Run evaluation
    print("\n🚀 Running agentic evaluation...")
    config_dict = asdict(config)
    evaluator.evaluate(config=config_dict, run_name="agentic_eval_run", verbose=True)
    print("\n✅ Agentic evaluation complete. Results saved to evaluation_results.csv")


if __name__ == "__main__":
    main()
