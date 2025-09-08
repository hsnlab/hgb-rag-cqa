import os
import pandas as pd
import mlflow
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.config import PipelineConfig
from huggingface_hub import login
from .metrics import ( 
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_f1_at_k,
    calculate_rr,
    calculate_iou,
    evaluate_answer,
    evaluate_with_bertscore,
    evaluate_semantic_similarity,
)

from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer


class RAGEvaluator:
    def __init__(
        self,
        df: pd.DataFrame,
        rag_model,
        k_values=[3, 5, 10],
        mlflow_uri: str = None,
        huggingface_apikey: str = "",
        eval_embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.df = df.copy()
        self.rag = rag_model
        self.k_values = k_values

        if mlflow_uri:
            current = mlflow.get_tracking_uri()
            if mlflow_uri != current:
                mlflow.set_tracking_uri(mlflow_uri)

        # Set HF tokens
        #login(huggingface_apikey)

        # Models for answer evaluation
        self.bert_scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            rescale_with_baseline=True,
        )
        self.eval_embeddings = SentenceTransformer(eval_embed_model_name)

        self._prepare_columns()

    def _prepare_columns(self):
        for k in self.k_values:
            self.df[f"precision_{k}"] = None
            self.df[f"recall_{k}"] = None
            self.df[f"f1_{k}"] = None
            self.df[f"iou_{k}"] = None
        for metric in [
            "mrr",
            "bleu",
            "meteor",
            "bertscore",
            "semantic_similarity",
            "generated_answer",
            "retrieved_functions",
            "retrieved_docs",
            "pred_query_class",
        ]:
            self.df[metric] = None

    def _run_rag(self, question: str, config: PipelineConfig):
        """
        Run RAG and return its outputs.
        RepositoryRAG.run already orchestrates retrieval, dedup, rerank, enrichment, generation.
        """
        result = self.rag.run(question, config)
        return (
            result["top_functions"],
            result["answer"],
            result["top_docs"],
            result.get("query_type", "default"),
        )

    def evaluate_single(self, idx, row, config: PipelineConfig):
        """Evaluate a single datapoint and store results in df."""
        question = row.get("question", "")
        context = row.get("edit_functions", [])
        answer_ref = row.get("answer", "")

        top_functions, answer_gen, top_docs, query_type = self._run_rag(question, config)

        self.df.at[idx, "pred_query_class"] = query_type
        self.df.at[idx, "generated_answer"] = answer_gen
        self.df.at[idx, "retrieved_functions"] = top_functions
        self.df.at[idx, "retrieved_docs"] = top_docs

        # Retrieval metrics
        for k in self.k_values:
            self.df.at[idx, f"precision_{k}"] = calculate_precision_at_k(top_functions, context, k)
            self.df.at[idx, f"recall_{k}"] = calculate_recall_at_k(top_functions, context, k)
            self.df.at[idx, f"f1_{k}"] = calculate_f1_at_k(top_functions, context, k)
            self.df.at[idx, f"iou_{k}"] = calculate_iou(top_functions, context, k)

        self.df.at[idx, "mrr"] = calculate_rr(top_functions, context)

        # QA metrics
        bleu, meteor = evaluate_answer(answer_ref, answer_gen)
        bertscore = evaluate_with_bertscore(answer_ref, answer_gen, self.bert_scorer)
        semsim = evaluate_semantic_similarity(answer_ref, answer_gen, self.eval_embeddings)

        self.df.at[idx, "bleu"] = bleu
        self.df.at[idx, "meteor"] = meteor
        self.df.at[idx, "bertscore"] = bertscore
        self.df.at[idx, "semantic_similarity"] = semsim

    def get_live_summary(self, idx):
        """Return running average metrics for tqdm display."""
        df_slice = self.df.iloc[: idx + 1]
        return {
            "MRR": round(df_slice["mrr"].mean(), 4),
            "BLEU": round(df_slice["bleu"].mean(), 3),
            "BERT": round(df_slice["bertscore"].mean(), 3),
            "SemSim": round(df_slice["semantic_similarity"].mean(), 3),
        }

    def evaluate(self, config: PipelineConfig, run_name: str = None, verbose=True):
        """
        Run evaluation over dataset with tqdm + MLflow logging.
        """
        with mlflow.start_run(run_name=run_name):
            # log config params
            mlflow.log_params(config.__dict__)

            pbar = tqdm(range(len(self.df)), desc="Evaluating", unit="item")
            for idx in pbar:
                row = self.df.iloc[idx]
                self.evaluate_single(idx, row, config)
                summary = self.get_live_summary(idx)
                pbar.set_postfix(summary)

            # Compute aggregate metrics
            metrics = {
                "bleu_mean": float(self.df["bleu"].mean()),
                "meteor_mean": float(self.df["meteor"].mean()),
                "bertscore_mean": float(self.df["bertscore"].mean()),
                "semantic_similarity_mean": float(self.df["semantic_similarity"].mean()),
                "mrr_mean": float(self.df["mrr"].mean()),
            }
            for k in self.k_values:
                metrics[f"precision_{k}"] = float(self.df[f"precision_{k}"].mean())
                metrics[f"recall_{k}"] = float(self.df[f"recall_{k}"].mean())
                metrics[f"f1_{k}"] = float(self.df[f"f1_{k}"].mean())
                metrics[f"iou_{k}"] = float(self.df[f"iou_{k}"].mean())

            mlflow.log_metrics(metrics)

            # Save results to CSV + log artifact
            out_path = "evaluation_results.csv"
            self.df.to_csv(out_path, index=False)
            mlflow.log_artifact(out_path)

            # Save config used
            with open("config_used.txt", "w") as f:
                f.write(str(config))
            mlflow.log_artifact("config_used.txt")

            if verbose:
                print("Evaluation complete. Aggregate metrics:")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")

    def export(self, path):
        self.df.to_csv(path, index=False)
