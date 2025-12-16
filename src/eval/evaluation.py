import pandas as pd
import mlflow
import torch
import gc
from tqdm import tqdm
import traceback
import io, math, os
from src.rag.config import PipelineConfig
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
import json

from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from datetime import datetime

class RAGEvaluator:
    def __init__(
        self,
        df: pd.DataFrame,
        rag_model,
        k_values=[3, 5, 10],
        mlflow_uri: str = None,
        save_df_every: int = 50,
        eval_embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        class_context_column: str = "class_context",
        function_context_column: str = "function_context",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.rag = rag_model
        self.k_values = k_values
        self.class_context_column = class_context_column
        self.function_context_column = function_context_column
        
        if mlflow_uri:
            current = mlflow.get_tracking_uri()
            if mlflow_uri != current:
                mlflow.set_tracking_uri(mlflow_uri)

        self.save_df_every = save_df_every

        # Models for answer evaluation
        dev="cpu"
        if torch.cuda.is_available():
          dev = "cuda"
        self.bert_scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            rescale_with_baseline=True,
            device="cpu",
        )
        self.eval_embeddings = SentenceTransformer(eval_embed_model_name, device="cpu")

        self._prepare_columns()

    def _gpu_cleanup(self):
        """Private helper to clear GPU memory and reinitialize the agentic runner."""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  
        except Exception as e:
            print(f"[WARN] GPU cleanup failed: {e}\n")

    def _prepare_columns(self):
        for k in self.k_values:
            self.df[f"precision_{k}"] = None
            self.df[f"recall_{k}"] = None
            self.df[f"f1_{k}"] = None
            self.df[f"iou_{k}"] = None
            self.df[f"class_precision_{k}"] = None
            self.df[f"class_recall_{k}"] = None
            self.df[f"class_f1_{k}"] = None
            self.df[f"class_iou_{k}"] = None
        for metric in [
            "mrr",
            "class_mrr",
            "bleu",
            "meteor",
            "bertscore",
            "semantic_similarity",
            "generated_answer",
            "retrieved_functions",
            "retrieved_classes",
            "retrieved_docs",
            "alt_queries",
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
            result["top_classes"],
            result["answer"],
            result["top_docs"],
            result.get("query_type", "default"),
            result.get("alt_queries",[])
        )

    def evaluate_single(self, idx, row, config: PipelineConfig):
        """Evaluate a single datapoint and store results in df."""
        question = row.get("question", "")
        context = row.get(self.function_context_column, [])
        class_context = row.get(self.class_context_column, [])
        answer_ref = row.get("answer", "")

        top_functions, top_classes, answer_gen, top_docs, alt_queries, query_type  = self._run_rag(question, config)

        self.df.at[idx, "pred_query_class"] = query_type
        self.df.at[idx, "alt_queries"] = alt_queries
        self.df.at[idx, "generated_answer"] = answer_gen
        self.df.at[idx, "retrieved_functions"] = top_functions
        self.df.at[idx, "retrieved_classes"] = top_classes
        self.df.at[idx, "retrieved_docs"] = top_docs

        # Retrieval metrics
        for k in self.k_values:
            self.df.at[idx, f"precision_{k}"] = calculate_precision_at_k(top_functions, context, k)
            self.df.at[idx, f"recall_{k}"] = calculate_recall_at_k(top_functions, context, k)
            self.df.at[idx, f"f1_{k}"] = calculate_f1_at_k(top_functions, context, k)
            self.df.at[idx, f"iou_{k}"] = calculate_iou(top_functions, context, k)
            
            self.df.at[idx, f"class_precision_{k}"] = calculate_precision_at_k(top_classes, class_context, k)
            self.df.at[idx, f"class_recall_{k}"] = calculate_recall_at_k(top_classes, class_context, k)
            self.df.at[idx, f"class_f1_{k}"] = calculate_f1_at_k(top_classes, class_context, k)
            self.df.at[idx, f"class_iou_{k}"] = calculate_iou(top_classes, class_context, k)

        self.df.at[idx, "mrr"] = calculate_rr(top_functions, context)
        self.df.at[idx, "class_mrr"] = calculate_rr(top_classes, class_context)
        
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
                try:
                    self.evaluate_single(idx, row, config)
                except Exception as error:
                    print("[ERROR] Error while evaluating, saving results...")
                    traceback.print_exc()
                    break    
                self._gpu_cleanup()
                if config.verbose:
                    print("[GPU CLEANUP] Memory cleared successfully.")
                summary = self.get_live_summary(idx)
                pbar.set_postfix(summary)

            # Compute aggregate metrics
            metrics = {
                "bleu_mean": float(self.df["bleu"].mean()),
                "meteor_mean": float(self.df["meteor"].mean()),
                "bertscore_mean": float(self.df["bertscore"].mean()),
                "semantic_similarity_mean": float(self.df["semantic_similarity"].mean()),
                "mrr_mean": float(self.df["mrr"].mean()),
                "class_mrr_mean": float(self.df["class_mrr"].mean()),
            }
            for k in self.k_values:
                metrics[f"precision_{k}"] = float(self.df[f"precision_{k}"].mean())
                metrics[f"recall_{k}"] = float(self.df[f"recall_{k}"].mean())
                metrics[f"f1_{k}"] = float(self.df[f"f1_{k}"].mean())
                metrics[f"iou_{k}"] = float(self.df[f"iou_{k}"].mean())
                
                metrics[f"class_precision_{k}"] = float(self.df[f"class_precision_{k}"].mean())
                metrics[f"class_recall_{k}"] = float(self.df[f"class_recall_{k}"].mean())
                metrics[f"class_f1_{k}"] = float(self.df[f"class_f1_{k}"].mean())
                metrics[f"class_iou_{k}"] = float(self.df[f"class_iou_{k}"].mean())

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

class AgenticRAGEvaluator:
    """
    Evaluator for the Agentic LangGraph pipeline.
    Evaluates retrieval (functions/docs) and generation quality.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        agentic_runner,
        k_values=[3, 5, 10],
        mlflow_uri: str = None,
        eval_embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        class_context_column: str = "class_context",
        function_context_column: str = "function_context",
        gpu_cleanup_every: int | None = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.agentic_runner = agentic_runner
        self.k_values = k_values
        self.class_context_column = class_context_column
        self.function_context_column = function_context_column
        self.gpu_cleanup_every = gpu_cleanup_every

        if mlflow_uri:
            current = mlflow.get_tracking_uri()
            if mlflow_uri != current:
                mlflow.set_tracking_uri(mlflow_uri)

        # Models for evaluation
        dev="cpu"
        if torch.cuda.is_available():
          dev = "cuda"
        self.bert_scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            rescale_with_baseline=True,
            device="cpu",
        )
        self.eval_embeddings = SentenceTransformer(eval_embed_model_name, device="cpu")

        self._prepare_columns()
        
    async def _gpu_cleanup(self):
        """Private helper to clear GPU memory and reinitialize the agentic runner."""
        print("\n[GPU CLEANUP] Releasing GPU memory and rebuilding agentic runner...\n")

        try:
            # Step 1: Run garbage collection and free CUDA memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[GPU CLEANUP] Memory cleared successfully.")
    
            # Step 2: If the agentic runner supports graph rebuilding, trigger it
            if hasattr(self.agentic_runner, "reset_graph"):
                print("[GPU CLEANUP] Rebuilding agentic graph...")
                await self.agentic_runner.reset_graph()
                print("[GPU CLEANUP] Graph rebuilt successfully.")
            else:
                print("[GPU CLEANUP] Agentic runner does not support graph reset - skipping rebuild.")
        except Exception as e:
            print(f"[WARN] GPU cleanup failed: {e}\n")
    def _prepare_columns(self):
        """Initialize columns for metrics and logs."""
        for k in self.k_values:
            self.df[f"precision_{k}"] = None
            self.df[f"recall_{k}"] = None
            self.df[f"f1_{k}"] = None
            self.df[f"iou_{k}"] = None
            
            self.df[f"class_precision_{k}"] = None
            self.df[f"class_recall_{k}"] = None
            self.df[f"class_f1_{k}"] = None
            self.df[f"class_iou_{k}"] = None

        for metric in [
            "mrr",
            "class_mrr",  
            "bleu",
            "meteor",
            "bertscore",
            "semantic_similarity",
            "generated_answer",
            "relevant_functions",
            "relevant_classes",
            "retrieved_node_ids",
            "retrieved_docs",
            "tool_log",
            "tool_count",
        ]:
            self.df[metric] = None

    async def _run_agentic_pipeline(self, question: str, idx:int = None):
        """
        Run the full LangGraph pipeline and capture all tool calls and outputs.
        The `agentic_runner` should return:
            {
                'answer': str,
                'relevant_node_ids': [...],
                'relevant_docs': [...],
                'tool_log': [...],  # list of tool call dicts
            }
        """
        session_id = f"eval_{idx}"
        result = await self.agentic_runner.run_async(question, session_id)
        return result

    async def evaluate_single(self, idx, row):
        """Run one evaluation example."""
        question = row.get("question", "")
        context = row.get(self.function_context_column, [])
        class_context = row.get(self.class_context_column, [])
        answer_ref = row.get("answer", "")

        result = await self._run_agentic_pipeline(question, idx)
        answer_gen = result.get("answer", "")
        retrieved_nodes = result.get("relevant_node_ids", [])
        retrieved_functions = result.get("relevant_functions", [])
        retrieved_classes = result.get("relevant_classes", [])
        retrieved_docs = result.get("relevant_docs", [])
        tool_log = result.get("tool_log", [])

        self.df.at[idx, "generated_answer"] = answer_gen
        self.df.at[idx, "retrieved_node_ids"] = retrieved_nodes
        self.df.at[idx, "relevant_functions"] = retrieved_functions
        self.df.at[idx, "relevant_classes"] = retrieved_classes  
        if retrieved_docs and isinstance(retrieved_docs[0], dict):
            self.df.at[idx, "retrieved_docs"] = [d.get("page_content", "") for d in retrieved_docs]
        else:
             self.df.at[idx, "retrieved_docs"] = [d.page_content if hasattr(d, "page_content") else d for d in retrieved_docs]
        self.df.at[idx, "tool_log"] = json.dumps(tool_log)
        self.df.at[idx, "tool_count"] = len(tool_log)

        # --- Retrieval Metrics ---
        for k in self.k_values:
            self.df.at[idx, f"precision_{k}"] = calculate_precision_at_k(retrieved_functions, context, k)
            self.df.at[idx, f"recall_{k}"] = calculate_recall_at_k(retrieved_functions, context, k)
            self.df.at[idx, f"f1_{k}"] = calculate_f1_at_k(retrieved_functions, context, k)
            self.df.at[idx, f"iou_{k}"] = calculate_iou(retrieved_functions, context, k)
            
            self.df.at[idx, f"class_precision_{k}"] = calculate_precision_at_k(retrieved_classes, class_context, k)
            self.df.at[idx, f"class_recall_{k}"] = calculate_recall_at_k(retrieved_classes, class_context, k)
            self.df.at[idx, f"class_f1_{k}"] = calculate_f1_at_k(retrieved_classes, class_context, k)
            self.df.at[idx, f"class_iou_{k}"] = calculate_iou(retrieved_classes, class_context, k)

        self.df.at[idx, "mrr"] = calculate_rr(retrieved_functions, context)
        self.df.at[idx, "class_mrr"] = calculate_rr(retrieved_classes, class_context)

        # --- Generation Metrics ---
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

    async def evaluate(self, run_name: str = None, verbose=True):
        """Main evaluation loop."""
        with mlflow.start_run(run_name=run_name):
            pbar = tqdm(range(len(self.df)), desc="Evaluating Agentic RAG", unit="item")

            for idx in pbar:
                row = self.df.iloc[idx]
                try:
                    await self.evaluate_single(idx, row)
                except Exception as error:
                    print("[ERROR] Error while evaluating, saving results...")
                    traceback.print_exc()
                    break
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pbar.set_postfix(self.get_live_summary(idx))
                
                if self.gpu_cleanup_every is not None and (idx + 1) % self.gpu_cleanup_every == 0:
                    await self._gpu_cleanup()
            # Aggregate metrics
            metrics = {
                "bleu_mean": float(self.df["bleu"].mean()),
                "meteor_mean": float(self.df["meteor"].mean()),
                "bertscore_mean": float(self.df["bertscore"].mean()),
                "semantic_similarity_mean": float(self.df["semantic_similarity"].mean()),
                "mrr_mean": float(self.df["mrr"].mean()),
                "class_mrr_mean": float(self.df["class_mrr"].mean()),
                "tool_calls_mean": float(self.df["tool_count"].mean()),
            }
            for k in self.k_values:
                metrics[f"precision_{k}"] = float(self.df[f"precision_{k}"].mean())
                metrics[f"recall_{k}"] = float(self.df[f"recall_{k}"].mean())
                metrics[f"f1_{k}"] = float(self.df[f"f1_{k}"].mean())
                metrics[f"iou_{k}"] = float(self.df[f"iou_{k}"].mean())
                
                metrics[f"class_precision_{k}"] = float(self.df[f"class_precision_{k}"].mean())
                metrics[f"class_recall_{k}"] = float(self.df[f"class_recall_{k}"].mean())
                metrics[f"class_f1_{k}"] = float(self.df[f"class_f1_{k}"].mean())
                metrics[f"class_iou_{k}"] = float(self.df[f"class_iou_{k}"].mean())

            mlflow.log_metrics(metrics)

            # Write results
            out_path = f"evaluation_results.csv"
            self.df.to_csv(out_path, index=False)
            mlflow.log_artifact(out_path)

            if verbose:
                print("\nEvaluation complete. Aggregate metrics:")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")

    def export(self, path):
        """Export the evaluated dataframe."""
        self.df.to_csv(path, index=False)