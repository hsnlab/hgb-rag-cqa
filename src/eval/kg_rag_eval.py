# Data
import pandas as pd
import argparse
from ast import literal_eval
import os
import sys
sys.path.append("..")
from rag.repo_rag import RepositoryRAG
from eval.evaluation import RAGEvaluator  

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
    dataset = pd.read_csv(eval_df_path, sep=None, engine="python")

    if "contexts" in dataset.columns:
        dataset["contexts"] = dataset["contexts"].apply(literal_eval)
    else:
        print("⚠️  No 'contexts' column found — using empty lists for evaluation context.")
        dataset["contexts"] = [[] for _ in range(len(dataset))]

    #sklearn_hier_json = pd.read_pickle("../graph/sklearn/sklearn_with_summaries.pkl")
    tool = RepositoryRAG(qdrant_api_key="@lmafa12", quantize=True)

    evaluator = RAGEvaluator(df=dataset, rag_model=tool, k_values=[3, 5, 10])
    evaluator.evaluate(verbose=True)
    evaluator.print_summary()
    
    df_with_eval = evaluator.df
    # Save to same folder with _w_metrics.csv
    base_path, _ = os.path.splitext(eval_df_path)
    output_path = f"{base_path}_w_metrics_3.csv"
    df_with_eval.to_csv(output_path, index=False)
if __name__ == "__main__":
    main()