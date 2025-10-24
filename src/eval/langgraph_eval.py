# Data
import pandas as pd
import argparse
from ast import literal_eval
import os
import sys
from src.rag.agentic_langgraph import AgenticLangGraph
from .evaluation import AgenticRAGEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG model using a test DataFrame.")
    parser.add_argument(
            "eval_df_path",
            default="data/generated_qna_large_gpt-oss20b_medior_on_2025-10-17.csv",
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
    dataset = dataset.rename(columns={"LLM_questions": "question", "LLM_answers": "answer",
                                      "questions":"question", "answers":"answer", "contexts":"golden_context"})
    dataset = dataset.dropna(subset=["question", "answer", "golden_context"])
    dataset["golden_context"] = dataset["golden_context"].apply(literal_eval)
    dataset = dataset.loc[dataset["golden_context"].str.len() > 0]

    #sklearn_hier_json = pd.read_pickle("../graph/sklearn/sklearn_with_summaries.pkl")
    model_name = "gpt-oss:20b"
    tool = AgenticLangGraph(model_name=model_name)

    evaluator = AgenticRAGEvaluator(df=dataset, agentic_runner=tool, k_values=[3, 5, 10], context_column="golden_context")
    evaluator.evaluate(verbose=True)
    #evaluator.print_summary()
    
    df_with_eval = evaluator.df
    # Save to same folder with _w_metrics.csv
    base_path, _ = os.path.splitext(eval_df_path)
    output_path = f"{base_path}_w_metrics_langgraph_{model_name.replace(':', '_')}.csv"
    df_with_eval.to_csv(output_path, index=False)
if __name__ == "__main__":
    main()