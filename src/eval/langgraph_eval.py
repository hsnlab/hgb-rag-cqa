# Data
import pandas as pd
import argparse
import traceback
from ast import literal_eval
import os
import re
import html
import sys
from src.rag.agentic_langgraph import AgenticLangGraph
from .evaluation import AgenticRAGEvaluator
import time
import asyncio

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

async def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG model using a test DataFrame.")
    parser.add_argument(
            "--eval-path",
            default="data/generated_qna_large_gpt-oss20b_medior_on_2025-10-17.csv",
            type=str,
            help="Path to the CSV file containing the evaluation dataframe."
        )
    parser.add_argument(
        "--model-name", "-ml",
        default="gpt-oss:20b",
        type=str,
        help="Name of Ollama model to use for agents.",
        required=False
    )
    parser.add_argument(
        "--question-limit", "-ql",
        default=None,
        type=int,
        help="Limit the evaluation dataset to this number of questions",
        required=False
    )
    parser.add_argument(
        "--pipeline",
        choices=["free", "strict"],
        default="free",
        type=str,
        required=False,
        help="Pipeline version to run: 'free' (src.rag.agentic_langgraph) or 'strict' (src.rag.agentic_langgraph_strict)."
    )

    args = parser.parse_args()
    
    eval_df_path = args.eval_path
    q_limit = args.question_limit
    model_name = args.model_name
    pipeline_version = args.pipeline

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
    if q_limit:
        dataset = dataset.iloc[:min(q_limit,len(dataset))]
    dataset["question"] = dataset["question"].apply(remove_html_tags)
    dataset["answer"] = dataset["answer"].apply(remove_html_tags)

    if pipeline_version == "free":
        print(f"Running with pipeline: free (src.rag.agentic_langgraph)")
        from src.rag.agentic_langgraph import AgenticLangGraph
        tool = AgenticLangGraph(model_name=model_name)
    elif pipeline_version == "strict":
        print(f"Running with pipeline: strict (src.rag.agentic_langgraph_strict)")
        from src.rag.agentic_langgraph_strict import StrictAgenticLangGraph
        tool = StrictAgenticLangGraph(model_name=model_name)
    else:
        print(f"Error: Unknown pipeline version '{pipeline_version}'")
        sys.exit(1)

    evaluator = AgenticRAGEvaluator(df=dataset, agentic_runner=tool, k_values=[3, 5, 10], context_column="golden_context", gpu_cleanup_every=10)
    try:
        await evaluator.evaluate(verbose=True)
    except Exception as e:
        print("-" * 50)
        print(f"[ERROR] Evaluation halted on an exception: {e}")
        # Use traceback.print_exc() to print the full stack trace
        traceback.print_exc() 
        print("-" * 50)

    #evaluator.print_summary()
    
    df_with_eval = evaluator.df
    # Save to same folder with _w_metrics.csv
    base_path, _ = os.path.splitext(eval_df_path)
    finish_time = time.time()
    output_path = f"{base_path}_w_metrics_langgraph_{model_name.replace(':', '_')}_{finish_time}.csv"
    df_with_eval.to_csv(output_path, index=False)
if __name__ == "__main__":
    asyncio.run(main())