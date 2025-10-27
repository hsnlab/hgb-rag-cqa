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

def remove_html_tags(text):
  """
  Removes HTML tags from a string and unescapes HTML entities.

  Args:
    text: The input string containing HTML.

  Returns:
    The cleaned string without HTML tags or entities.
  """
  # 1. Compile a regular expression to find all HTML tags
  # This pattern '<[^>]+>' matches '<', followed by one or more characters
  # that are NOT '>', and then matches '>'.
  tag_re = re.compile('<[^>]+>')
  
  # 2. Use re.sub() to replace all matches of the pattern with an empty string
  no_tags = tag_re.sub('', text)
  
  # 3. Use html.unescape() to convert HTML entities (like &quot;, &lt;, &amp;)
  # back into their corresponding characters (", <, &)
  cleaned_text = html.unescape(no_tags)
  
  return cleaned_text.strip()

def main():
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
    args = parser.parse_args()
    
    eval_df_path = args.eval_path
    q_limit = args.question_limit

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

    #sklearn_hier_json = pd.read_pickle("../graph/sklearn/sklearn_with_summaries.pkl")
    model_name = args.model_name
    tool = AgenticLangGraph(model_name=model_name)

    evaluator = AgenticRAGEvaluator(df=dataset, agentic_runner=tool, k_values=[3, 5, 10], context_column="golden_context")
    try:
        evaluator.evaluate(verbose=True)
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
    main()