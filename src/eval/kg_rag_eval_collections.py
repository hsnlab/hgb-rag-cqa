# eval_rag_collections.py
import pandas as pd
import argparse
from ast import literal_eval
import os
import sys
from tqdm import tqdm
sys.path.append("..")
from kg_rag import RepositoryRAG
from evaluation import RAGEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG model using a test DataFrame.")
    parser.add_argument(
        "eval_df_path",
        type=str,
        help="Path to the CSV file containing the evaluation dataframe."
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=[
#            "rag_collection_all-MiniLM-L6-v2_cosine",
#            "rag_collection_bart-large_cosine",
#            "rag_collection_paraphrase-mpnet-base-v2_cosine",
#            "rag_collection_codebert-base_cosine",
            "rag_collection_codet5-base_cosine",
        ],
        help="List of Qdrant collections to evaluate."
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        default = [
#            "sentence-transformers/all-MiniLM-L6-v2",       
#            "facebook/bart-large",                          
#            "sentence-transformers/paraphrase-mpnet-base-v2",
#            "microsoft/codebert-base",
            "Salesforce/codet5-base",
        ],
        help="List of Embedding models to use."
    )
    parser.add_argument(
        "--llm_model", type=str, default="mistralai/mistral-7b-instruct-v0.3",
        help="LLM model to use for generation (default: mistralai/mistral-7b-instruct-v0.3)."
    )
    

    args = parser.parse_args()
    assert len(args.collections) == len(args.embedders), "Number of collections must match number of embedders"
    eval_df_path = args.eval_df_path
    collections = args.collections
    embedders = args.embedders
    llm_model = args.llm_model


    if not os.path.isfile(eval_df_path):
        print(f"Error: File '{eval_df_path}' does not exist.")
        sys.exit(1)
        
    print(f"Loading dataset from: {eval_df_path}")
    dataset = pd.read_csv(eval_df_path)
    #dataset["edit_functions"] = dataset["edit_functions"].apply(literal_eval)

    for i in range(len(collections)):
        collection = collections[i]
        embedder = embedders[i]
        print(f"\nEvaluating collection: {collection}")
        tool = RepositoryRAG(
            data_dict={},
            qdrant_api_key="@lmafa12",
            qdrant_collection=collection,
            quantize=True,
            model_name = embedder,
            llm_model=llm_model
        )

        evaluator = RAGEvaluator(df=dataset, rag_model=tool, k_values=[3, 5, 10])

        # Use tqdm to track progress
        pbar = tqdm(range(len(dataset)), desc=f"Evaluating {collection}", unit="item")
        for idx in pbar:
            row = dataset.iloc[idx]
            evaluator.evaluate_single(idx, row)
            # Update live metrics in progress bar
            summary = evaluator.get_live_summary(idx)
            pbar.set_postfix(summary)

        # Save results
        output_path = os.path.splitext(eval_df_path)[0] + f"_{llm_model}_{collection}_metrics.csv"
        evaluator.export(output_path)
        print(f"Saved evaluation results to: {output_path}")
        evaluator.print_summary()


if __name__ == "__main__":
    main()
