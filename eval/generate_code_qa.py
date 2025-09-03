import argparse
import random
import pandas as pd
from neo4j import GraphDatabase
from qa_generator import CodeQAGenerator
from huggingface_hub import login
from datasets import Dataset

def fetch_functions_from_neo4j(uri: str, user: str, password: str) -> list:
    """
    Fetch all FUNCTION nodes from Neo4j, returning their code bodies.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = "MATCH (f:FUNCTION) RETURN f.function_code AS code"
    with driver.session() as session:
        result = session.run(query)
        functions = [record["code"] for record in result if record["code"]]
    driver.close()
    return functions


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs for function code stored in Neo4j.")
    parser.add_argument(
        "-n", "--num-samples", type=int, required=True,
        help="Number of functions to sample from Neo4j."
    )
    parser.add_argument(
        "--uri", type=str, default="bolt://localhost:7687",
        help="Neo4j bolt URI (default: bolt://localhost:7687)."
    )
    parser.add_argument(
        "--user", type=str, default="neo4j",
        help="Neo4j username (default: neo4j)."
    )
    parser.add_argument(
        "--password", type=str, required=True,
        help="Neo4j password."
    )
    parser.add_argument(
        "--output", type=str, default="qa_output.csv",
        help="Output CSV filename (default: qa_output.csv)."
    )
    args = parser.parse_args()

    print("[Neo4j] Fetching functions...")
    functions = fetch_functions_from_neo4j(args.uri, args.user, args.password)
    if not functions:
        print("No functions found in Neo4j.")
        return

    print(f"[Neo4j] Retrieved {len(functions)} functions.")
    #random.shuffle(functions)
    #selected_functions = functions[:args.num_samples]
    dataset = Dataset.from_dict({"function_code": functions})
    dataset = dataset.shuffle(seed=42).select(range(args.num_samples))
    print(f"[Dataset] Loaded the functions into a Dataset (len:{len(dataset)})")
    # Read hf token
    with open ('../_/hf_token.txt', 'r') as f:
        hf_token = f.read().strip()
    login(hf_token)

    generator = CodeQAGenerator(quantize=True)
    #all_results = []

    #for func in selected_functions:
    #    qa_pairs = generator.generate(func)
    #    all_results.extend(qa_pairs)
    #df = pd.DataFrame(all_results)
    #df.to_csv(args.output, index=False)
    
    def generate_for_batch(batch):
        qa_results = generator.generate_batch(batch["function_code"])  # list of dicts

        # unpack dicts into column lists (this is not regenerating, just restructuring)
        categories = [r["category"] for r in qa_results]
        questions  = [r["question"] for r in qa_results]
        answers    = [r["answer"] for r in qa_results]
        funcs      = [r["function_code"] for r in qa_results]

        return {
            "category": categories,
            "question": questions,
            "answer": answers,
            "function_code": funcs,
        }
    qa_dataset = dataset.map(
        generate_for_batch,
        batched=True,
        batch_size=4  # how many functions per batch
    )
    qa_dataset.to_csv(args.output)
    print(f"[Done] Saved {len(qa_dataset)} Q&A pairs to {args.output}")


if __name__ == "__main__":
    main()
