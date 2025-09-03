import argparse
import random
import pandas as pd
from neo4j import GraphDatabase
from code_qa_generator import CodeQAGenerator


def fetch_functions_from_neo4j(uri: str, user: str, password: str) -> list:
    """
    Fetch all FUNCTION nodes from Neo4j, returning their code bodies.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = "MATCH (f:FUNCTION) RETURN f.code AS code"
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
    random.shuffle(functions)
    selected_functions = functions[:args.num_samples]

    generator = CodeQAGenerator()
    all_results = []

    for func in selected_functions:
        qa_pairs = generator.generate(func)
        all_results.extend(qa_pairs)

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"[Done] Saved {len(all_results)} Q&A pairs to {args.output}")


if __name__ == "__main__":
    main()
