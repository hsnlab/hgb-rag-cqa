import asyncio
import json
import traceback
import argparse
from src.rag.agentic_langgraph import AgenticLangGraph
from src.rag.agentic_langgraph_strict import StrictAgenticLangGraph


async def interactive_mode():
    """
    Launch an interactive shell to manually test the AgenticLangGraph.
    """
    parser = argparse.ArgumentParser(description="Run LangGraph Agentic RAG.")
    parser.add_argument(
        "--model-name", "-ml",
        default="gpt-oss:20b",
        type=str,
        help="Name of Ollama model to use for agents.",
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

    model_name = args.model_name
    pipeline_version = args.pipeline
    if pipeline_version == "strict":
        print("Initializing StrictAgenticLangGraph...")
        agent = StrictAgenticLangGraph(model_name=model_name)
    elif pipeline_version == "free":
        print("Initializing AgenticLangGraph...")
        agent = AgenticLangGraph(model_name=model_name)
    else:
        print("Pipeline version not recognized, falling back to 'free'")
        agent = AgenticLangGraph(model_name=model_name)
    

    # Ensure the graph is initialized
    await agent.setup_graph()

    print("\n[READY] Agent initialized successfully.")
    print("Type your query below (or 'exit' to quit).\n")

    while True:
        try:
            query = input("User: ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            print("\n[INFO] Processing query — please wait...\n")
            result = await agent.run_async(query)

            print(f"\n[ANSWER]: {result['answer']}")
            #print(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"\n[RETRIEVED NODES]: {result['relevant_node_ids']}")
            
            print("\n--------------------------------------------\n")

        except Exception as e:
            print("[ERROR] Exception during run:")
            traceback.print_exc()
            print("\nRestarting interactive mode...\n")


async def run_batch_tests():
    """
    Example of running automated batch tests for multiple queries.
    Useful for evaluation loops or regression testing.
    """
    queries = [
        "How does pca.fit work?",
        "What functions are related to data normalization?",
        "Which classes handle database connections?",
    ]

    agent = AgenticLangGraph(model_name="gpt-oss:20b")
    await agent.setup_graph()

    print(f"\n[RUNNING BATCH TESTS] {len(queries)} queries...\n")
    results = {}

    for q in queries:
        print(f"\n[QUERY] {q}")
        try:
            result = await agent.run_async(q)
            results[q] = result
            print(f"✅ Completed in test_langgraph for query: {q}")
        except Exception as e:
            results[q] = {"error": str(e)}
            print(f"❌ Error processing query '{q}': {e}")
            traceback.print_exc()

    print("\n[BATCH RESULTS]")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        asyncio.run(interactive_mode())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")