import asyncio
import json
import traceback
from src.rag.agentic_langgraph import AgenticLangGraph


async def interactive_mode():
    """
    Launch an interactive shell to manually test the AgenticLangGraph.
    """
    print("Initializing AgenticLangGraph (v1)...")
    agent = AgenticLangGraph(model_name="gpt-oss:20b")

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

            print("\n[RESULT]")
            print(json.dumps(result, indent=2, ensure_ascii=False))
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
    import sys

    mode = "interactive"
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        mode = "batch"

    try:
        if mode == "batch":
            asyncio.run(run_batch_tests())
        else:
            asyncio.run(interactive_mode())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")