from src.rag.agentic_rag import AgenticRAG

def main():
    rag = AgenticRAG()
    crew = rag.crew()
    try:
        while True:
            question = input("\nPlease enter your question (Ctrl+C to exit): ").strip()
            if not question:
                continue

            print("\nRunning Agentic RAG pipeline...")
            result = crew.kickoff(inputs={"query": question})

            print("\n=== Final Answer ===")
            print(result)

    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting. Goodbye!")

if __name__ == "__main__":
    main()
