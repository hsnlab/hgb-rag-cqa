import os
from crewai import Agent, Task, Crew
from typing import Optional
import traceback

from dotenv import load_dotenv

from src.rag.hf_llm_wrapper import HFLocalLLM, clean_final_output
from src.utils.qdrant_store import QdrantStore


def run_agents(question: str, context: dict, huggingface_apikey: Optional[str] = None) -> str:
    """
    Runs an agentic multi-step reasoning pipeline using CrewAI with:
    1. Interpreter (decomposes task)
    2. Retriever (fetches documentation/code)
    3. Summarizer (writes final explanation)
    """

    llm = HFLocalLLM(context["gen"])

    # --- Loading Qdrant API key ---
    try:
        with open("_/drant_api_key.txt", "r", encoding="utf-8") as f:
            qdrant_api_key = f.read().strip()
            print("✅ Qdrant API key loaded from file.")
    except FileNotFoundError:
        qdrant_api_key = None
        print("⚠️ Qdrant API key file not found — proceeding without API key.")

    retrieval_layer = QdrantStore(
        qdrant_url="http://localhost:6333",
        api_key=qdrant_api_key,
        neo4j_uri="bolt://localhost:7687",
        neo4j_auth=("neo4j", "password")
    )

    # ==============================================================
    # 1️⃣ Interpreter agent – understands and decomposes the task
    # ==============================================================

    interpreter = Agent(
        role="Interpreter",
        goal="Understand and decompose the user's question into logical subtasks.",
        backstory="Analyzes the question and determines what information must be retrieved.",
        allow_delegation=True,
        verbose=False,
        llm=llm,
    )

    interpret_task = Task(
        description=f"Interpret the user's question: '{question}'. Identify what data should be retrieved.",
        expected_output="A plan describing what the Retriever should search for.",
        agent=interpreter,
    )

    # ==============================================================
    # 2️⃣ Retriever agent – database query (Qdrant + Neo4j)
    # ==============================================================

    print("\n[Retrieval] Searching databases for relevant context...")

    try:
        contextual_data = retrieval_layer.search(question)
        if not contextual_data or len(contextual_data) == 0:
            print("⚠️ No relevant results found in Qdrant/Neo4j.")
            contextual_data = []  # ✅ Important: ensure it's a list
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        contextual_data = []  # ✅ Also a list fallback

    retriever = Agent(
        role="Retriever",
        goal="Find relevant code, documentation, and explanations from Qdrant and Neo4j.",
        backstory="You specialize in retrieving the most relevant technical context to support answers.",
        allow_delegation=False,
        verbose=False,
        llm=llm,
    )

    retrieve_task = Task(
        description=f"Retrieve information related to: '{question}'.",
        expected_output="Relevant database context retrieved from Qdrant and Neo4j.",
        agent=retriever,
        context=contextual_data
    )

    # ==============================================================
    # 3️⃣ Summarizer agent – final answer generation
    # ==============================================================

    summarizer = Agent(
        role="Summarizer",
        goal="Use the retrieved context to generate a clear and factually correct explanation.",
        backstory="Combines the retrieved evidence into a coherent, accurate, and human-readable answer.",
        allow_delegation=False,
        verbose=False,
        llm=llm,
    )

    summarize_task = Task(
        description=f"Based on the retrieved information, explain clearly and factually:\n\n{question}",
        expected_output="A factual, database-backed explanation.",
        agent=summarizer,
        context=contextual_data
    )

    # ==============================================================
    # Crew assembly and execution
    # ==============================================================

    crew = Crew(
        agents=[interpreter, retriever, summarizer],
        tasks=[interpret_task, retrieve_task, summarize_task],
        verbose=True,
    )

    # --- Execute pipeline ---
    try:
        result = crew.kickoff()

        if hasattr(result, "final_output"):
            final_answer = result.final_output
        elif isinstance(result, dict) and "final_output" in result:
            final_answer = result["final_output"]
        else:
            final_answer = str(result)

        final_answer = clean_final_output(final_answer)
        return final_answer

    except Exception:
        print("❌ Error during Crew execution:")
        print(traceback.format_exc())
        return "[Crew execution error]"
