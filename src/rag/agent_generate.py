from datetime import datetime
from email import message
import os
from crewai import Agent, Task, Crew
from typing import Optional
import traceback

from src.rag.hf_llm_wrapper import HFLocalLLM, clean_final_output
from src.utils.retrieval import KnowledgeGraphRetriever

def log_event(message: str):
    """Simple timestamped console logger."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def run_agents(question: str, context: dict, huggingface_apikey: Optional[str] = None) -> str:
    """
    Runs an agentic multi-step reasoning pipeline using CrewAI with:
    1. Interpreter (decomposes task)
    2. Retriever (fetches documentation/code)
    3. Summarizer (writes final explanation)
    """

    llm = HFLocalLLM(context["gen"])

    log_event("🚀 Agent-based RAG pipeline started.")
    log_event(f"User question: {question}")

    # === Initialize LLM shared by all agents ===
    gen_pipeline = context.get("gen")
    llm = HFLocalLLM(gen_pipeline)
    log_event("LLM initialized successfully.")

    # === Initialize the retrieval layer ===
    try:
        neo4j_cfg = context.get("neo4j", {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password"
        })

        vectorstore = context.get("qdrant_vectorstore") or context.get("vectorstore") or context.get("repo_rag")
        retrieval = KnowledgeGraphRetriever(
            vector_store=vectorstore,
            neo4j_url=neo4j_cfg["url"],
            neo4j_username=neo4j_cfg["username"],
            neo4j_password=neo4j_cfg["password"],
        )
        log_event("Retrieval layer successfully initialized.")
        log_event("Starting data retrieval...")

        retrieved_docs, query_type = retrieval.retrieve(question)

        if not retrieved_docs:
            context_text = "[No relevant results found in Qdrant/Neo4j.]"
            log_event("⚠️ No relevant results found in databases.")
        else:
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            log_event(f"✅ Retrieved {len(retrieved_docs)} context chunks (query type: {query_type}).")
            log_event(f"📚 Context sample:\n{context_text[:400]}...\n")
    except Exception as e:
        log_event(f"❌ Retrieval error: {e}")
        log_event(traceback.format_exc())
        context_text = "[Retrieval error — proceeding without external context.]"


    # ==============================================================
    # 1️⃣ Interpreter agent – understands and decomposes the task
    # ==============================================================
    
    log_event("Defining agents (Interpreter, Retriever, Summarizer)...")
    log_event("Defining multi-agent tasks...")

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
    )

    log_event("Agents successfully initialized.")
    log_event("Tasks defined and assigned.")

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
        log_event("🧠 Launching agent crew execution...")
        result = crew.kickoff()

        if hasattr(result, "final_output"):
            final_answer = result.final_output
        elif isinstance(result, dict) and "final_output" in result:
            final_answer = result["final_output"]
        else:
            final_answer = str(result)

        final_answer = clean_final_output(final_answer)
        log_event("✅ Crew execution completed successfully.")
        log_event("=== Final Answer ===")
        return final_answer

    except Exception:
        log_event("❌ Error during Crew execution:")
        log_event(traceback.format_exc())
        return "[Crew execution error]"
