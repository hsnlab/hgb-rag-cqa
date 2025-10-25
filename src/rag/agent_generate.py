import datetime
import logging
from crewai import Agent, Task, Crew
from src.rag.hf_llm_wrapper import HFLocalLLM
from src.rag.retrieval_tool import RetrievalTool


# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

# -----------------------------
# Agent-based RAG orchestration
# -----------------------------
def run_agents(question: str, context, huggingface_apikey: str):
    print(f"[{_ts()}] 🚀 Agent-based RAG pipeline started.")
    print(f"[{_ts()}] User question: {question}")

    # -----------------------------
    # Step 1: Initialize the LLM
    # -----------------------------
    llm = HFLocalLLM(context["gen"])
    print(f"[{_ts()}] LLM initialized successfully.")

    # -----------------------------
    # Step 2: Attach retrieval tool
    # -----------------------------
    retriever = context.get("retriever", None)
    if retriever:
        retrieval_tool = RetrievalTool(retriever)
        print(f"[{_ts()}] Retrieval layer available. Tool will be attached to the Retriever.")
    else:
        retrieval_tool = None
        print(f"[{_ts()}] ⚠️ No retriever found in context! Retrieval tool disabled.")

    # -----------------------------
    # Step 3: Define agents
    # -----------------------------
    print(f"[{_ts()}] Defining agents (Interpreter, Retriever, Synthesizer)...")

    interpreter_agent = Agent(
        name="Interpreter",
        role="Query Analyst",
        goal="Understand the user's question and determine what type of data or context is needed.",
        backstory="You interpret the intent of the question, decompose it if needed, and guide the retrieval agent.",
        llm=llm,
    )

    retriever_agent = Agent(
        name="Retriever",
        role="Information Retriever",
        goal="Use the retrieval tool multiple times if necessary to gather enough evidence from Neo4j and Qdrant.",
        backstory="You have access to structured and unstructured sources. Use the retrieval tool to get information dynamically.",
        llm=llm,
        tools=[retrieval_tool] if retrieval_tool else [],
    )

    synthesizer_agent = Agent(
        name="Synthesizer",
        role="Answer Synthesizer",
        goal="Combine the interpreted query and retrieved data into a clear, complete answer.",
        backstory="You merge insights from multiple agents to produce a well-structured and factual explanation.",
        llm=llm,
    )

    print(f"[{_ts()}] Agents successfully initialized.")

    # -----------------------------
    # Step 4: Define Tasks
    # -----------------------------
    print(f"[{_ts()}] Defining multi-agent tasks...")

    interpret_task = Task(
        description=f"Analyze the question: '{question}' and decide what information must be retrieved.",
        agent=interpreter_agent,
        expected_output="A clear plan of what to retrieve and which data sources may be relevant.",
    )

    retrieve_task = Task(
        description=f"Using the retrieval tool, search the knowledge base (Neo4j/Qdrant) for data relevant to: '{question}'.",
        agent=retriever_agent,
        expected_output="A list of retrieved snippets, functions, or relationships relevant to the user's query.",
    )

    synthesize_task = Task(
        description=f"Summarize the findings into a clear, concise, and informative answer to: '{question}'.",
        agent=synthesizer_agent,
        expected_output="A final, human-readable answer that connects retrieved data and reasoning.",
    )

    print(f"[{_ts()}] Tasks defined and assigned.")

    # -----------------------------
    # Step 5: Build the Crew
    # -----------------------------
    crew = Crew(
        agents=[interpreter_agent, retriever_agent, synthesizer_agent],
        tasks=[interpret_task, retrieve_task, synthesize_task],
        verbose=True,
    )

    print(f"[{_ts()}] 🧠 Launching agent crew execution...")
    result = crew.kickoff()

    print(f"\n=== Final Answer ===\n{result}\n")
    return result
