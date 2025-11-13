import datetime
import logging
from crewai import Agent, Task, Crew
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from src.rag.hf_llm_wrapper import HFLocalLLM


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

    # =========================================================
    # Step 2: Attach MCP Qdrant tools to existing CrewAI agents
    # =========================================================
    print("🔌 Connecting to external MCP Qdrant server @ http://localhost:8001/mcp")
    mcp = MCPServerAdapter({
        "transport": "sse",
        "url": "http://localhost:8001/mcp",
        "headers": {"accept": "text/event-stream"}
    })
    mcp_tools = mcp.get_tools()

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
        goal="Query the Qdrant vector DB through MCP based on the Interpreter plan.",
        backstory="Finds multiple relevant chunks via MCP Qdrant tools.",
        llm=llm,
        tools=mcp_tools,
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
        expected_output="Plan for retrieval steps & query structure.",
    )

    retrieve_task = Task(
        description=f"Use MCP Qdrant tools to retrieve all relevant information about: '{question}'",
        agent=retriever_agent,
        expected_output="Retrieved chunks / metadata useful to answer."
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
