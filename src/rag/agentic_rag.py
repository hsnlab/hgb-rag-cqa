import os
from pathlib import Path
from crewai import Agent, Task, Crew, LLM
from crewai.project import CrewBase, agent, task, crew, llm

@CrewBase
class AgenticRAG:
    """
    Multi-agent Agentic RAG Crew integrating Qdrant, Neo4j, and Evaluator components.

    Agents:
    - QdrantAgent: semantic retrieval from Qdrant
    - GraphAgent: graph traversal and neighborhood expansion via Neo4j
    - RetrievalOrchestrator: combines both retrieval results
    - ReasonerAgent: synthesizes the final grounded answer
    - EvaluatorAgent: evaluates the reasoned output for factuality & completeness
    """

    # ---------------------------------------------------
    # YAML configuration paths
    # ---------------------------------------------------
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ---------------------------------------------------
    # MCP server configuration
    # ---------------------------------------------------
    mcp_server_params = [
        {
            "url": "http://localhost:8765/mcp",  # Neo4j
            "transport": "streamable-http",
        },
        {
            "url": "http://localhost:8766/mcp",  # Qdrant
            "transport": "streamable-http",
        },
    ]

    mcp_connect_timeout = 60

    # ---------------------------------------------------
    # LLM configuration
    # ---------------------------------------------------
    @llm
    def custom_llm(self):
        """
        LLM configuration for all agents (local Ollama endpoint or similar).
        """
        return LLM(
            model="openai/mistral",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    # ---------------------------------------------------
    # AGENTS
    # ---------------------------------------------------
    @agent
    def qdrant_agent(self) -> Agent:
        tools = self.get_mcp_tools("qdrant_search")
        return Agent(config=self.agents_config["qdrant_agent"], tools=tools, verbose=True)

    @agent
    def graph_agent(self) -> Agent:
        tools = self.get_mcp_tools(
            "expand_function_neighbors",
            "expand_cfg_neighbors",
            "functions_linked_to_issues_prs",
        )
        return Agent(config=self.agents_config["graph_agent"], tools=tools, verbose=True)

    @agent
    def retrieval_orchestrator(self) -> Agent:
        return Agent(config=self.agents_config["retrieval_orchestrator"], tools=[], verbose=True)

    @agent
    def reasoner_agent(self) -> Agent:
        return Agent(config=self.agents_config["reasoner_agent"], tools=[], verbose=True)

    @agent
    def evaluator_agent(self) -> Agent:
        return Agent(config=self.agents_config["evaluator_agent"], tools=[], verbose=True)

    # ---------------------------------------------------
    # TASKS
    # ---------------------------------------------------
    @task
    def qdrant_task(self) -> Task:
        return Task(config=self.tasks_config["qdrant_task"], agent=self.qdrant_agent())

    @task
    def graph_task(self) -> Task:
        return Task(config=self.tasks_config["graph_task"], agent=self.graph_agent())

    @task
    def retrieval_orchestration_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieval_orchestration_task"],
            agent=self.retrieval_orchestrator(),
            context=[self.qdrant_task(), self.graph_task()],
        )

    @task
    def reason_task(self) -> Task:
        return Task(
            config=self.tasks_config["reason_task"],
            agent=self.reasoner_agent(),
            context=[self.retrieval_orchestration_task()],
        )

    @task
    def evaluate_task(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_task"],
            agent=self.evaluator_agent(),
            context=[self.reason_task()],
        )

    # ---------------------------------------------------
    # CREW
    # ---------------------------------------------------
    @crew
    def crew(self) -> Crew:
        """
        Full multi-agent RAG crew with retrieval, reasoning, and evaluation.
        """
        return Crew(
            agents=[
                self.qdrant_agent(),
                self.graph_agent(),
                self.retrieval_orchestrator(),
                self.reasoner_agent(),
                self.evaluator_agent(),
            ],
            tasks=[
                self.qdrant_task(),
                self.graph_task(),
                self.retrieval_orchestration_task(),
                self.reason_task(),
                self.evaluate_task(),
            ],
            verbose=True,
        )
