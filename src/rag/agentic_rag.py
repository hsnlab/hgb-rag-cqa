import os
from crewai import Agent, Task, Crew
from crewai.project import CrewBase, agent, task, crew

@CrewBase
class AgenticRAG:
    """
    Two-agent RAG Crew integrating Neo4j and Qdrant MCP servers via CrewBase.

    - RetrieverAgent: hybrid retrieval using Qdrant + Neo4j
    - ReasonerAgent: synthesizes final answers
    """

    # ---------------------------------------------------
    # YAML configuration paths (relative to project root)
    # ---------------------------------------------------
    agents_config_path = os.path.join("src", "rag", "agent_config", "agents.yaml")
    tasks_config_path = os.path.join("src", "rag", "agent_config", "tasks.yaml")

    # ---------------------------------------------------
    # MCP server configuration
    # ---------------------------------------------------
    mcp_server_params = [
        {
            "url": "http://localhost:8765",  
            "transport": "streamable-http",
        },
        {
            "url": "http://localhost:8766", 
            "transport": "streamable-http",
        },
    ]

    mcp_connect_timeout = 60 

    # ---------------------------------------------------
    # Agents
    # ---------------------------------------------------
    @agent
    def retriever_agent(self) -> Agent:
        """
        Agent that performs hybrid retrieval from Qdrant and Neo4j MCP tools.
        """
        tools = self.get_mcp_tools(
            "qdrant_search",
            "expand_function_neighbors",
            "expand_cfg_neighbors",
            "functions_linked_to_issues_prs",
        )

        return Agent(
            config=self.agents_config["retriever_agent"],
            tools=tools,
            verbose=True,
        )

    @agent
    def reasoner_agent(self) -> Agent:
        """
        Agent that synthesizes answers based on retrieved information.
        """
        return Agent(
            config=self.agents_config["reasoner_agent"],
            tools=[],
            verbose=True,
        )

    # ---------------------------------------------------
    # Tasks
    # ---------------------------------------------------
    @task
    def retrieve_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_task"],
            agent=self.retriever_agent,
        )

    @task
    def reason_task(self) -> Task:
        return Task(
            config=self.tasks_config["reason_task"],
            agent=self.reasoner_agent,
        )

    # ---------------------------------------------------
    # Crew
    # ---------------------------------------------------
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )
