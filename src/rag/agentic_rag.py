from crewai import Agent, Crew, Task
from mcp.client import MCPClient


class AgenticRAG:
    """
    Object-oriented wrapper around a Two-Agent RAG Crew.

    - RetrieverAgent: queries Qdrant + Neo4j MCP tools
    - ReasonerAgent: synthesizes answers from retrieved docs
    """

    def __init__(self, qdrant_url="http://localhost:8000", neo4j_url="http://localhost:8001"):
        # --------------------------
        # MCP Clients (tool handles)
        # --------------------------
        self.qdrant_client = MCPClient("qdrant-store", qdrant_url)
        self.neo4j_client = MCPClient("neo4j-retriever", neo4j_url)

        # Tools exposed from MCP
        self.qdrant_search = self.qdrant_client.get_tool("qdrant_search")
        self.expand_function_neighbors = self.neo4j_client.get_tool("expand_function_neighbors")
        self.expand_cfg_neighbors = self.neo4j_client.get_tool("expand_cfg_neighbors")
        self.functions_linked_to_issues_prs = self.neo4j_client.get_tool("functions_linked_to_issues_prs")

        # --------------------------
        # Agents
        # --------------------------
        self.retriever = Agent(
            role="RetrieverAgent",
            goal="Retrieve the most relevant knowledge for the query using Qdrant and Neo4j MCP tools.",
            backstory=(
                "You are responsible for hybrid retrieval. "
                "Classify the query type (general question, bug report, feature request, performance issue). "
                "Use Qdrant for semantic vector search. Use Neo4j to expand related nodes. "
                "Return the retrieved documents in structured JSON (with content + metadata)."
            ),
            tools=[
                self.qdrant_search,
                self.expand_function_neighbors,
                self.expand_cfg_neighbors,
                self.functions_linked_to_issues_prs,
            ],
            verbose=True,
        )

        self.reasoner = Agent(
            role="ReasonerAgent",
            goal="Read retrieved documents and synthesize a clear, factual answer to the user query.",
            backstory=(
                "You are the reasoning brain. You take all retrieved docs and form a coherent response. "
                "You avoid hallucinations by sticking to retrieved content. "
                "If retrieval returns nothing useful, you say so explicitly."
            ),
            tools=[],  # reasoning only
            verbose=True,
        )

        # --------------------------
        # Tasks
        # --------------------------
        self.retrieval_task = Task(
            description=(
                "Given a user query, classify its type and retrieve relevant documents. "
                "Use Qdrant search for semantic similarity. "
                "Use Neo4j tools if expansion is needed. "
                "Return results as structured JSON with 'docs' and 'query_type'."
            ),
            expected_output="A JSON object with retrieved documents and query classification.",
            agent=self.retriever,
        )

        self.reasoning_task = Task(
            description=(
                "Take the retrieved documents and synthesize an answer. "
                "Summarize the findings clearly, cite metadata where possible. "
                "If no relevant documents are retrieved, respond that no information was found."
            ),
            expected_output="A natural language answer to the user query.",
            agent=self.reasoner,
        )

        # --------------------------
        # Crew
        # --------------------------
        self.crew = Crew(
            agents=[self.retriever, self.reasoner],
            tasks=[self.retrieval_task, self.reasoning_task],
            verbose=True,
        )

    def run(self, query: str) -> str:
        """
        Run the Two-Agent RAG pipeline for a given query.

        Args:
            query (str): The user query.

        Returns:
            str: Final synthesized answer.
        """
        return self.crew.kickoff(inputs={"query": query})
