from typing import List, Dict, Any
from src.rag.config import PipelineConfig
from src.rag.base_rag import BaseRAG
from langchain_core.documents import Document
from transformers import pipeline
from neo4j import GraphDatabase

class SimpleRAG:
    def __init__(self, vectorstore, neo4j_uri, neo4j_auth, llm=None, llm_model: str = None):
        """
        Args:
            vectorstore: Vector store backend
            neo4j_uri, neo4j_auth: KG connection info
            llm: (optional) pre-initialized HuggingFace pipeline or model wrapper
            llm_model: (optional) model name string, used if llm is None
        """
        self.vectorstore = vectorstore
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth

        if llm is not None:
            self.llm = llm
        elif llm_model is not None:
            self.llm = pipeline("text-generation", model=llm_model, device=0)
        else:
            raise ValueError("Either llm or llm_model must be provided")

    def _retrieve(self, query: str, config: PipelineConfig):
        docs = self.vectorstore.search(query, top_k=config.top_k)
        return docs, "default"

    def _enrich(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Enrich retrieved docs with Neo4j data (function names, issue IDs, PR IDs),
        and return them as a dictionary.
        """
        function_names = set()
        issue_ids = set()
        pr_ids = set()

        driver = GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
        with driver.session() as session:
            for doc in docs:
                meta = doc.metadata
                doc_type = meta.get("type", "").lower()
                node_id = meta.get("node_id")

                if not node_id:
                    continue

                if "function" in doc_type:
                    query = """
                        MATCH (n:FUNCTION)
                        WHERE n.ID = $node_id
                        RETURN n.combinedName AS name
                    """
                    result = session.run(query, node_id=node_id)
                    for record in result:
                        if record["name"]:
                            function_names.add(record["name"])

                elif "issue" in doc_type:
                    query = """
                        MATCH (n:ISSUE)
                        WHERE n.ID = $node_id
                        RETURN n.ID AS id
                    """
                    result = session.run(query, node_id=node_id)
                    for record in result:
                        if record["id"]:
                            issue_ids.add(str(record["id"]))

                elif "pr" in doc_type:
                    query = """
                        MATCH (n:PR)
                        WHERE n.ID = $node_id
                        RETURN n.ID AS id
                    """
                    result = session.run(query, node_id=node_id)
                    for record in result:
                        if record["id"]:
                            pr_ids.add(str(record["id"]))

        return {
            "functions": sorted(function_names),
            "issues": sorted(issue_ids),
            "prs": sorted(pr_ids),
        }

    def _generate_answer_from_docs(self, question: str, docs: List[Document], config: PipelineConfig) -> str:
        context = "\n\n".join([f"- {d.page_content}" for d in docs])
        prompt = f"""<s>[INST] You are a helpful assistant.
Use the context below to answer the question. If unsure, say you don’t know.

### Question
{question}

### Context
{context}

### Answer
[/INST]"""

        response = self.generation_pipeline(prompt, max_new_tokens=config.llm_max_tokens, return_full_text=False)
        return response[0]["generated_text"].strip()
