from typing import List, Dict, Any, Tuple
from .config import PipelineConfig
from .base_rag import BaseRAG
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from transformers import pipeline
from neo4j import GraphDatabase
import os, httpx

class SimpleRAG(BaseRAG):
    def __init__(self, vectorstore, neo4j_uri, neo4j_auth, llm=None, llm_model: str = None, classifier=None, classifier_model: str = None):
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

        if classifier is not None:
            self.classifier = classifier
        elif classifier_model is not None:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=classifier_model,
                device= "cpu"
            )
        else:
            raise ValueError("Either classifier or classifier_model must be provided")
        if llm is not None:
            self.llm = llm
        elif llm_model is not None:
            os.environ["OLLAMA_KEEP_ALIVE"] = "0"
            self.llm = ChatOllama(
                model=self.model_name, 
                base_url="http://localhost:11434", 
                async_client_kwargs={
                    "headers": {"Connection": "close"},
                    "timeout": 120,
                    "limits": httpx.Limits(
                        max_keepalive_connections=0, 
                        max_connections=10,           
                    ),
                },
            )
        else:
            raise ValueError("Either llm or llm_model must be provided")

    def _retrieve(self, query: str, query_type:str, config: PipelineConfig) -> Tuple[List[Document], str, List[str]]:
        """
        Simple retrieval from vectorstore.
        Builds top_nodes from Qdrant metadata if available.
        """
        docs = self.vectorstore.search(query, top_k=config.top_k)
        top_nodes = []

        for d in docs:
            meta = d.metadata or {}
            node_id = meta.get("node_id")
            top_nodes.append(node_id)

        # deduplicate
        top_nodes = list(set(top_nodes))
        return docs, top_nodes


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
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            if config.verbose:
                print(f"[WARN] LLM generation failed: {e}")
            return "I don't know."
    
    def classify_query(self, query: str) -> str:
        labels = ["general_question", "bug_report", "feature_request", "performance_issue"]
        result = self.classifier(query, labels, multi_label=False)
        return result["labels"][0]

    def _expand_query_with_llm(self, query: str, config: PipelineConfig) -> list[str]:
        """Use the existing LLM to produce a few alternative search queries."""
        queries = []
        prompt = (
            f"Generate {getattr(config, 'query_expansion_variants', 3)} concise alternative search queries "
            f"to help retrieve relevant code snippets or technical documentation related to:\n\"{query}\"\n\n"
            f"- Focus on code elements explicitly mentioned (functions, classes, APIs, variables, libraries).\n"
            f"- Keep each query under 8 words.\n"
            f"- Prefer specific identifiers over generic terms.\n"
            f"- Return each query on a new line, without numbering or punctuation."
        )
        try:
            response = self.llm.invoke(prompt)
            text = response.content.strip()
            variants = [ln.strip("-• ").strip() for ln in text.split("\n") if ln.strip()]
            variants = [v for v in variants if len(v.split()) > 1]
            queries.extend(variants)
            if config.verbose:
                print(f"[DEBUG] Query expansion generated: {variants}")
        except Exception as e:
            if config.verbose:
                print(f"[WARN] Query expansion failed: {e}")
            return [query]
        return list(dict.fromkeys(queries))

    def _get_function_names(self, node_ids: list[str]) -> list[str]:
        """Return combinedName values for FUNCTION nodes in node_ids."""
        if not node_ids:
            return []
        
        func_ids = [nid for nid in node_ids if nid.startswith("FUNCTION:")]
        if not func_ids:
            return []

        driver = GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
        func_names = []
        with driver.session() as session:
            records = session.run(
                """
                MATCH (f:FUNCTION)
                WHERE f.global_id IN $ids
                RETURN DISTINCT f.combinedName AS name
                """,
                ids=func_ids,
            )
            func_names = [r["name"] for r in records if r["name"]]
        driver.close()
        return func_names
