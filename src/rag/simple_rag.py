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
        n = getattr(config, "query_expansion_variants", 3)
        prompt = (
            "You are a code search assistant that expands developer queries into concise, code-focused alternatives.\n\n"
            f"Generate EXACTLY {n} alternative search queries that could match code identifiers, functions, classes, "
            "variables, or technical symbols relevant to the user's intent.\n\n"
            "Follow these strict rules:\n"
            "1. Focus on actual code elements and identifiers — functions, methods, classes, constants, modules, filenames, configs, etc.\n"
            "2. Use terms already present in the user query or clear canonical equivalents. Never introduce unrelated technologies or libraries.\n"
            "3. Keep each line concise — ideally 2–6 tokens. Prefer patterns like:\n"
            "   - ClassName.method()\n"
            "   - module.function_name\n"
            "   - object.attribute\n"
            "   - VariableName or CONSTANT_NAME\n"
            "4. Avoid generic or verbose text (e.g., 'example', 'tutorial', 'how to', 'guide', 'repo', 'documentation').\n"
            "5. Do not number or bullet the lines. No quotes or backticks.\n"
            "6. Output only the alternative queries, one per line.\n\n"
            "Good outputs:\n"
            "  TransformerLayer.forward\n"
            "  config.load_yaml\n"
            "  database.connect\n"
            "  user_authenticate\n"
            "  HttpClient.send_request\n"
            "Bad outputs:\n"
            "  Example of how TransformerLayer.forward works\n"
            "  Guide for connecting to database\n"
            "  Documentation for HttpClient\n\n"
            f"User query:\n{query}\n\n"
            "Now output the alternative search queries:"
        )

        try:
            response = self.llm.invoke(prompt)
            text = (response.content or "").strip()

            raw = [ln.strip() for ln in text.splitlines() if ln.strip()]

            def _clean(s: str) -> str:
                s = s.lstrip("0123456789.-) ").strip()
                s = s.strip("`'\"")
                return s

            candidates = [_clean(s) for s in raw]

            forbid = {
                "repo", "project", "example", "examples", "tutorial", "explanation",
                "details", "how to", "guide", "overview", "documentation", "docs"
            }

            def _ok(s: str) -> bool:
                low = s.lower()
                if any(w in low for w in forbid):
                    return False
                if len(s.split()) < 1 or len(s.split()) > 6:
                    return False
                # looks code-ish
                return any(ch in s for ch in "._()/") or any(t[0].isupper() for t in s.split())

            variants = [s for s in candidates if _ok(s)]

            if len(variants) > n:
                variants = variants[:n]
            elif len(variants) < n:
                if len(query.split()) <= 6:
                    variants.append(query.strip())
                variants = variants[:n]

            deduped = list(dict.fromkeys(variants))

            if config.verbose:
                print(f"[DEBUG] Query expansion generated: {deduped}")

            return deduped or [query]

        except Exception as e:
            if config.verbose:
                print(f"[WARN] Query expansion failed: {e}")
            return [query]

    def _get_function_names(self, node_ids: list[str]) -> list[str]:
        """Return combinedName values for FUNCTION nodes in node_ids."""
        if not node_ids:
            return []
        
        func_ids = [nid for nid in node_ids if nid.startswith("FUNCTION:")]
        if not func_ids:
            return []

        driver = GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
        name_map = {}
        with driver.session() as session:
            records = session.run(
                """
                MATCH (f:FUNCTION)
                WHERE f.global_id IN $ids
                RETURN f.global_id AS id, f.combinedName AS name
                """,
                ids=func_ids,
            )
            for r in records:
              if r["name"] is not None:
                  name_map[r["id"]] = r["name"]
        driver.close()
        seen = set()
        ordered_unique_names = []
    
        for nid in node_ids:
            if nid in name_map:
                name = name_map[nid]
                if name not in seen:
                    seen.add(name)
                    ordered_unique_names.append(name)
    
        return ordered_unique_names
        
    def _get_class_names(self, node_ids: list[str]) -> list[str]:
        """Return class names for certain node types in node_ids, preserving input order and removing duplicates."""
        if not node_ids:
            return []
    
        driver = GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
  
        name_map = {}
    
        func_ids = [nid for nid in node_ids if nid.startswith("FUNCTION:")]
        if func_ids:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (f:FUNCTION)
                    WHERE f.global_id IN $ids
                    RETURN f.global_id AS id, f.class_name AS name
                    """,
                    ids=func_ids,
                )
                for r in result:
                    if r["name"] is not None:
                        name_map[r["id"]] = r["name"]
    
        class_config_ids = [nid for nid in node_ids
                            if nid.startswith("CLASS:") or nid.startswith("CONFIG:")]
        if class_config_ids:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (n)
                    WHERE n.global_id IN $ids
                    RETURN n.global_id AS id, n.name AS name
                    """,
                    ids=class_config_ids,
                )
                for r in result:
                    if r["name"] is not None:
                        name_map[r["id"]] = r["name"]
    
        driver.close()
  
  
        seen = set()
        ordered_unique_names = []
        
        for nid in node_ids:
            if nid in name_map:
                name = name_map[nid]
                if name not in seen:
                    seen.add(name)
                    ordered_unique_names.append(name)
    
        return ordered_unique_names
    
