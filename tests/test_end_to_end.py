import pytest
import pandas as pd
import networkx as nx
from unittest.mock import patch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from langchain_core.documents import Document

from kg_rag import RepositoryRAG


@pytest.fixture
def fake_data_dict():
    return {
        "issues": pd.DataFrame([{"issue_number": 1, "issue_title": "Crash", "issue_body": "App crashes","issue_labels":["bug"],	"issue_state":"open"}]),
        "prs": pd.DataFrame([{"pr_number": 42, "pr_title": "Fix crash"}]),
        "cg_nodes": pd.DataFrame([{"combinedName": "add", "function_code": "def add(a,b): return a+b", "docstring": "adds a and b","function_location": "./", "func_id":101, "cluster_id":0}]),
        "cg_edges": pd.DataFrame([{"source_id": 101, "target_id": 101}]),
        "sg_nodes": pd.DataFrame([{"node_id": 201, "func_id": 101, "code": "def add(): pass"}]),
        "sg_edges": pd.DataFrame([{"source_id": 201, "target_id": 201, "func_id":101}]),
        "hier_1": pd.DataFrame([{"source_id": 201, "target_id": 101}]),
        "issue_to_pr_function_edges": pd.DataFrame([{"issue_number": 1, "pr_number": 42, "func_id": 101}]),
        "pr_edges": pd.DataFrame([{"pr_number": 42, "func_id_1": 101, "func_id_2": 101, "pr_title": "Fix crash"}]),
        "issue_to_pr_edges": pd.DataFrame([{"issue_number":1,"pr_number":42}]),
    }


@pytest.fixture
def rag(fake_data_dict):
    # Patch tokenizer, model, pipeline, and embeddings to avoid real model loads
    with patch("kg_rag_new.AutoTokenizer.from_pretrained"), \
         patch("kg_rag_new.AutoModelForCausalLM.from_pretrained"), \
         patch("kg_rag_new.pipeline"), \
         patch("faiss_store.HuggingFaceEmbeddings.embed_query", return_value=[0.1]*384), \
         patch("faiss_store.HuggingFaceEmbeddings.embed_documents", return_value=[[0.1]*384]):
        return RepositoryRAG(fake_data_dict)


def test_end_to_end_flow(rag):
    # Mock out FAISS store searches to return fake results
    with patch.object(rag.retriever.store, "search_with_scores") as mock_search:
        mock_search.side_effect = [
            # retrieve_code_functions
            [(Document(page_content="def add(a,b): return a+b", metadata={"func_id": 100}), 0.9)],
            # retrieve_issues
            [(Document(page_content="App crashes", metadata={"issue_number": 1}), 0.8)],
            # retrieve_prs
            [(Document(page_content="Fix crash bug", metadata={"pr_number": 10}), 0.7)],
        ]

        funcs_df = rag.retrieve_code_functions("bug")
        issues_df = rag.retrieve_issues("bug")
        prs_df = rag.retrieve_prs("bug")

    # Rerank functions
    reranked = rag.rerank_functions(funcs_df, issues_df, prs_df, top_n=5)
    assert not reranked.empty
    assert "relevance_score" in reranked.columns

    # Build filtered knowledge graph
    G = rag._filter_knowledge_graph(reranked, issues_df, prs_df)
    assert isinstance(G, nx.Graph)
    assert any(n.startswith("F_") for n in G.nodes)
    assert any(n.startswith("I_") for n in G.nodes)

    # Mock LLM generation
    with patch.object(rag, "generation_pipeline", return_value=[{"generated_text": "This is a test answer"}]):
        answer = rag._generate_answer("Why does the app crash?", G)
        assert isinstance(answer, str)
        assert "test answer" in answer
