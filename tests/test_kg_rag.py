import pytest
import pandas as pd
import networkx as nx
from unittest.mock import patch, MagicMock
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from kg_rag_new import RepositoryRAG
from langchain_core.documents import Document


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


def test_results_to_df_creates_dataframe(rag):
    doc = Document(page_content="hello", metadata={"func_id": 101})
    results = [(doc, 0.8)]
    df = rag._results_to_df(results, key_field="func_id")
    assert "similarity" in df.columns
    assert "text" in df.columns
    assert df.iloc[0]["func_id"] == 101


def test_rerank_functions_scores_functions(rag):
    funcs_df = pd.DataFrame([{"func_id": 101, "similarity": 0.5}])
    issues_df = pd.DataFrame([{"issue_number": 1, "similarity": 0.6}])
    prs_df = pd.DataFrame([{"pr_number": 42, "similarity": 0.7}])
    scored = rag.rerank_functions(funcs_df, issues_df, prs_df, top_n=5)
    assert "relevance_score" in scored.columns
    assert not scored.empty


def test_filter_knowledge_graph_builds_graph(rag):
    funcs_df = pd.DataFrame([{"func_id": 101, "similarity": 0.5}])
    issues_df = pd.DataFrame([{"issue_number": 1, "similarity": 0.6}])
    prs_df = pd.DataFrame([{"pr_number": 42, "similarity": 0.7}])
    G = rag._filter_knowledge_graph(funcs_df, issues_df, prs_df)
    assert isinstance(G, nx.Graph)
    assert any(n.startswith("F_") for n in G.nodes)
    assert any(n.startswith("I_") for n in G.nodes)


def test_generate_answer_returns_text(rag):
    G = nx.Graph()
    G.add_node("F_101", label="utils.add")
    with patch.object(rag, "generation_pipeline", return_value=[{"generated_text": "Answer"}]):
        result = rag._generate_answer("What does add do?", G)
        assert isinstance(result, str)
        assert "Answer" in result
