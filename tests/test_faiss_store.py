import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from faiss_store import FaissStore
from langchain_core.documents import Document


@pytest.fixture
def sample_issues_df():
    return pd.DataFrame([{
        "issue_number": 1,
        "issue_title": "Bug in code",
        "issue_body": "Something is broken"
    }])


@pytest.fixture
def sample_prs_df():
    return pd.DataFrame([{
        "pr_number": 42,
        "pr_title": "Fix bug in code"
    }])


@pytest.fixture
def sample_code_df():
    return pd.DataFrame([{
        "func_id": 101,
        "function_code": "def add(a, b): return a+b"
    }])


def test_init_creates_embeddings_and_index():
    store = FaissStore()
    assert store.embeddings is not None
    assert store.index is not None
    assert store.splitter is not None


def test_add_issues_adds_documents(sample_issues_df):
    store = FaissStore()
    with patch.object(store.index, "add_documents") as mock_add:
        store.add_issues(sample_issues_df)
        mock_add.assert_called_once()
        docs = mock_add.call_args[0][0]
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].metadata["type"] == "issue"


def test_add_prs_adds_documents(sample_prs_df):
    store = FaissStore()
    with patch.object(store.index, "add_documents") as mock_add:
        store.add_prs(sample_prs_df)
        mock_add.assert_called_once()
        docs = mock_add.call_args[0][0]
        assert docs[0].metadata["type"] == "pr"


def test_add_code_adds_documents(sample_code_df):
    store = FaissStore()
    with patch.object(store.index, "add_documents") as mock_add:
        store.add_code(sample_code_df)
        mock_add.assert_called_once()
        docs = mock_add.call_args[0][0]
        assert docs[0].metadata["type"] == "code"


def test_search_filters_by_type():
    store = FaissStore()
    fake_doc = Document(page_content="test", metadata={"type": "issue"})
    with patch.object(store.index, "similarity_search", return_value=[fake_doc]) as mock_search:
        results = store.search("query", index_type="issue")
        assert results[0].metadata["type"] == "issue"
        mock_search.assert_called_once()


def test_search_with_scores_returns_tuples():
    store = FaissStore()
    fake_doc = Document(page_content="test", metadata={"type": "issue"})
    with patch.object(store.index, "similarity_search_with_score", return_value=[(fake_doc, 0.9)]):
        results = store.search_with_scores("query", index_type="issue")
        assert isinstance(results[0][0], Document)
        assert isinstance(results[0][1], float)


