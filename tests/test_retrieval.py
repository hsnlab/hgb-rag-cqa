import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from retrieval import Retrieval
from langchain_core.documents import Document


@pytest.fixture
def retriever():
    return Retrieval(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_index_data_calls_faiss_methods(retriever):
    with patch.object(retriever.store, "add_issues") as mock_issues, \
         patch.object(retriever.store, "add_prs") as mock_prs, \
         patch.object(retriever.store, "add_code") as mock_code:
        df = pd.DataFrame([{"dummy": 1}])
        retriever.index_data(df, df, df)
        mock_issues.assert_called_once()
        mock_prs.assert_called_once()
        mock_code.assert_called_once()


def test_retrieve_transforms_documents(retriever):
    fake_doc = Document(page_content="some text", metadata={"type": "issue", "issue_number": 1})
    with patch.object(retriever.store, "search", return_value=[fake_doc]):
        results = retriever.retrieve("bug", source="issue")
        assert isinstance(results, list)
        assert results[0]["text"] == "some text"
        assert results[0]["issue_number"] == 1
