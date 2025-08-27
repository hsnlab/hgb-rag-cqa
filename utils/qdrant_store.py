from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import uuid

class QdrantStore:
    def __init__(self, 
                 model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 chunk_size=512, 
                 chunk_overlap=50,
                 qdrant_url="http://localhost:6333",
                 collection_name="rag_collection",
                 api_key=None):
        
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=qdrant_url, api_key=api_key)
        
        # Get embedding dimension
        dim = len(self.embeddings.embed_query("hello world"))
        
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        
        # Initialize LangChain Qdrant vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )
        
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def add_issues(self, issues_df):
        """Index issues into Qdrant with chunking."""
        docs = []
        for _, row in issues_df.iterrows():
            text = f"{row['issue_title']} {row['issue_body']}"
            for chunk in self.splitter.split_text(text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "type": "issue",
                            "issue_number": row["issue_number"],
                            "doc_id": str(uuid.uuid4())  # Add unique ID for Qdrant
                        },
                    )
                )
        if docs:
            self.vector_store.add_documents(docs)

    def add_prs(self, prs_df):
        """Index PRs into Qdrant with chunking."""
        docs = []
        for _, row in prs_df.iterrows():
            text = f"{row['pr_title']}"
            for chunk in self.splitter.split_text(text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "type": "pr",
                            "pr_number": row["pr_number"],
                            "doc_id": str(uuid.uuid4())
                        },
                    )
                )
        if docs:
            self.vector_store.add_documents(docs)

    def add_code(self, code_df):
        """Index code functions into Qdrant with chunking."""
        docs = []
        for _, row in code_df.iterrows():
            for chunk in self.splitter.split_text(row["function_code"]):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "type": "code",
                            "func_id": row["func_id"],
                            "doc_id": str(uuid.uuid4())
                        },
                    )
                )
        if docs:
            self.vector_store.add_documents(docs)

    def search(self, query, index_type="issue", top_k=5):
        """Search the Qdrant store with metadata filtering."""
        # Qdrant filtering syntax
        filter_condition = {"must": [{"key": "metadata.type", "match": {"value": index_type}}]}
        
        return self.vector_store.similarity_search(
            query, 
            k=top_k, 
            filter=filter_condition
        )

    def search_with_scores(self, query, index_type="issue", top_k=5):
        """Search the Qdrant store with scores returned."""
        filter_condition = {"must": [{"key": "metadata.type", "match": {"value": index_type}}]}
        
        return self.vector_store.similarity_search_with_score(
            query, 
            k=top_k, 
            filter=filter_condition
        )

    def delete_by_type(self, doc_type):
        """Delete all documents of a specific type."""
        filter_condition = {"must": [{"key": "metadata.type", "match": {"value": doc_type}}]}
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={"filter": filter_condition}
        )

    def get_collection_info(self):
        """Get information about the collection."""
        return self.client.get_collection(self.collection_name)

    def clear_collection(self):
        """Clear all data from the collection."""
        self.client.delete_collection(self.collection_name)
        
        # Recreate the collection
        dim = len(self.embeddings.embed_query("hello world"))
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    # Note: Qdrant doesn't have built-in save/load like FAISS
    # Data persistence is handled by the Qdrant server itself
    def backup_collection(self, backup_path):
        """Create a snapshot of the collection."""
        return self.client.create_snapshot(collection_name=self.collection_name)

    def restore_collection(self, snapshot_name):
        """Restore collection from snapshot."""
        # This would require server-side snapshot management
        # Implementation depends on your Qdrant deployment setup
        pass