from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from neo4j import GraphDatabase
import uuid

class QdrantStore:
    def __init__(self, 
                 model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 chunk_size=512, 
                 chunk_overlap=50,
                 qdrant_url="http://localhost:6333",
                 collection_name="rag_collection",
                 api_key=None,
                 neo4j_uri=None,
                 neo4j_auth=None):
        
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=qdrant_url, api_key=api_key)
        print("Successfully connected to Qdrant")
        
        # Get embedding dimension
        dim = len(self.embeddings.embed_query("hello world"))
        
        # Create collection if it doesn't exist
        if self.client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' already exists.")
        else:
            print(f"Creating collection '{collection_name}'.")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        
        # Initialize LangChain Qdrant vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )
        
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Neo4j connection details
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth

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

    def add_from_neo4j(self, node_label="FUNCTION", text_property="combinedName", id_property="ID", metadata_type="function"):
        """Fetch nodes from Neo4j, embed them, and store in Qdrant."""
        if not self.neo4j_uri or not self.neo4j_auth:
            raise ValueError("Neo4j URI and authentication must be provided")

        driver = GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
        docs = []

        with driver.session() as session:
            query = f"MATCH (n:{node_label}) RETURN n.{id_property} as id, n.{text_property} as text"
            result = session.run(query)
            seen = set()
            for record in result:
                node_id = record["id"]
                text = record["text"]
                
                if isinstance(text, list):
                    text = ", ".join(str(item) for item in text)
                if not text or not text.strip():
                    continue
                # Split text if needed
                for chunk in self.splitter.split_text(text):
                    if not chunk.strip():
                        continue
                    if chunk in seen:
                        continue
                    seen.add(chunk)
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "type": metadata_type,
                                "node_id": node_id,
                                "doc_id": str(uuid.uuid4())
                            }
                        )
                    )
                    print(f"Prepared document for node ID {node_id}, with metadata: {metadata_type}, chunk length: {len(chunk)}")

        if docs:
            self.vector_store.add_documents(docs)
        driver.close()

    def search(self, query, top_k=5, filter=None):
        """Search the Qdrant store with metadata filtering."""
        return self.vector_store.similarity_search(
            query, 
            k=top_k, 
            filter=filter
        )

    def search_with_scores(self, query, top_k=5, filter=None):
        """Search the Qdrant store with scores returned."""        
        return self.vector_store.similarity_search_with_score(
            query, 
            k=top_k, 
            filter=filter
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

    def backup_collection(self):
        """Create a snapshot of the collection."""
        return self.client.create_snapshot(collection_name=self.collection_name)
