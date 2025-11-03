from src.rag.repo_rag import RepositoryRAG
from src.rag.config import PipelineConfig
from src.utils.qdrant_store import QdrantStore
from src.utils.retrieval import KnowledgeGraphRetriever
from src.utils.deduplicator import Deduplicator
from src.utils.reranker import Reranker
from typing import Tuple
import os, httpx
from langchain_ollama import ChatOllama
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


from huggingface_hub import login

def build_llm(llm_model: str):
    """
    Build an Ollama-based LLM client for generation.
    Supports models served by the local Ollama daemon, e.g. mistral:7b-instruct.
    """
    if not llm_model:
        raise ValueError("llm_model must be provided (e.g., mistral:7b-instruct)")

    os.environ["OLLAMA_KEEP_ALIVE"] = "0"
    llm = ChatOllama(
        model=llm_model, 
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
    return llm


def search(rag, config):
        try:
            while True:
                question = input("\nPlease enter your question (Ctrl+C to exit): ").strip()
                if not question:
                    continue

                print("\nRunning RAG pipeline...")  
                result = rag.run(question, config)

                answer = result["answer"]
                top_functions = result.get("top_functions", [])
                top_nodes = result.get("top_nodes", {})
                query_type = result.get("query_type", "default")

                print(f"\nYour query was classified as type: {query_type}")
                print(f"\nRetrieved {len(result['top_docs'])} documents.")
                if top_functions:
                    print(f"\nTop functions: {top_functions}")
                if top_nodes:
                    print(f"\nTop nodes retrieved (neo4j global ids): {top_nodes}")
                print("\nAnswer:", answer)

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting search. Goodbye!")

def main():
    token_path ="./_/hf_token.txt"
    with open(token_path, "r") as f:
        huggingface_apikey = f.read().strip()

    login(huggingface_apikey)
    
    qdrant_config_path = "_/qdrant_config.json"
    try:
        with open(qdrant_config_path, "r") as f:
            qdrant_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Qdrant config file not found at {qdrant_config_path}")
    except json.JSONDecodeError:
        print(f"Error: Qdrant config file is not valid JSON: {qdrant_config_path}")
    
    neo4j_config_path = "_/neo4j_config.json"
    try:
        with open(neo4j_config_path, "r") as f:
            neo4j_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Neo4j config file not found at {neo4j_config_path}")
    except json.JSONDecodeError:
        print(f"Error: Neo4j config file is not valid JSON: {neo4j_config_path}")
        
    rag_config_path = "_/rag_config.json"
    try:
        with open(rag_config_path, "r") as f:
            rag_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: RAG config file not found at {rag_config_path}")
    except json.JSONDecodeError:
        print(f"Error: RAG config file is not valid JSON: {rag_config_path}")
    
    # Configuration
    config = PipelineConfig(**rag_config)
    
    # Instantiate backend components
    vectorstore = QdrantStore(
            model_name=qdrant_config["model_name"],
            qdrant_url=qdrant_config["url"],
            collection_name=qdrant_config["collection"],
            api_key=qdrant_config["api_key"],
            distance_type=qdrant_config["distance"]
        )
        
    # Build LLM
    llm_model = "mistral:7b"
    gen = build_llm(llm_model)

    kg_retriever = KnowledgeGraphRetriever(vector_store=vectorstore, neo4j_url =neo4j_config["url"], neo4j_username=neo4j_config["user"], neo4j_password =neo4j_config["password"], database= "neo4j")
    deduplicator = Deduplicator(embedder=vectorstore.embeddings)
    reranker = Reranker()

    
    # Instantiate both RAG variants

    repo_rag = RepositoryRAG(
        vectorstore,
        retriever=kg_retriever,
        deduplicator=deduplicator,
        reranker=reranker,
        llm=gen,
        neo4j_auth=(neo4j_config["user"], neo4j_config["password"]),
        neo4j_uri=neo4j_config["url"],
        classifier_model = "facebook/bart-large-mnli"
    )

    # Start interactive search
    search(repo_rag, config)

if __name__ == "__main__":
    main()