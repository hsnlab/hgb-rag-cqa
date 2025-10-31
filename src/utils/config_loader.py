import json
from huggingface_hub import login
from src.rag.config import PipelineConfig

def load_all_configs(rag_config_name:str="rag_config.json") -> tuple[dict, dict, dict, PipelineConfig]:
    """Load HF token, Qdrant, Neo4j, and RAG configs from the ./_/ directory."""
    # --- Hugging Face token
    with open("./_/hf_token.txt", "r") as f:
        hf_token = f.read().strip()
    login(hf_token)

    # --- Qdrant config
    with open("./_/qdrant_config.json", "r") as f:
        qdrant_config = json.load(f)

    # --- Neo4j config
    with open("./_/neo4j_config.json", "r") as f:
        neo4j_config = json.load(f)

    # --- RAG config
    with open(f"./_/{rag_config_name}", "r") as f:
        rag_json = json.load(f)
    rag_config = PipelineConfig(**rag_json)

    return qdrant_config, neo4j_config, rag_json, rag_config
