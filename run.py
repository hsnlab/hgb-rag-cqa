from src.rag.repo_rag import RepositoryRAG
from src.rag.config import PipelineConfig
from src.utils.qdrant_store import QdrantStore
from src.utils.retrieval import KnowledgeGraphRetriever
from src.utils.deduplicator import Deduplicator
from src.utils.reranker import Reranker
from typing import Tuple
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


from huggingface_hub import login

def build_llm(
    llm_model: str,
    quantize: bool = False,
    use_4bit: bool = True,
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    use_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, pipeline]:
    """
    Build (optionally quantized) LLM + tokenizer + generation pipeline.

    Args:
      llm_model: HF model id
      quantize: whether to load a quantized model
      use_4bit: if quantize=True, prefer 4-bit (nf4) quantization. If False and quantize=True, will try 8-bit
      bnb_4bit_*: bitsandbytes config options for 4-bit
      use_8bit: explicit 8-bit flag (overrides use_4bit when quantize=True)

    Returns:
      (model, tokenizer, gen_pipeline)
    """

    # tokenizer (safe to always load)
    tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # decide device availability
    has_cuda = torch.cuda.is_available()
    if quantize and not has_cuda:
        raise RuntimeError("Quantization (bitsandbytes) requires CUDA. Set quantize=False or run on GPU.")

    model = None

    if quantize:
        # prefer explicit 4-bit if use_4bit True and use_8bit False
        if use_4bit and not use_8bit:
            # 4-bit config using BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            )
        
        else:
            # fallback to 8-bit (older but widely supported)
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16
        )

    else:
        # Full precision or mixed precision (let transformers pick optimal device_map)
        # If CUDA present, we request float16 for speed; otherwise default dtype.
        if has_cuda:
            model = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto", torch_dtype=torch.float16)
        else:
            # CPU fallback (may be slow)
            model = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto")

    # Build generation pipeline. We pass the already loaded model + tokenizer.
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"  # keep this consistent with model device_map
    )

    return model, tokenizer, gen


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
                if top_nodes["issues"]:
                    print(f"\nTop issue numbers: {top_nodes['issues']}")
                if top_nodes["prs"]:
                    print(f"\nTop PR numbers: {top_nodes['prs']}")
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
    config = PipelineConfig(
        verbose=True,
        retriever="kg",
        top_k=10,
        llm_max_tokens=200,
        deduplicate=True,
        dedup_use_minhash=True,
        dedup_use_semantic=False,
        rerank=True,
        rerank_use_graph=True,
        rerank_use_popularity=True,
        over_retrieve_factor=10,
        over_retrieve_cap=200,
        rerank_candidate_cap=200,
    )
    
    # Instantiate backend components
    vectorstore = QdrantStore(
            model_name=qdrant_config["model_name"],
            qdrant_url=qdrant_config["url"],
            collection_name=qdrant_config["collection"],
            api_key=qdrant_config["api_key"],
            distance_type=qdrant_config["distance"]
        )
        
    # Build LLM
    llm_model = "mistralai/mistral-7b-instruct-v0.3"
    llm, tokenizer, gen = build_llm(llm_model,quantize=True,use_8bit=True)
    print(gen.device)
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
    )

    # Start interactive search
    search(repo_rag, config)

if __name__ == "__main__":
    main()