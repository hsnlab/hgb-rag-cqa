from src.rag.repo_rag import RepositoryRAG
from src.rag.config import PipelineConfig
from src.utils.qdrant_store import QdrantStore
from src.utils.retrieval import KnowledgeGraphRetriever
from src.utils.deduplicator import Deduplicator
from src.utils.reranker import Reranker
from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import login
from src.rag.agent_generate import run_agents


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
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
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


def search(context, config, huggingface_apikey: str):
        try:
            while True:
                question = input("\nPlease enter your question (Ctrl+C to exit): ").strip()
                if not question:
                    continue

                print("\nRunning agent-based pipeline...")
                result = run_agents(question, context=context, huggingface_apikey=huggingface_apikey)

                print("\n=== Final Answer ===")
                print(result)

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting search. Goodbye!")

def main():
    token_path ="./_/hf_token.txt"
    with open(token_path, "r") as f:
        huggingface_apikey = f.read().strip()

    login(huggingface_apikey)
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
        model_name="microsoft/codebert-base",
        qdrant_url="http://localhost:6333",
        collection_name="rag_collection_codebert-base_cosine",
        api_key="@lmafa12",
        distance_type="cosine"
    )
    kg_retriever = KnowledgeGraphRetriever(vector_store=vectorstore, neo4j_url ="bolt://localhost:7687", neo4j_username="neo4j", neo4j_password ="password", database= "neo4j")
    deduplicator = Deduplicator(embedder=vectorstore.embeddings)
    reranker = Reranker()

    # Build LLM
    llm_model = "Qwen/Qwen2.5-7B-Instruct"
    llm, tokenizer, gen = build_llm(llm_model,quantize=True,use_8bit=True)
    # Instantiate both RAG variants

    repo_rag = RepositoryRAG(
        vectorstore,
        retriever=kg_retriever,
        deduplicator=deduplicator,
        reranker=reranker,
        llm=gen,
        neo4j_auth=("neo4j", "password"),
        neo4j_uri="bolt://localhost:7687",
    )

    # Context to pass agents
    context = {
        "repo_rag": repo_rag,
        "vectorstore": vectorstore,
        "retriever": kg_retriever,
        "deduplicator": deduplicator,
        "reranker": reranker,
        "gen": gen,
    }

    # Start interactive search
    search(context, config, huggingface_apikey)

if __name__ == "__main__":
    main()
