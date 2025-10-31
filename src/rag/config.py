from dataclasses import dataclass
from typing import Optional

@dataclass
class PipelineConfig:
    verbose: bool = False
    retriever: str = "simple"
    use_query_classification: bool = True
    use_query_expansion: bool = True
    query_expansion_variants: int = 3
    deduplicate: bool = False
    dedup_use_minhash: bool = False
    dedup_jaccard_threshold: float = 0.9
    dedup_use_semantic: bool = False
    dedup_sim_threshold: float = 0.95
    rerank: bool = False
    rerank_use_graph: bool = False
    rerank_use_popularity: bool = False
    top_k: int = 10
    llm_max_tokens: int = 200
    over_retrieve: bool = True               
    over_retrieve_factor: int = 10            
    over_retrieve_cap: int = 200              
    rerank_candidate_cap: int = 50
    use_shortest_path_context: bool = True,
    top_k_paths: int = 10,
    top_k_docs_per_path: int = 10          
