from typing import List
from langchain_core.documents import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH


class Deduplicator:
    def __init__(
        self,
        prefer_small_chunks: bool = True,
        use_minhash: bool = False,
        jaccard_threshold: float = 0.9,
        use_semantic: bool = False,
        sim_threshold: float = 0.95,
        embedder=None
    ):
        """
        Deduplicates retrieved documents with multiple strategies.
        
        Args:
            prefer_small_chunks: keep smaller chunks when duplicates exist.
            use_minhash: enable MinHash Jaccard deduplication.
            jaccard_threshold: Jaccard similarity threshold for MinHash dedup.
            use_semantic: enable semantic deduplication (embedding-based).
            sim_threshold: cosine similarity threshold for semantic dedup.
            embedder: embedding model (must implement .embed_documents).
        """
        self.prefer_small_chunks = prefer_small_chunks
        self.use_minhash = use_minhash
        self.jaccard_threshold = jaccard_threshold
        self.use_semantic = use_semantic
        self.sim_threshold = sim_threshold
        self.embedder = embedder

    # ------------------------
    # Helpers
    # ------------------------
    def _compute_minhash(self, text: str, num_perm: int = 128) -> MinHash:
        mh = MinHash(num_perm=num_perm)
        tokens = text.lower().split()
        for t in tokens:
            mh.update(t.encode("utf8"))
        return mh

    # ------------------------
    # Deduplication pipeline
    # ------------------------
    def deduplicate(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        # --- Step 1: exact deduplication ---
        seen = {}
        for doc in docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen[key] = doc
            else:
                existing = seen[key]
                if self.prefer_small_chunks:
                    if doc.metadata.get("chunk_size", 9999) < existing.metadata.get("chunk_size", 9999):
                        seen[key] = doc
                else:
                    if doc.metadata.get("chunk_size", 0) > existing.metadata.get("chunk_size", 0):
                        seen[key] = doc

        unique_docs = list(seen.values())

        # --- Step 2: MinHash Jaccard dedup (optional) ---
        if self.use_minhash and len(unique_docs) > 1:
            lsh = MinHashLSH(threshold=self.jaccard_threshold, num_perm=128)
            mh_map = {}
            kept = []

            for i, doc in enumerate(unique_docs):
                mh = self._compute_minhash(doc.page_content)
                mh_map[i] = mh

                # Query LSH for near-duplicates
                dup_indices = lsh.query(mh)
                if dup_indices:
                    # conflict resolution: keep smaller chunk
                    to_keep = min(
                        [doc] + [unique_docs[j] for j in dup_indices],
                        key=lambda d: d.metadata.get("chunk_size", 9999)
                    )
                    if to_keep == doc:
                        kept.append(doc)

                else:
                    lsh.insert(i, mh)
                    kept.append(doc)

            unique_docs = kept

        # --- Step 3: semantic similarity dedup (optional) ---
        if self.use_semantic and self.embedder is not None and len(unique_docs) > 1:
            embeddings = self.embedder.embed_documents([d.page_content for d in unique_docs])
            keep = []
            removed = set()

            for i, emb_i in enumerate(embeddings):
                if i in removed:
                    continue
                keep.append(unique_docs[i])
                for j in range(i + 1, len(embeddings)):
                    if j in removed:
                        continue
                    sim = cosine_similarity([emb_i], [embeddings[j]])[0][0]
                    if sim >= self.sim_threshold:
                        removed.add(j)

            unique_docs = keep

        return unique_docs
