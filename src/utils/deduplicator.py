from typing import List, Tuple
from langchain_core.documents import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH


class Deduplicator:
    def __init__(
        self,
        prefer_small_chunks: bool = True,
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
    def deduplicate(self, docs: List[Document],use_minhash: bool = False,
        jaccard_threshold: float = 0.9,
        use_semantic: bool = False,
        sim_threshold: float = 0.95,
        ) -> List[Document]:
        """
        Deduplicate documents with multiple strategies.

        Args:
            docs: list of LangChain Documents.
            use_minhash: apply MinHash Jaccard deduplication.
            jaccard_threshold: threshold for MinHash LSH.
            use_semantic: apply embedding-based semantic deduplication.
            sim_threshold: cosine similarity threshold for semantic dedup.
        """
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
        if use_minhash and len(unique_docs) > 1:
            lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=128)
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
                    kept.append(doc)
                lsh.insert(i, mh)

            unique_docs = kept

        # --- Step 3: semantic similarity dedup (optional) ---
        if use_semantic and self.embedder is not None and len(unique_docs) > 1:
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
                    if sim >= sim_threshold:
                        removed.add(j)

            unique_docs = keep

        return unique_docs
    
    def deduplicate_scored(
        self,
        scored_docs: List[Tuple[Document, float]],
        use_minhash: bool = False,
        jaccard_threshold: float = 0.9,
        use_semantic: bool = False,
        sim_threshold: float = 0.95,
    ) -> List[Tuple[Document, float]]:
        """
        Score-aware deduplication pipeline.
        Input:  [(Document, score), ...]
        Output: [(Document, score), ...] sorted by score desc
        """

        if not scored_docs:
            return []

        # -------------------------
        # 1) Exact dedup (by content)
        #    Keep highest-score document, tie → smaller chunk
        # -------------------------
        seen = {}
        for doc, score in scored_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen[key] = (doc, score)
            else:
                existing_doc, existing_score = seen[key]

                # Prefer higher score
                if score > existing_score:
                    seen[key] = (doc, score)
                # Tie → fallback to smaller chunk
                elif score == existing_score and self.prefer_small_chunks:
                    if doc.metadata.get("chunk_size", 9999) < existing_doc.metadata.get("chunk_size", 9999):
                        seen[key] = (doc, score)

        unique = list(seen.values())

        # -------------------------
        # 2) MinHash Jaccard dedup (optional)
        #    Among similar hashes, keep highest score
        # -------------------------
        if use_minhash and len(unique) > 1:
            from datasketch import MinHashLSH  # lazy import if not always used
            lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=128)
            mh_map = {}
            kept = []

            for i, (doc, score) in enumerate(unique):
                mh = self._compute_minhash(doc.page_content)
                mh_map[i] = (mh, doc, score)

                dup_indices = lsh.query(mh)

                if dup_indices:
                    # gather all conflict docs + the current one
                    candidates = [(doc, score)] + [
                        (unique[j][0], unique[j][1]) for j in dup_indices
                    ]
                    # choose best by score → tie by chunk size
                    best_doc, best_score = max(
                        candidates,
                        key=lambda x: (x[1], -x[0].metadata.get("chunk_size", 9999))
                    )
                    # only keep if the current doc is the best representative
                    if best_doc is doc:
                        kept.append((doc, score))
                else:
                    kept.append((doc, score))

                lsh.insert(i, mh)

            unique = kept

        # -------------------------
        # 3) Semantic similarity dedup (optional)
        #    Within similarity clusters, keep highest score
        # -------------------------
        if use_semantic and self.embedder is not None and len(unique) > 1:
            docs, scores = zip(*unique)
            embeddings = self.embedder.embed_documents([d.page_content for d in docs])

            keep_flags = [True] * len(docs)
            for i in range(len(docs)):
                if not keep_flags[i]:
                    continue
                for j in range(i + 1, len(docs)):
                    if not keep_flags[j]:
                        continue
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    if sim >= sim_threshold:
                        # keep the one with higher score
                        if scores[j] > scores[i]:
                            keep_flags[i] = False
                            break
                        else:
                            keep_flags[j] = False

            unique = [(doc, score) for (doc, score), keep in zip(unique, keep_flags) if keep]

        # -------------------------
        # 4) Final: sort by score desc
        # -------------------------
        unique.sort(key=lambda x: x[1], reverse=True)

        return unique
