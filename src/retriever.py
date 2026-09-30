"""Embeddings + ChromaDB vector store + BM25 hybrid retrieval."""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from .utils import get_vector_db_path, get_embedding_model_name


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class Retriever:
    """Handles embedding generation, vector search, and BM25 hybrid retrieval."""

    def __init__(self, collection_name: str = "documents"):
        self.model = SentenceTransformer(get_embedding_model_name())
        self.db_path = get_vector_db_path()
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict] = []
        self._bm25_corpus_tokens: List[List[str]] = []
        self._rebuild_bm25()

    # ---------- Indexing ----------
    def add_chunks(self, chunks: List[Dict]) -> int:
        """Add chunks to vector DB and refresh BM25 index."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ]

        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self._rebuild_bm25()
        return len(chunks)

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from current ChromaDB contents."""
        data = self.collection.get(include=["documents", "metadatas"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        self._bm25_docs = []
        self._bm25_corpus_tokens = []

        for text, meta in zip(docs, metas):
            self._bm25_docs.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", 0),
            })
            self._bm25_corpus_tokens.append(_tokenize(text))

        if self._bm25_corpus_tokens:
            self._bm25 = BM25Okapi(self._bm25_corpus_tokens)
        else:
            self._bm25 = None

    # ---------- Search ----------
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Pure semantic search via ChromaDB."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i]["page"],
                "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                "distance": results["distances"][0][i] if "distances" in results else None,
            })
        return hits

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """Keyword search via BM25."""
        if not self._bm25 or not self._bm25_docs:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top-K indices
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        hits = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            doc = self._bm25_docs[i]
            hits.append({
                "id": f"bm25::{i}",
                "text": doc["text"],
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": doc["chunk_index"],
                "bm25_score": float(scores[i]),
            })
        return hits

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Dict]:
        """
        Hybrid search combining vector and BM25 results.

        Args:
            query: search query
            top_k: number of results to return
            alpha: weight for vector search (0 = BM25 only, 1 = vector only)
        """
        if self.collection.count() == 0:
            return []

        # Fetch extra candidates from each retriever
        candidates = max(top_k * 3, 10)
        vector_hits = self._vector_search(query, candidates)
        bm25_hits = self._bm25_search(query, candidates)

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        lookup: Dict[str, Dict] = {}

        for rank, hit in enumerate(vector_hits):
            key = f"{hit['source']}::{hit['page']}::{hit['chunk_index']}"
            rrf_scores[key] = rrf_scores.get(key, 0) + alpha * (1.0 / (rank + 60))
            lookup[key] = hit

        for rank, hit in enumerate(bm25_hits):
            key = f"{hit['source']}::{hit['page']}::{hit['chunk_index']}"
            rrf_scores[key] = rrf_scores.get(key, 0) + (1 - alpha) * (1.0 / (rank + 60))
            if key not in lookup:
                lookup[key] = hit

        # Sort by fused score
        ranked_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for key in ranked_keys[:top_k]:
            hit = lookup[key].copy()
            hit["fused_score"] = rrf_scores[key]
            results.append(hit)

        return results

    def count(self) -> int:
        """Return number of chunks in DB."""
        return self.collection.count()

    def reset(self):
        """Delete all data (careful!)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_docs = []
        self._bm25_corpus_tokens = []