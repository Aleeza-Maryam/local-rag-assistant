"""Embeddings + ChromaDB vector store for retrieval."""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .utils import get_vector_db_path, get_embedding_model_name


class Retriever:
    """Handles embedding generation and vector search."""

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

    def add_chunks(self, chunks: List[Dict]) -> int:
        """Add chunks to vector DB. Returns number added."""
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

        # Upsert so re-ingesting same file doesn't duplicate
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for the most relevant chunks."""
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
                "distance": results["distances"][0][i] if "distances" in results else None,
            })
        return hits

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