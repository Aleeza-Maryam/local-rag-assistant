"""End-to-end test of the RAG pipeline."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import ingest_pdf
from src.retriever import Retriever
from src.generator import generate_answer


def main():
    pdf_path = "data/sample.pdf"

    print("=" * 60)
    print("STEP 1: Ingesting PDF...")
    print("=" * 60)
    chunks = ingest_pdf(pdf_path)
    print(f"Total chunks: {len(chunks)}")
    for c in chunks[:3]:
        print(f"  - Page {c['page']}, chunk {c['chunk_index']}: {c['text'][:60]}...")

    print("\n" + "=" * 60)
    print("STEP 2: Storing in vector DB...")
    print("=" * 60)
    retriever = Retriever()
    retriever.reset()  # fresh start
    count = retriever.add_chunks(chunks)
    print(f"Added {count} chunks. DB now has {retriever.count()} chunks.")

    print("\n" + "=" * 60)
    print("STEP 3: Searching...")
    print("=" * 60)
    query = "What is machine learning?"
    print(f"Query: {query}")
    hits = retriever.search(query, top_k=3)
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h['source']} p.{h['page']}: {h['text'][:80]}...")

    print("\n" + "=" * 60)
    print("STEP 4: Generating answer with Gemini...")
    print("=" * 60)
    answer = generate_answer(query, hits, provider="gemini")
    print(f"\nAnswer:\n{answer}")

    print("\n" + "=" * 60)
    print("STEP 5: Trying Groq...")
    print("=" * 60)
    answer = generate_answer(query, hits, provider="groq")
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()