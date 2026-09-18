"""PDF ingestion: parse PDFs and split into chunks."""
from pathlib import Path
from typing import List, Dict
import hashlib

from pypdf import PdfReader
from .utils import chunk_text


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract all text from a PDF file."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    full_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        full_text.append(page_text)
    return "\n".join(full_text)


def extract_text_with_pages(pdf_path: str | Path) -> List[Dict]:
    """
    Extract text page by page (needed for citations).
    Returns list of {page_number, text, source}.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "page_number": i,
                "text": text,
                "source": pdf_path.name,
            })
    return pages


def make_chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Create a stable unique ID for a chunk."""
    raw = f"{source}::{page}::{index}::{text[:50]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def ingest_pdf(pdf_path: str | Path, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Full ingestion pipeline for one PDF.
    Returns list of chunks with metadata for citations.
    """
    pages = extract_text_with_pages(pdf_path)
    all_chunks = []

    for page in pages:
        chunks = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "id": make_chunk_id(page["source"], page["page_number"], idx, chunk),
                "text": chunk,
                "source": page["source"],
                "page": page["page_number"],
                "chunk_index": idx,
            })

    return all_chunks


def ingest_folder(folder: str | Path) -> List[Dict]:
    """Ingest all PDFs in a folder."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    all_chunks = []
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDFs found in {folder}")

    for pdf in pdf_files:
        print(f"Processing: {pdf.name}")
        chunks = ingest_pdf(pdf)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

    return all_chunks