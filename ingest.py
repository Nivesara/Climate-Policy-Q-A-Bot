"""Ingests PDF and text documents into a TF‑IDF vector index.

Usage:
    python ingest.py

All PDF and text files under the `docs/` directory are parsed.  The script extracts
plain text, splits it into overlapping chunks, computes a TF‑IDF matrix and
saves the index (vectorizer, matrix and metadata) to `index.pkl`.

Dependencies: scikit‑learn, pymupdf, pandas, python‑dotenv (optional).
"""

import os
import pickle
import pathlib
from typing import List, Tuple

import numpy as np
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer


DOCS_DIR = pathlib.Path(__file__).resolve().parent / "docs"
INDEX_FILE = pathlib.Path(__file__).resolve().parent / "index.pkl"


def extract_text_from_pdf(pdf_path: pathlib.Path) -> List[str]:
    """Extract text from each page of a PDF file."""
    texts = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text("text") or ""
                # normalize whitespace
                text = " ".join(text.split())
                texts.append(text)
    except Exception as e:
        print(f"Failed to read {pdf_path}: {e}")
    return texts


def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Split a long string into overlapping chunks.

    Parameters
    ----------
    text : str
        The document text.
    chunk_size : int, optional
        Maximum number of characters per chunk.
    overlap : int, optional
        Number of characters to overlap between chunks.
    """
    out = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        start = end - overlap  # slide window with overlap
    return out


def collect_documents() -> Tuple[List[str], List[dict]]:
    """Collect all document chunks and their metadata.

    Returns
    -------
    chunks : List[str]
        Text chunks extracted from the documents.
    metadata : List[dict]
        Metadata for each chunk, with keys `source` and `page`.
    """
    chunks = []
    metadata = []
    for path in DOCS_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}:
            if path.suffix.lower() == ".pdf":
                pages = extract_text_from_pdf(path)
            else:
                # plain text
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                pages = [content]
            for i, page_text in enumerate(pages, 1):
                for chunk in split_into_chunks(page_text):
                    chunks.append(chunk)
                    metadata.append({"source": path.name, "page": i})
    return chunks, metadata


def build_index() -> None:
    """Build the TF‑IDF vector index and persist it to disk."""
    chunks, metadata = collect_documents()
    if not chunks:
        print("No documents found in docs/. Place PDFs or text files there and retry.")
        return
    print(f"Indexing {len(chunks)} chunks from {len(set(m['source'] for m in metadata))} documents…")
    vectorizer = TfidfVectorizer(stop_words='english')
    # Fit the vectorizer and transform chunks into a sparse matrix
    X = vectorizer.fit_transform(chunks)
    # Persist index
    data = {
        "vectorizer": vectorizer,
        "matrix": X,
        "chunks": chunks,
        "metadata": metadata,
    }
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved index to {INDEX_FILE}.")


if __name__ == "__main__":
    build_index()