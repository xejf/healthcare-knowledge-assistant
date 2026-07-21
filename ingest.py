"""
ingest.py
---------
Builds the vector index for the Healthcare Staff Knowledge Assistant.

This is step 1 of the RAG pipeline:

    documents  ->  chunks  ->  embeddings  ->  FAISS index (saved to disk)

Run it directly whenever the knowledge base changes:

    python ingest.py

What happens:
1. Every .md / .txt file inside knowledge_base/ is read.
2. Each document is split into overlapping chunks of text.
3. A local Sentence Transformers model turns each chunk into an
   embedding (a list of numbers that captures its meaning).
4. Embeddings are normalized so FAISS inner-product search behaves
   like cosine similarity (scores from roughly 0 to 1).
5. The FAISS index is saved to vector_store/faiss_index.bin and the
   chunk metadata (source file, chunk id, original text) is saved to
   vector_store/metadata.pkl.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config

# The embedding model is loaded lazily (only when first needed) because
# it takes a few seconds and downloads ~90 MB the first time.
_embedding_model = None


def _is_model_cached() -> bool:
    """True if the embedding model is already in the local HF cache.

    Why we care: on some Windows machines, Hugging Face's "check for
    updates" network call can crash after PyTorch is loaded. Once the
    model is cached there is no reason to touch the network at all, so
    we load it fully offline. On the very first run (nothing cached yet)
    we allow the download.
    """
    try:
        from huggingface_hub import snapshot_download

        # local_files_only=True raises an error if the model is NOT cached.
        snapshot_download(config.EMBEDDING_MODEL_NAME, local_files_only=True)
        return True
    except Exception:
        return False


def _force_offline_mode():
    """Switch huggingface_hub to offline mode for this process.

    Setting the environment variable alone is not enough, because
    huggingface_hub reads HF_HUB_OFFLINE only once, at import time —
    so we also patch the already-imported constant.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = True
    except Exception:
        pass


def get_embedding_model() -> SentenceTransformer:
    """Load the Sentence Transformers model once and reuse it."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
        cached = _is_model_cached()
        if cached:
            _force_offline_mode()
        _embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME, local_files_only=cached
        )
    return _embedding_model


def load_documents() -> list[dict]:
    """Read all .md and .txt files from the knowledge_base folder.

    Returns a list of dicts: [{"filename": "onboarding.md", "text": "..."}]
    """
    documents = []
    kb_dir = Path(config.KNOWLEDGE_BASE_DIR)

    if not kb_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base folder not found: {kb_dir}. "
            "Create it and add .md or .txt documents."
        )

    for file_path in sorted(kb_dir.iterdir()):
        if file_path.suffix.lower() in (".md", ".txt"):
            text = file_path.read_text(encoding="utf-8")
            if text.strip():  # skip empty files
                documents.append({"filename": file_path.name, "text": text})

    if not documents:
        raise ValueError(
            f"No .md or .txt documents found in {kb_dir}. "
            "Add at least one document before building the index."
        )

    return documents


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split one document into overlapping chunks of characters.

    Why overlap? If a sentence sits right on a chunk boundary, the
    overlap makes sure it still appears complete in at least one chunk.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move forward, but keep `overlap` characters from the last chunk.
        start += chunk_size - overlap
    return chunks


def build_faiss_index():
    """Full pipeline: load docs -> chunk -> embed -> build FAISS index.

    Returns (index, metadata) where metadata is a list of dicts, one per
    chunk, in the same order as the vectors inside the index.
    """
    documents = load_documents()
    print(f"Loaded {len(documents)} documents from {config.KNOWLEDGE_BASE_DIR}")

    # ---- 1. Chunk every document, remembering where each chunk came from.
    metadata = []   # one entry per chunk
    all_chunks = []  # the raw text of every chunk (for embedding)
    for doc in documents:
        chunks = chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            metadata.append(
                {
                    "source": doc["filename"],  # which file the chunk is from
                    "chunk_id": i,              # position within that file
                    "text": chunk,              # the original chunk text
                }
            )
            all_chunks.append(chunk)

    print(f"Created {len(all_chunks)} chunks")

    # ---- 2. Convert chunks into embeddings with the local ML model.
    model = get_embedding_model()
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # ---- 3. Normalize each vector to length 1. With normalized vectors,
    # inner product == cosine similarity, so scores are easy to interpret.
    faiss.normalize_L2(embeddings)

    # ---- 4. Build the FAISS index (IndexFlatIP = exact inner-product search).
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors of dimension {dimension}")

    return index, metadata


def save_index(index, metadata):
    """Save the FAISS index and chunk metadata to the vector_store folder."""
    Path(config.VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved index to    {config.FAISS_INDEX_PATH}")
    print(f"Saved metadata to {config.METADATA_PATH}")


def load_index():
    """Load a previously saved FAISS index and metadata from disk.

    Returns (index, metadata). Raises FileNotFoundError if the index has
    not been built yet — the app catches this and tells the user to run
    `python ingest.py` or use the rebuild button.
    """
    if not Path(config.FAISS_INDEX_PATH).exists() or not Path(config.METADATA_PATH).exists():
        raise FileNotFoundError(
            "Vector index not found. Build it first by running: python ingest.py"
        )

    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def rebuild_index():
    """Convenience helper used by the Streamlit rebuild button."""
    index, metadata = build_faiss_index()
    save_index(index, metadata)
    return len(metadata)


if __name__ == "__main__":
    # Running `python ingest.py` rebuilds the whole index from scratch.
    print("Building the vector index from the knowledge base ...")
    rebuild_index()
    print("Done! You can now run: streamlit run app.py")
