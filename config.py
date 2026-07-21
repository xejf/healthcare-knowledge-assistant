"""
config.py
---------
Central configuration for the Healthcare Staff Knowledge Assistant.

Everything the other files need to know (folder paths, model names,
tuning values, API keys) lives here in ONE place, so a beginner only
has to look in one file to change settings.

The Gemini API key is NEVER written in code. It is loaded from a .env
file using python-dotenv. See .env.example for the format.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from the .env file (if it exists).
# load_dotenv() looks for a file named ".env" in the project folder and
# makes its values available through os.getenv().
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Folder paths
# ---------------------------------------------------------------------------
# BASE_DIR = the folder that contains this config.py file.
# Using Path(__file__) means the app works no matter where you run it from.
BASE_DIR = Path(__file__).resolve().parent

# Folder that holds the approved knowledge base documents (.md / .txt files).
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

# Folder where the FAISS vector index and chunk metadata are stored.
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.pkl"

# Folder with the evaluation questions used on the Testing page.
TESTS_DIR = BASE_DIR / "tests"
SAMPLE_QUESTIONS_CSV = TESTS_DIR / "sample_questions.csv"

# ---------------------------------------------------------------------------
# Embedding (machine learning) settings
# ---------------------------------------------------------------------------
# This Sentence Transformers model runs locally on your machine.
# It converts text into 384-dimensional numeric vectors ("embeddings").
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# How the documents are split into chunks before embedding.
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
# How many knowledge base chunks to retrieve for each question.
TOP_K = 3

# Minimum cosine similarity (0.0 - 1.0) for a chunk to be considered
# relevant. If the BEST retrieved chunk scores below this, the assistant
# refuses to answer instead of guessing.
SIMILARITY_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Settings that come from secrets (API key, model, admin password)
# ---------------------------------------------------------------------------
# These are read from two possible places, in order:
#   1. Environment variables — filled from the local .env file (running on
#      your own computer).
#   2. Streamlit secrets — the secure box you fill in when the app is
#      deployed to Streamlit Community Cloud (there is no .env file there).
# This lets the SAME code run both locally and online without changes.


def _get_secret(name: str, default: str = "") -> str:
    """Return a secret from the environment, or Streamlit secrets, or default."""
    # 1. Local .env / environment variable.
    value = os.getenv(name)
    if value:
        return value.strip()

    # 2. Streamlit Cloud secrets (only imported if step 1 found nothing,
    #    so the command-line tools stay fast when a .env file is present).
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass  # not running inside Streamlit, or no secrets configured

    return default


GEMINI_API_KEY = _get_secret("GEMINI_API_KEY", "")
# "gemini-flash-latest" is an alias that always points to Google's current
# fast Gemini model, so the app keeps working when old versions are retired.
GEMINI_MODEL = _get_secret("GEMINI_MODEL", "gemini-flash-latest")

# ---------------------------------------------------------------------------
# Admin settings
# ---------------------------------------------------------------------------
# Password that unlocks admin-only parts of the app (system status panel,
# rebuild-index button). Set it in .env locally or in Streamlit secrets
# online. If it is empty, admin features stay locked for everyone.
ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD", "")


def is_api_key_configured() -> bool:
    """Return True if a Gemini API key was found in the environment."""
    return bool(GEMINI_API_KEY)
