"""
rag_pipeline.py
---------------
The heart of the project: Retrieval-Augmented Generation (RAG).

Flow for every question:

    question
       |
       v
    safety check (safety.py)  -- risky? --> escalation message (Gemini NOT called)
       |
       v
    embed the question (local Sentence Transformers model)
       |
       v
    FAISS search  -->  top-k most similar knowledge base chunks
       |
       v
    similarity too low / nothing found? --> "not in approved documents" message
       |
       v
    build a strict prompt containing ONLY the retrieved chunks
       |
       v
    Gemini generates the answer from that context alone
"""

import re

import faiss
import numpy as np

import config
import ingest
import safety

# ---------------------------------------------------------------------------
# The strict system instruction sent to Gemini with EVERY request.
# It forbids outside knowledge and forces escalation for unsafe topics.
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are a healthcare staff knowledge assistant. Answer the user's "
    "question using ONLY the provided context from the approved knowledge "
    "base. Do not use outside knowledge. Do not invent procedures, "
    "policies, or medical advice. If the answer is not clearly found in "
    "the provided context, say: 'I could not find this information in the "
    "approved documents. Please contact your supervisor or the relevant "
    "department.' If the question asks for diagnosis, treatment, "
    "medication, dosage, emergency advice, or patient-specific clinical "
    "guidance, say: 'I’m not authorized to answer this. Please contact "
    "your supervisor or the relevant department.'"
)

# Standard message when the knowledge base has no good match.
NOT_FOUND_MESSAGE = (
    "I could not find this information in the approved documents. "
    "Please contact your supervisor or the relevant department."
)

# ---------------------------------------------------------------------------
# Friendly handling of greetings and small talk.
# A plain "hi" has no meaning to search for in the knowledge base, so
# instead of a confusing refusal we show a short welcome. This is
# interface behaviour (like the escalation message), not a knowledge
# base answer.
# ---------------------------------------------------------------------------
GREETING_PHRASES = {
    "hi", "hello", "hey", "hi there", "hello there", "yo",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye",
}

WELCOME_MESSAGE = (
    "Hello! 👋 I'm the staff knowledge assistant. Ask me anything about "
    "our internal procedures in your own words — no special format needed.\n\n"
    "For example:\n"
    "- *How do I reschedule an appointment?*\n"
    "- *What should I do on my first day?*\n"
    "- *A patient is upset about a bill — what do I do?*\n"
    "- *How do I report a safety issue?*\n\n"
    "You can also ask *\"what can you help me with?\"* to learn more."
)


def _is_greeting(question: str) -> bool:
    """True for short greetings/small talk like 'hi' or 'thanks'."""
    cleaned = question.lower().strip(" \t!?.,:;")
    return cleaned in GREETING_PHRASES


# ---------------------------------------------------------------------------
# Questions ABOUT the assistant itself ("who are you?", "how do I use
# you?"). These are too short and too different from the policy documents
# to retrieve well, so — like greetings — they get a built-in help answer
# describing how the tool works. This is interface behaviour, not a
# knowledge base answer.
# ---------------------------------------------------------------------------
META_QUESTION_PATTERNS = [
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\bwhat is this (assistant|app|tool|chatbot|bot)\b",
    r"\bhow (do|can|should) i use (you|this|it|the assistant)\b",
    r"\bhow (do|does) (you|this|it) work\b",
    r"\bwhat can you (do|answer|help)\b",
    r"\bwhat (can|do) i ask( you)?\b",
    r"\bwhat can you help\b",
    r"\bcan you help me\b",
    r"^help$",
    r"^what do you do$",
]

HELP_MESSAGE = (
    "I'm the **staff knowledge assistant** — an internal helper that answers "
    "questions using only our approved documents (onboarding, appointments, "
    "billing direction, patient service, safety, and escalation rules).\n\n"
    "**How to use me:** just type your question in your own words, like you "
    "would ask a colleague. No special format is needed. For example:\n"
    "- *How do I reschedule an appointment?*\n"
    "- *A patient is upset about a bill — what do I do?*\n"
    "- *I'm new here, what happens on my first day?*\n"
    "- *How do I report a safety issue?*\n\n"
    "**What I won't answer:** medical, medication, dosage, emergency, "
    "insurance-approval, or legal questions — for those I'll always point "
    "you to your supervisor or the relevant department.\n\n"
    "Every answer shows which approved document it came from, and you can "
    "read the full documents on the *Knowledge Base Viewer* page."
)


def _is_meta_question(question: str) -> bool:
    """True if the question is about the assistant itself."""
    text = question.lower().strip(" \t!?.,:;")
    return any(re.search(pattern, text) for pattern in META_QUESTION_PATTERNS)

# Cached index + metadata so we don't reload from disk on every question.
_index = None
_metadata = None


def _get_index():
    """Load the FAISS index and metadata once, then reuse them."""
    global _index, _metadata
    if _index is None or _metadata is None:
        _index, _metadata = ingest.load_index()  # may raise FileNotFoundError
    return _index, _metadata


def reset_index_cache():
    """Forget the cached index (called after the index is rebuilt)."""
    global _index, _metadata
    _index = None
    _metadata = None


def retrieve_relevant_chunks(question: str, top_k: int = config.TOP_K) -> list[dict]:
    """Find the knowledge base chunks most similar to the question.

    Returns a list of dicts sorted by similarity (best first):
        [{"source": ..., "chunk_id": ..., "text": ..., "score": 0.72}, ...]
    """
    index, metadata = _get_index()

    # 1. Turn the question into an embedding with the SAME model used
    #    during ingestion (this is essential — vectors must be comparable).
    model = ingest.get_embedding_model()
    question_embedding = model.encode([question])
    question_embedding = np.asarray(question_embedding, dtype="float32")

    # 2. Normalize so the inner-product score equals cosine similarity.
    faiss.normalize_L2(question_embedding)

    # 3. Search the index for the top_k closest chunks.
    scores, indices = index.search(question_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 when there are fewer chunks than top_k
            continue
        chunk = metadata[idx]
        results.append(
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(score),
            }
        )
    return results


def _build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Assemble the full prompt sent to Gemini.

    It contains the strict instructions, the retrieved context (each
    chunk labelled with its source filename), and the user question.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Context {i} | Source: {chunk['source']}]\n{chunk['text']}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""{SYSTEM_INSTRUCTION}

Additional formatting rules:
- Be concise and clear.
- Use short bullet points or numbered steps when they make the answer easier to follow.
- At the end of your answer, cite the source document name(s) you used, like: Sources: appointments.md
- Never invent information that is not in the context below.

=== APPROVED KNOWLEDGE BASE CONTEXT ===
{context_block}
=== END OF CONTEXT ===

Staff question: {question}

Answer:"""
    return prompt


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """Send the question + retrieved context to Gemini and return the text.

    Raises RuntimeError with a friendly message if the API key is missing
    or the API call fails, so the Streamlit app can display it nicely.
    """
    if not config.is_api_key_configured():
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add "
            "your Gemini API key."
        )

    prompt = _build_prompt(question, retrieved_chunks)

    try:
        # Imported here so the rest of the app (retrieval, safety, viewer)
        # still works even if the google-genai package has an issue.
        from google import genai

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty response.")
        return answer
    except RuntimeError:
        raise
    except Exception as exc:  # network errors, invalid key, quota, etc.
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc


def answer_question(question: str) -> dict:
    """Full end-to-end pipeline for one question.

    Returns a dict the UI can display:
        {
            "answer":     str,          # what to show the user
            "sources":    [str],        # document names used
            "scores":     [float],      # similarity of each retrieved chunk
            "escalated":  bool,         # True if blocked by safety rules
            "low_confidence": bool,     # True if best similarity is weak
            "error":      str or None,  # set when something went wrong
        }
    """
    result = {
        "answer": "",
        "sources": [],
        "scores": [],
        "escalated": False,
        "low_confidence": False,
        "error": None,
    }

    question = (question or "").strip()
    if not question:
        result["error"] = "Please type a question first."
        return result

    # --- Step 0: greetings get a friendly welcome, not a refusal. --------
    if _is_greeting(question):
        result["answer"] = WELCOME_MESSAGE
        return result

    # --- Step 0b: questions about the assistant itself get a help answer.
    if _is_meta_question(question):
        result["answer"] = HELP_MESSAGE
        return result

    # --- Step 1: safety check. Risky questions never reach Gemini. -------
    if safety.is_risky_question(question):
        reason = safety.get_risk_reason(question)
        result["answer"] = safety.escalation_response(reason)
        result["escalated"] = True
        return result

    # --- Step 2: retrieve the most relevant knowledge base chunks. -------
    try:
        retrieved = retrieve_relevant_chunks(question, config.TOP_K)
    except FileNotFoundError as exc:
        result["error"] = str(exc)
        return result
    except Exception as exc:
        result["error"] = f"Retrieval failed: {exc}"
        return result

    result["sources"] = [c["source"] for c in retrieved]
    result["scores"] = [c["score"] for c in retrieved]

    # --- Step 3: refuse if nothing relevant was found. --------------------
    best_score = max(result["scores"], default=0.0)
    if not retrieved or best_score < config.SIMILARITY_THRESHOLD:
        result["answer"] = NOT_FOUND_MESSAGE
        result["low_confidence"] = True
        return result

    # Flag weak (but still usable) matches so the UI can show a warning.
    if best_score < config.SIMILARITY_THRESHOLD + 0.15:
        result["low_confidence"] = True

    # --- Step 4: generate the answer with Gemini. -------------------------
    try:
        result["answer"] = generate_answer(question, retrieved)
    except RuntimeError as exc:
        result["error"] = str(exc)
    return result
