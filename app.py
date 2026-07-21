"""
app.py
------
Streamlit web interface for the Healthcare Staff Knowledge Assistant.

Run with:
    streamlit run app.py

Pages (sidebar navigation):
    1. Chat Assistant        - ask staff questions, see sources + scores
    2. Knowledge Base Viewer - browse documents, rebuild the index
    3. Testing & Evaluation  - run the sample questions from tests/
    4. About Project         - plain-language explanation of the project
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import config
import ingest
import rag_pipeline
import safety

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Staff Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def index_exists() -> bool:
    """True if the FAISS index files are on disk."""
    return (
        Path(config.FAISS_INDEX_PATH).exists()
        and Path(config.METADATA_PATH).exists()
    )


@st.cache_resource(show_spinner="Setting up the assistant for first use (this can take a minute)...")
def ensure_index_ready() -> bool:
    """Build the vector index automatically if it does not exist yet.

    Locally you run `python ingest.py` yourself, but when the app is
    deployed to the cloud there is no terminal — so on the very first
    boot we build the index automatically. @st.cache_resource makes this
    run only ONCE per running app, not on every click.
    """
    if not index_exists():
        try:
            ingest.rebuild_index()
        except Exception as exc:
            # Don't crash the whole app; the setup warnings will explain.
            print(f"Automatic index build failed: {exc}")
            return False
    return True


def is_admin() -> bool:
    """True if the current user unlocked admin mode this session."""
    return st.session_state.get("is_admin", False)


def show_setup_warnings():
    """Show banners if the app is not fully set up yet.

    Technical details (API keys, index files) are internal information,
    so they are only shown to a logged-in admin. Regular staff just see
    a generic 'contact your administrator' message.
    """
    problems = []
    if not config.is_api_key_configured():
        problems.append("api_key")
    if not index_exists():
        problems.append("index")

    if not problems:
        return

    if is_admin():
        if "api_key" in problems:
            st.error(
                "**GEMINI_API_KEY is missing.** "
                "Copy `.env.example` to `.env`, paste your Gemini API key "
                "inside, then restart the app. The key must live in the "
                "`.env` file, never in the code."
            )
        if "index" in problems:
            st.warning(
                "**The vector index has not been built yet.** "
                "Run `python ingest.py` in a terminal, or use the "
                "*Rebuild Vector Index* button on the Knowledge Base "
                "Viewer page."
            )
    else:
        st.error(
            "The assistant is currently unavailable. "
            "Please contact your administrator or the IT helpdesk."
        )


def rebuild_index_with_spinner():
    """Rebuild the FAISS index and clear the pipeline cache."""
    with st.spinner("Rebuilding the vector index (this may take a minute)..."):
        try:
            num_chunks = ingest.rebuild_index()
            rag_pipeline.reset_index_cache()
            st.success(f"Index rebuilt successfully with {num_chunks} chunks.")
        except Exception as exc:
            st.error(f"Failed to rebuild the index: {exc}")


def render_result(result: dict):
    """Display one pipeline result: answer, sources, scores, warnings."""
    if result["error"]:
        st.error(result["error"])
        return

    if result["escalated"]:
        st.warning(result["answer"])
        st.caption(
            "This question was blocked by the safety layer before reaching "
            "the AI model."
        )
        return

    st.markdown(result["answer"])

    if result["low_confidence"]:
        st.warning(
            "⚠️ Low similarity between your question and the knowledge base. "
            "The answer above may be incomplete — double-check with your "
            "supervisor if unsure."
        )

    # Transparency: show which documents were used and how similar they were.
    if result["sources"]:
        with st.expander("📄 Sources and similarity scores"):
            for source, score in zip(result["sources"], result["scores"]):
                st.write(f"- **{source}** — similarity: `{score:.3f}`")
            st.caption(
                f"Similarity ranges from 0 (unrelated) to 1 (identical). "
                f"Threshold to answer: {config.SIMILARITY_THRESHOLD}."
            )


# ---------------------------------------------------------------------------
# Page 1: Chat Assistant
# ---------------------------------------------------------------------------
def page_chat():
    st.title("💬 Chat Assistant")
    st.caption(
        "Ask a staff question. Answers come ONLY from the approved "
        "knowledge base documents — never from general AI knowledge."
    )
    show_setup_warnings()

    # Keep chat history in the session so the conversation stays visible.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (question, result) pairs

    # Replay previous exchanges.
    for question, result in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            render_result(result)

    # Chat input box at the bottom of the page.
    question = st.chat_input("Type a staff question, e.g. 'How do I reschedule an appointment?'")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge base..."):
                result = rag_pipeline.answer_question(question)
            render_result(result)
        st.session_state.chat_history.append((question, result))


# ---------------------------------------------------------------------------
# Page 2: Knowledge Base Viewer
# ---------------------------------------------------------------------------
def page_knowledge_base():
    st.title("📚 Knowledge Base Viewer")
    st.caption("Browse the approved documents the assistant answers from.")

    kb_dir = Path(config.KNOWLEDGE_BASE_DIR)
    if not kb_dir.exists():
        st.error(f"Knowledge base folder not found: {kb_dir}")
        return

    files = sorted(
        p.name for p in kb_dir.iterdir() if p.suffix.lower() in (".md", ".txt")
    )
    if not files:
        st.warning("No .md or .txt documents found in the knowledge_base folder.")
        return

    st.write(f"**{len(files)} documents** in `{kb_dir.name}/`:")
    selected = st.selectbox("Select a document to view:", files)

    if selected:
        content = (kb_dir / selected).read_text(encoding="utf-8")
        st.markdown("---")
        st.subheader(selected)
        st.markdown(content)

    st.markdown("---")
    st.subheader("Rebuild the vector index")
    if is_admin():
        st.write(
            "If you edited, added, or removed documents, rebuild the index "
            "so the assistant sees the changes."
        )
        if st.button("🔄 Rebuild Vector Index", type="primary"):
            rebuild_index_with_spinner()
    else:
        st.caption(
            "🔒 Rebuilding the index is an admin action. Log in through "
            "the Admin section in the sidebar."
        )


# ---------------------------------------------------------------------------
# Page 3: Testing and Evaluation
# ---------------------------------------------------------------------------
def page_testing():
    st.title("🧪 Testing and Evaluation")
    st.caption(
        "Run the sample questions from `tests/sample_questions.csv` through "
        "the assistant. The expected behaviors are ONLY for checking results "
        "— they are never used as answers."
    )
    show_setup_warnings()

    csv_path = Path(config.SAMPLE_QUESTIONS_CSV)
    if not csv_path.exists():
        st.error(f"Test file not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    mode = st.radio(
        "Choose a test mode:",
        ["Run one question", "Run all questions"],
        horizontal=True,
    )

    if mode == "Run one question":
        options = [
            f"{i + 1}. {row['question']}" for i, row in df.iterrows()
        ]
        choice = st.selectbox("Pick a test question:", options)
        if st.button("▶️ Run this test", type="primary"):
            row = df.iloc[options.index(choice)]
            run_single_test(row)
    else:
        st.info(
            f"This will run all {len(df)} questions. Non-risky questions "
            "call the Gemini API, so this may take a couple of minutes."
        )
        if st.button("▶️ Run all tests", type="primary"):
            progress = st.progress(0.0)
            for i, (_, row) in enumerate(df.iterrows()):
                run_single_test(row)
                progress.progress((i + 1) / len(df))
            st.success("All tests finished.")


def run_single_test(row):
    """Run one CSV row through the real pipeline and show everything."""
    question = str(row["question"])
    expected = str(row.get("expected_behavior", ""))
    category = str(row.get("category", ""))

    with st.container(border=True):
        st.markdown(f"**Question:** {question}")
        st.markdown(f"**Category:** `{category}` &nbsp;|&nbsp; **Expected behavior:** {expected}")

        with st.spinner("Running through the pipeline..."):
            result = rag_pipeline.answer_question(question)

        if result["escalated"]:
            st.markdown("**Outcome:** 🚫 Escalated (blocked by safety layer)")
        elif result["error"]:
            st.markdown("**Outcome:** ❌ Error")
        elif result["low_confidence"] and not result["sources"]:
            st.markdown("**Outcome:** 🔍 Refused (no relevant context)")
        else:
            st.markdown("**Outcome:** ✅ Answered from knowledge base")

        st.markdown("**Actual answer:**")
        render_result(result)


# ---------------------------------------------------------------------------
# Page 4: About Project
# ---------------------------------------------------------------------------
def page_about():
    st.title("ℹ️ About This Project")
    st.markdown(
        f"""
### What problem does it solve?
Healthcare staff constantly need answers to operational questions —
*How do I reschedule an appointment? What do I do on my first day?
How do I handle a billing question?* — but the answers are buried across
policy documents. This assistant lets staff ask in plain language and
get an answer sourced **only** from approved internal documents.

### Why do healthcare staff need it?
- New staff get instant, consistent onboarding answers.
- Experienced staff save time hunting through policies.
- Every answer cites its source document, so it can be verified.
- Risky questions are automatically escalated to a human, never guessed.

### How does RAG work here?
**RAG = Retrieval-Augmented Generation.** Instead of letting the AI
answer from its general training knowledge, we:
1. Split the approved documents into small chunks.
2. Convert each chunk into an **embedding** (a vector of numbers that
   captures its meaning) using a local machine learning model.
3. Store all vectors in a **FAISS** similarity-search index.
4. When a question arrives, embed it the same way and find the most
   similar chunks.
5. Send **only those chunks** to Gemini with strict instructions to
   answer from them alone.

### What machine learning is used?
- **`{config.EMBEDDING_MODEL_NAME}`** — a Sentence Transformers model
  that runs locally and produces 384-dimensional embeddings. Texts with
  similar meaning get vectors that point in similar directions, which we
  measure with **cosine similarity**.
- **FAISS** — Facebook AI Similarity Search, a library for finding the
  nearest vectors extremely fast.

### What does Gemini do?
Gemini (**`{config.GEMINI_MODEL}`**) only performs the final step:
turning the retrieved chunks into a natural, readable answer. It is
explicitly instructed not to use outside knowledge and to say when the
answer is not in the provided context.

### What safety rules are included?
- A keyword-based safety filter (`safety.py`) blocks questions about
  diagnosis, treatment, medication, dosage, emergencies, patient-specific
  decisions, insurance approvals, and legal decisions **before** any AI
  call happens.
- A **similarity threshold** ({config.SIMILARITY_THRESHOLD}): if no
  knowledge base chunk is similar enough, the assistant refuses instead
  of guessing.
- The Gemini prompt forbids outside knowledge and invented policies.
- Every answer shows its source documents and similarity scores.

### Limitations
- Keyword safety matching can miss creative phrasings (a production
  system would add an ML-based classifier).
- Answer quality depends entirely on the quality of the documents.
- The index must be rebuilt manually after documents change.
- Gemini is a cloud API, so answering requires internet access.

### Future improvements
- Automatic index rebuilding when documents change.
- Multi-language support for diverse staff.
- Conversation memory for follow-up questions.
- An admin dashboard with usage and escalation analytics.
- A trained ML safety classifier instead of keyword matching.
- Role-based access (front desk vs. billing vs. nursing content).
"""
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
# Build the search index automatically on first boot (needed for the cloud,
# where there is no terminal to run `python ingest.py`). Runs only once.
ensure_index_ready()

st.sidebar.title("🏥 Staff Knowledge Assistant")
page = st.sidebar.radio(
    "Navigate:",
    ["Chat Assistant", "Knowledge Base Viewer", "Testing and Evaluation", "About Project"],
)

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Admin section: system status is internal information, so it is only
# visible after entering the admin password (set via ADMIN_PASSWORD in .env).
# ---------------------------------------------------------------------------
with st.sidebar.expander("🔐 Admin"):
    if is_admin():
        st.markdown("**System status**")
        st.write("🔑 API key:", "✅ found" if config.is_api_key_configured() else "❌ missing")
        st.write("🗂️ Vector index:", "✅ built" if index_exists() else "❌ not built")
        st.write("🤖 Model:", f"`{config.GEMINI_MODEL}`")
        if st.button("Log out"):
            st.session_state["is_admin"] = False
            st.rerun()
    else:
        # A form submits the password and button click together, so
        # pressing Enter in the field also logs in.
        with st.form("admin_login_form", clear_on_submit=True):
            admin_password = st.text_input("Admin password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            if config.ADMIN_PASSWORD and admin_password == config.ADMIN_PASSWORD:
                st.session_state["is_admin"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Answers come only from approved internal documents. "
    "Clinical and sensitive questions are escalated to a human supervisor."
)

if page == "Chat Assistant":
    page_chat()
elif page == "Knowledge Base Viewer":
    page_knowledge_base()
elif page == "Testing and Evaluation":
    page_testing()
else:
    page_about()
