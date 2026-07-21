# AI-Powered Healthcare Staff Training & Knowledge Assistant (Gemini RAG)

A beginner-friendly capstone project: an internal assistant that answers healthcare **staff** questions (onboarding, appointments, billing direction, patient service, safety, escalation) using **Retrieval-Augmented Generation (RAG)**.

The assistant does **not** use hardcoded answers and does **not** answer from general AI knowledge. Every answer is generated dynamically from approved documents in the `knowledge_base/` folder — and if the information is missing, unclear, clinical, or sensitive, it escalates to a human supervisor instead.

> **Note:** All knowledge base content is **fictional sample data** for a training project. No real patient data is used or stored.

---

## Features

- **Chat Assistant** — ask staff questions in plain language; answers cite their source documents and show similarity scores.
- **Safety layer** — clinical, medication, dosage, emergency, insurance-approval, and legal questions are blocked *before* any AI call and escalated to a human.
- **Grounded answers only** — if no knowledge base chunk is similar enough to the question, the assistant refuses instead of guessing.
- **Knowledge Base Viewer** — browse the approved documents and rebuild the index after edits.
- **Testing & Evaluation page** — run ~20 sample questions through the real pipeline and compare outcomes with expected behavior.
- **No secrets in code** — the Gemini API key lives in a `.env` file only.
- **Admin mode** — system status and index rebuilding are hidden behind an admin password; staff see only the chat and documents.
- **Natural phrasing** — staff can type questions in their own words; greetings and "who are you / how do I use you" questions get a friendly built-in help answer, and an `assistant_guide.md` document teaches usage tips.

## How RAG Works Here

```
knowledge_base/*.md ──► split into chunks ──► embeddings (local ML model) ──► FAISS index
                                                                                  │
user question ──► safety check ──► question embedding ──► similarity search ◄────┘
                       │                                        │
                  risky? escalate                     top-3 relevant chunks
                                                                │
                                              Gemini (strict prompt: context only)
                                                                │
                                                    answer + sources + scores
```

1. **Ingestion** (`ingest.py`): documents are split into ~800-character chunks with 150-character overlap. Each chunk is converted into an embedding and stored in a FAISS index.
2. **Retrieval** (`rag_pipeline.py`): the user's question is embedded with the *same* model and FAISS finds the most similar chunks (cosine similarity).
3. **Generation**: only those retrieved chunks are sent to Gemini with strict instructions to answer from them alone and to cite sources.
4. **Safety** (`safety.py`): risky questions are intercepted before retrieval and never reach Gemini.

## The Machine Learning Part

- **Sentence Transformers** model `sentence-transformers/all-MiniLM-L6-v2` runs **locally** and converts text into 384-dimensional vectors (embeddings). Texts with similar meaning produce vectors pointing in similar directions.
- **FAISS** (Facebook AI Similarity Search) stores those vectors and finds the nearest ones to a question vector extremely fast.
- Embeddings are **L2-normalized**, so FAISS inner-product scores equal **cosine similarity** (≈0 = unrelated, ≈1 = identical), which powers the transparency scores and the refusal threshold.

## What Gemini Is Used For

Gemini (`gemini-flash-latest` by default — an alias for Google's current fast model) performs **only the final step**: turning the retrieved chunks into a natural, readable answer. The prompt explicitly forbids outside knowledge, invented policies, and any clinical advice.

---

## Setup

Requires **Python 3.10+**.

```bash
# 1. Clone / open the project folder
cd healthcare-knowledge-assistant

# 2. Create and activate a virtual environment
python -m venv venv

# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file from the template
cp .env.example .env        # (Windows: copy .env.example .env)

# 5. Build the vector index
python ingest.py

# 6. Run the app
streamlit run app.py
```

### Your Gemini API key

Open the `.env` file and paste your key:

```
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-flash-latest
ADMIN_PASSWORD=choose_a_strong_admin_password
```

`ADMIN_PASSWORD` unlocks the admin-only parts of the app (the system
status panel in the sidebar and the rebuild-index button) — regular staff
never see technical details like API key status or index files. If it is
left empty, admin features stay locked for everyone.

Get a free key at https://aistudio.google.com/apikey. **Never put the key in the Python code** — the `.env` file is git-ignored so it stays private. If the key is missing, the app shows a clear error banner.

## Adding or Editing Knowledge Base Documents

1. Add or edit `.md` or `.txt` files inside `knowledge_base/`.
2. Rebuild the index (see below) so the assistant sees the changes.

Documents should contain approved, non-clinical operational information only. Never add real patient data.

## Rebuilding the Vector Index

Either:

- Run `python ingest.py` in the terminal, **or**
- Click **Rebuild Vector Index** on the *Knowledge Base Viewer* page in the app.

## Testing the Assistant

Open the **Testing and Evaluation** page in the app. It loads `tests/sample_questions.csv` (question, expected_behavior, category) and runs each question through the *real* pipeline — the CSV is never used as an answer key. For each test you see the actual answer, retrieved sources, similarity scores, and whether it was escalated, answered, or refused.

## Safety Limitations

- The safety filter uses keyword/phrase matching — clear and explainable, but it can miss creative phrasings. A production system would add an ML safety classifier.
- Answer quality depends entirely on the documents; outdated documents mean outdated answers.
- Gemini is a cloud API: internet access is required for answer generation (embedding and search are fully local).
- This is a **training/operations assistant**, never a medical device. It refuses all diagnosis, treatment, medication, dosage, emergency, and patient-specific clinical questions.


## Future Improvements (hopefully)

- Automatic index rebuilding when documents change (file watcher).
- ML-based safety classifier alongside keyword matching.
- Conversation memory for follow-up questions.
- Role-based content access (front desk vs. billing vs. nursing).
- Admin analytics dashboard (common questions, escalation rates, knowledge gaps).
- Multi-language support.

## Project Structure

```
healthcare-knowledge-assistant/
├── app.py               # Streamlit UI (4 pages)
├── ingest.py            # Documents → chunks → embeddings → FAISS index
├── rag_pipeline.py      # Retrieval + Gemini answer generation
├── safety.py            # Risky-question detection + escalation message
├── config.py            # Paths, models, thresholds, .env loading
├── requirements.txt
├── .env.example         # Template — copy to .env and add your key
├── .gitignore
├── knowledge_base/      # Approved documents (the ONLY source of truth)
├── vector_store/        # Generated FAISS index + metadata (built by ingest.py)
└── tests/
    └── sample_questions.csv
```
