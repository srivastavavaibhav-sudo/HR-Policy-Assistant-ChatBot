# HR Policy Assistant

An Agentic AI assistant that answers employee HR-policy questions using
**Retrieval-Augmented Generation (RAG)** and **tool-calling**, built with
FastAPI, LangChain, LangGraph, and ChromaDB. Runs entirely on **free**
infrastructure: [Groq](https://console.groq.com) for the LLM (free API,
no card required) and a local HuggingFace sentence-transformer model for
embeddings (runs on your machine, no API cost at all).

## Architecture

```
User → FastAPI /chat → LangGraph ReAct Agent (OpenAI LLM)
                              │
                ┌─────────────┴─────────────┐
                │                            │
        search_hr_policy tool       calculate_leave_balance tool
                │
        ChromaDB vector search
                │
        Embedded HR policy docs (data/sample_policies/*.md)

Conversation history persisted to SQLite via SQLAlchemy.
```

- **RAG layer** (`app/rag/`): chunks and embeds HR policy markdown docs into a
  local ChromaDB store using a free local `sentence-transformers` model;
  retrieves the most relevant chunks per query.
- **Agent layer** (`app/agents/`): a LangGraph ReAct agent (running on Groq's
  free `llama-3.3-70b-versatile`) that decides, turn by turn, whether to call
  `search_hr_policy` (grounds answers in real policy text) or
  `calculate_leave_balance` (a deterministic tool, not an LLM guess).
- **API layer** (`app/main.py`): FastAPI exposes `/chat` and `/history/{thread_id}`;
  every turn is persisted to SQLite.

## Setup

1. Get a **free** Groq API key: sign up at https://console.groq.com (no card
   needed), then go to "API Keys" → "Create API Key".

```bash
git clone <your-repo-url>
cd hr-policy-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your GROQ_API_KEY into it
```

The first run will download the local embedding model (~90 MB) automatically
— no account or key needed for that part, it's fully free and offline after
the first download.

## Run

```bash
# 1. Ingest the sample HR policy docs into the vector store (one-time)
python -m app.rag.ingest

# 2. Start the API
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many days of paternity leave do I get?", "thread_id": "demo-1"}'
```

## Tests

```bash
pytest
```

## Extending this project

- Add more policy documents to `data/sample_policies/`.
- Add Elasticsearch as a hybrid keyword+semantic search layer alongside Chroma.
- Add a `/feedback` endpoint to log whether answers were helpful, and use it
  to evaluate retrieval quality over time.
- Swap SQLite for PostgreSQL in production (`DATABASE_URL` already supports it).

## Pushing to your own GitHub

```bash
git init
git add .
git commit -m "Initial commit: HR Policy Assistant (RAG + agentic tool-calling)"
git branch -M main
git remote add origin https://github.com/<your-username>/hr-policy-assistant.git
git push -u origin main
```
