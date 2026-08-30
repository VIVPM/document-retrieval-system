# 📄 Advanced Document Retrieval System

A multi-user RAG (Retrieval-Augmented Generation) application for PDF documents. Built with a **React** frontend and a **FastAPI** backend, featuring accounts, persistent per-document conversations, open-source document extraction and hybrid sparse-dense search.

**One account → many chats → exactly one document each.** A chat session's id *is* its Pinecone namespace, so chat ↔ document ↔ namespace is 1:1. Neon/Postgres owns identity, ownership and conversation history; Pinecone holds only vectors.

---

## 🚀 Key Features

*   **Accounts & persistent chats**: JWT auth over bcrypt, DB-backed login lockout, and conversations that survive a restart — including their source citations.
*   **Session rehydration**: the in-process retriever is a pure cache. On a miss it rebuilds from Neon + Pinecone in **~11s** instead of re-ingesting the document (**~180s**), by persisting the fitted BM25 encoder and recomputing centroids from the index.
*   **Asynchronous ingestion**: upload returns `202` and processes in the background with a pollable status, because extraction runs for minutes on a real packet.
*   **Pluggable Extraction**: **AWS Textract** (TABLES + FORMS, default) or **PyMuPDF** (local, no-AI, text-layer only) via `EXTRACT_METHOD`. Contextual chunking attaches each chunk's document identity so entity-specific queries stay unambiguous.
*   **Hybrid Search Engine**: A single **Pinecone** sparse-dense index holding `gemini-embedding-2` embeddings (768-dim) alongside **BM25** sparse vectors, fused by a tunable `alpha` (0.0 = pure keyword → 1.0 = pure semantic).
*   **Conversational follow-ups**: a follow-up like *"and when does it lock?"* is condensed into a standalone question **before retrieval**, because retrieval runs before any LLM sees a prompt. History is read server-side from Neon.
*   **Answer Generation**: **gemini-2.5-flash** with thinking capped at 2048 — on an ambiguous multi-candidate question it enumerates candidates with sources instead of guessing. Thinking tokens bill at the output rate, so the cap bounds the tail (dynamic permits 24,576) without touching the ~300-token median.
*   **Two models, split on measured need**: classification and per-page boundary detection run on **gemini-2.5-flash-lite** (closed-set label, yes/no answer — and boundary detection fires once per *page*, making it the volume driver of ingest cost). Answers and query rewriting stay on flash.
*   **Semantic Routing**: Automatic query routing to specific document sections via embedding centroids — no extra LLM call.

> **Note on reranking:** a cross-encoder reranking stage (BAAI/bge-reranker-base) was built and evaluated on 250 questions, then removed — it changed answer quality by a statistically indistinguishable amount while costing a 3× over-fetch and a GPU round-trip per query. It remains a reasonable optional addition under conditions this corpus does not meet. See [Design FAQ Q2](#q2-when-is-a-reranker-actually-worth-adding) for the measurements.

---

## 🏗️ Architecture

Six layers, read top to bottom. Each arrow is a hand-off between layers; the
ingest and retrieval pipelines flow left-to-right within their own band and the
bands stack one below the other, while the shared services (data, external AI)
are reached once per layer rather than by every stage, so the flow stays
legible. Observability is cross-cutting.

```mermaid
graph TD
    User(["👤 User"])

    subgraph CLIENT ["1 · Client layer — React / Vite"]
        Land["🛬 Landing page"]
        UI["💬 Chat UI · upload gate · live ingest stepper · streamed answers"]
    end

    subgraph APP ["2 · Application layer — FastAPI (main.py)"]
        Auth["🔐 Auth · bcrypt · access + refresh JWT · lockout · per-IP rate-limit"]
        REST["🗂️ Chat endpoints · POST /message → SSE · 202 async upload + polling"]
    end

    subgraph INGEST ["3 · Ingest pipeline — background task"]
        direction LR
        Ext["📄 Extract · Textract / PyMuPDF"] --> Split["🏷️ Classify + split · flash-lite"] --> Chunk["✂️ Chunk · tables atomic · contextual"] --> Emb["🧬 Embed · 768d"] --> Up["📤 BM25 fit + Pinecone upsert"]
    end

    subgraph DATA ["4 · Data layer — Neon Postgres + Pinecone"]
        PG[("🐘 Postgres · accounts · chats · messages · bm25_params")]
        Pine[("🌲 Pinecone · one namespace per user")]
    end

    subgraph QUERY ["5 · Retrieval + answer layer"]
        direction LR
        RW["📝 Rewrite follow-up → standalone"] --> Hyb["🔍 Hybrid query · α·dense + (1−α)·sparse"] --> Ans["🤖 gemini-2.5-flash · streamed, cited"]
    end

    subgraph EXT ["6 · External AI services"]
        Gem["☁️ Google Gemini · flash / flash-lite / embeddings"]
        Tex["☁️ AWS Textract · TABLES + FORMS"]
    end

    OBS["📈 Observability · cross-cutting<br>Langfuse (LLM) + Grafana (HTTP · metrics · dashboard)"]

    User --> CLIENT
    CLIENT -->|HTTP + JWT| APP
    APP -->|identity · ownership| DATA
    APP -->|upload| INGEST
    INGEST -->|vectors + bm25_params| DATA
    DATA -->|hybrid search| QUERY
    INGEST -->|extract · classify · embed| EXT
    QUERY -->|rewrite · answer| EXT
    APP -.->|HTTP traces · metrics| OBS
    INGEST -.->|LLM traces| OBS
    QUERY -.->|LLM traces| OBS
```

---

## 🛠️ Setup & Execution

### 1. Requirements
* **Node.js**: For the React frontend.
* **Python 3.10+**: For the FastAPI backend.
* **Neon (or any Postgres)**: Accounts, chat sessions and messages.
* **Pinecone**: Serverless index, `dimension=768`, `metric=dotproduct`.
* **Google AI API key**: `gemini-embedding-2` embeddings **and** `gemini-2.5-flash` answers.
* **AWS account** *(default extraction path)*: AWS Textract API (`AmazonTextractFullAccess` — or scoped to `textract:AnalyzeDocument`). Skip if you set `EXTRACT_METHOD=pymupdf` (local, no AWS calls).

### 2. Backend Setup
1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Extraction Setup (AWS Textract)

PDF extraction runs on **AWS Textract** (TABLES + FORMS features). Answers come from the Gemini API directly, so no self-hosted LLM server is required.

#### A. AWS
1.  Create an IAM user with `AmazonTextractFullAccess` (or a scoped policy granting `textract:AnalyzeDocument`).
2.  Generate an access key pair for that user.
3.  Set the credentials in `backend/.env` (see section E).

#### B. Pinecone
Create a serverless index with **`dimension=768`** and **`metric=dotproduct`** — dotproduct is required for sparse-dense hybrid queries, and the dimension must equal `EMBED_DIM` in `llm/llm_router.py`, which also drives the embedding call itself. The backend verifies both at startup and refuses to run on a mismatch, rather than failing minutes later at upsert.

#### C. Neon (Postgres)
Create a database and copy its **pooled** connection string (the `-pooler` host — PgBouncer multiplexes many client connections onto few backends, which a free-tier compute needs). Tables are prefixed `drs_` so this schema can share a database with other projects.

Create them with either:

```bash
# from backend/
python -c "from db.database import engine, Base; import db.models; Base.metadata.create_all(engine)"
# ...or paste migrations.sql into the Neon SQL editor
```

#### D. Complete `backend/.env`

```text
# Vector store
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index
PINECONE_HOST=https://your-index-xxxxx.svc.region.pinecone.io

# Text generation provider — GEMINI or CLOUDFLARE. Required, no default.
LLM_MODEL=GEMINI

# Embeddings (always Gemini) + generation when LLM_MODEL=GEMINI
GEMINI_API_KEY=your_gemini_key

# Cloudflare Workers AI — only required when LLM_MODEL=CLOUDFLARE
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_workers_ai_token

# Extraction — pluggable. Default textract; set to "pymupdf" for local text-layer read (no AWS calls).
EXTRACT_METHOD=textract
# AWS Textract (only required when EXTRACT_METHOD=textract)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1

# Database + auth
DATABASE_URL=postgresql://user:pass@ep-xxx-pooler.region.aws.neon.tech/dbname?sslmode=require
JWT_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(48))">

# Optional
ALLOWED_ORIGINS=https://your-frontend.onrender.com,http://localhost:5173  # CORS allow-list (comma-separated)
MAX_UPLOAD_MB=3           # upload cap (MB), enforced while streaming
DAILY_MESSAGE_CAP=5       # REQUIRED — messages per account per day (1 credit = question + answer)
GEMINI_THINKING_BUDGET=2048  # fixed ceiling (default); 0 = off, -1 = dynamic
GEMINI_FAST_MODEL=gemini-2.5-flash-lite   # classification + boundary detection
CONTEXTUAL_CHUNKING=1     # attach per-document identity to each chunk (default 1, set 0 to disable)

# Ingest queue / worker (all optional — defaults shown)
MAX_CONCURRENT_JOBS=2       # ingests one worker runs at once
INGEST_MAX_ATTEMPTS=3       # tries before a job is failed for good
INGEST_LEASE_SECONDS=1800   # claim lease; must exceed the slowest ingest
WORKER_POLL_SECONDS=2       # idle poll interval
TOKEN_TTL_HOURS=24

# Observability (optional — all tracing stays off unless these are set)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
GRAFANA_OTLP_ENDPOINT=https://otlp-gateway-prod-<region>.grafana.net/otlp
GRAFANA_OTLP_AUTH=Basic <base64>          # the full Authorization header value
OTEL_SERVICE_NAME=document-retrieval-system
```


### 4. Running the Backend and the Ingest Worker

Two processes. The API serves requests; the worker runs ingestion.

```bash
# API
python -m uvicorn main:app --app-dir backend --port 8000

# Ingest worker — in a second terminal
python backend/worker.py
```

**Uploads queue but never finish without the worker running.** Ingestion is a
row in `drs_ingest_jobs`, not a background task inside the API, so a deploy or
crash of the API no longer destroys work in flight — restarting the worker just
returns its jobs to the queue. `docker compose up` starts both.

Run more than one worker if ingest is the bottleneck: the claim is a conditional
`UPDATE ... FOR UPDATE SKIP LOCKED`, so a second worker takes different jobs
rather than duplicating the first one's.

### 5. Running the Frontend
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run Dev Server:
    ```bash
    npm run dev
    ```
4.  Open `http://localhost:5173`, create an account, then start a chat and upload a PDF.

Point a build at a deployed backend with `VITE_API_URL`:

```bash
VITE_API_URL=https://your-backend.onrender.com npm run build
```

---

## 🔌 API

Every endpoint except signup and login requires `Authorization: Bearer <token>`.

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/auth/signup` | → `{ token, user_id, username }` |
| `POST` | `/api/auth/login` | 5 failures / 15 min locks the username |
| `GET` | `/api/chats` | Sidebar list, newest first |
| `POST` | `/api/chats/new` | Reuses an existing empty chat |
| `GET` | `/api/chats/{id}` | Chat + full message history |
| `POST` | `/api/chats/{id}/document` | **202** — ingests in the background |
| `GET` | `/api/chats/{id}/status` | Poll while `processing` |
| `POST` | `/api/chats/{id}/message` | Ask a question. Returns `question_asked` and `question_searched` so a rewritten follow-up is diagnosable |
| `PATCH` | `/api/chats/{id}` | Rename |
| `DELETE` | `/api/chats/{id}` | Drops the namespace **and** the rows |

**Chat lifecycle:** `awaiting_document → processing → ready | failed`

A chat that belongs to another user returns **404, not 403** — a 403 would confirm the id exists.

---

## 📈 Performance & Evaluation

The system's performance is validated using the **Ragas** evaluation framework, focusing on faithfulness, relevancy, and retrieval quality.

### Current pipeline — k=6, 250 questions (AWS Textract extraction + contextual chunking; generator `gemini-2.5-flash-lite`, judge `gemini-3.5-flash-lite`)

All three retrieval modes over the same corpus and models — raw per-question output in `results/ragas_k_6_{hybrid_new, vector_6, sparse_6}.csv`:

| Metric | Hybrid (α=0.4) | Vector (α=1) | Sparse (α=0) |
|---|---|---|---|
| Answer Correctness | **0.941** | 0.890 | 0.909 |
| Context Precision | 0.847 | 0.805 | **0.880** |
| Context Recall | **0.980** | 0.972 | 0.968 |
| Faithfulness | **0.979** | 0.954 | 0.963 |
| — | | | |
| Fully correct (AC = 1) | **222 / 250** | 207 / 250 | 210 / 250 |
| Partial (0 < AC < 1) | 21 / 250 | 25 / 250 | 27 / 250 |
| Wrong (AC = 0) | **7 / 250** | 18 / 250 | 13 / 250 |

**Hybrid wins on Answer Correctness, Context Recall, Faithfulness, and the fully-correct count**, and cuts the wrong-answer bucket to **7/250** — sparse alone leaves 13 wrong, vector alone leaves 18. Sparse edges hybrid only on `context_precision` (0.880 vs 0.847), because BM25 tends to fetch a tighter set of exact-keyword matches; the α=0.4 blend trades a bit of that precision for large gains everywhere else.

> [!NOTE]
> Evaluation was performed on a multi-document mortgage packet (21 logical documents across 50 pages). Raw per-question output for all configurations lives in `results/`.
>
> **Model provenance for these numbers:** the 250-question RAGAS runs above used **`gemini-2.5-flash-lite`** as the answer generator (and **`gemini-3.5-flash-lite`** as the judge) — chosen to keep the eval affordable. The deployed app, however, answers real user questions with **`gemini-2.5-flash`** (see `GEMINI_CHAT_MODEL` in `backend/llm/llm_router.py`).

---

## 🔥 Load testing & capacity

`backend/load_test.py` spawns the **real** app with only the retrieval + LLM boundary stubbed, so a run is free and takes seconds — it exercises the async endpoints, the connection pool, JWT auth and SSE, not the model. Idle-vs-saturated phases, a `--ramp` capacity sweep, and `--calibrate` for a few real messages.

**Capacity** (live Render instance, `--ramp`, read mix):

| Concurrent browse clients | p50 | p95 | errors |
|---|---|---|---|
| 5 | 485ms | 625ms | 0 |
| 25 | 781ms | 1313ms | 0 |
| 50 | 1578ms | 6828ms | 0 |
| 100 | 3031ms | 8640ms | 0 |

Healthy to **~25 concurrent browse clients**, **zero errors even at 100** (it degrades in latency, never fails). The ceiling is the DB connection pool: an A/B raising it from 15 → 30 (`pool_size=10 + max_overflow=20`) roughly **doubled** read throughput (~18 → ~33 req/s) and pushed the knee from ~50 to ~100. Streaming a `/message` competes for pooled connections with browse reads (`_prepare` / `_save`), so heavy answering degrades browsing ~1.4–2.3× — the pool is the lever.

---

## 📈 Observability

OpenTelemetry over OTLP, wired programmatically (not the `opentelemetry-instrument` wrapper). `openinference`'s `GoogleGenAIInstrumentor` auto-traces every Gemini call:

* **LLM spans → Langfuse + Grafana.** Each message is one `chat-message` trace with the rewrite and answer generations nested under it, tagged with user + session.
* **HTTP spans → Grafana** (a separate provider, so Langfuse stays LLM-only).
* **`chat_messages_total` metric → Grafana**, with a paste-importable dashboard and a muted error-rate alert (`backend/grafana/`).

Everything is a no-op unless the env vars are set, and nothing raises — tracing must never break a request. Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`, and `GRAFANA_OTLP_ENDPOINT` / `GRAFANA_OTLP_AUTH` (the full `Basic <base64>` header) / `OTEL_SERVICE_NAME`.

---

## 🐳 CI/CD & Docker

`.github/workflows/ci.yml` runs on every push and PR:

1. **backend** — `ruff`, `compileall`, `load_test.py --selftest` (no network).
2. **frontend** — `npm ci`, `npm run lint`, `npm run build`.
3. **docker** — build both images (no push), so a broken Dockerfile fails here, not at deploy.
4. **deploy** — only after all three pass, only on push to `main`: POSTs the Render deploy hooks (`RENDER_DEPLOY_HOOK_*` secrets), skipping gracefully if they're unset.

`docker compose up --build` runs the stack locally: `backend/Dockerfile` (`python:3.12-slim` + uvicorn, non-root, `/api/health` probe) and `frontend/Dockerfile` (Vite build → nginx). Extraction runs on AWS Textract, so the backend image needs no GPU/GL libraries.

The backend runs with `--forwarded-allow-ips *` (in the Dockerfile CMD) so slowapi's per-IP rate limits key on the real client (`X-Forwarded-For`) behind a proxy/balancer rather than the proxy's own IP — otherwise every user shares one rate-limit bucket. On a non-Docker deploy, set `FORWARDED_ALLOW_IPS=*` in the service env instead (uvicorn reads it).

---

## 📂 Project Structure

```text
document-retrieval-system/
├── backend/
│   ├── core/                        # Processing & retrieval engine
│   │   ├── document_store.py          # Orchestration + rehydrate()
│   │   ├── retriever.py               # Hybrid search, routing, namespaces
│   │   ├── pdf_processor.py           # AWS Textract integration (TABLES + FORMS)
│   │   ├── chunker.py                 # Structure-aware chunking
│   │   ├── document_classifier.py     # Doc-type & boundary detection
│   │   ├── query_rewriter.py          # Follow-up → standalone question
│   │   ├── answer_generator.py        # Grounded answer prompt
│   │   └── models.py                  # Core dataclasses
│   ├── db/                          # Neon / Postgres
│   │   ├── database.py                # Engine + session factory
│   │   └── models.py                  # accounts, chat_sessions, messages
│   ├── llm/
│   │   └── llm_router.py              # Gemini answers + embeddings
│   ├── eval/                        # Measurement harnesses
│   │   ├── model_sweep.py             # Generated questions, no judge (trust this)
│   │   ├── model_eval.py              # flash vs flash-lite, LLM-judged
│   │   └── prompt_eval.py             # Prompt-change A/B
│   ├── grafana/                     # Grafana dashboard + alert provisioning
│   ├── auth.py                      # JWT + bcrypt + refresh tokens
│   ├── observability.py             # OpenTelemetry → Langfuse + Grafana
│   ├── main.py                      # API entry point
│   ├── load_test.py                 # Capacity / responsiveness harness
│   ├── Dockerfile                   # python:3.12-slim + uvicorn
│   ├── migrations.sql               # Schema + housekeeping queries
│   ├── requirements.txt
│   └── .env                         # Keys, DB URL, worker URLs
├── frontend/
│   ├── src/
│   │   ├── api.js                   # API client, owns the JWT
│   │   ├── Landing.jsx              # Animated landing page
│   │   ├── Login.jsx                # Login / signup
│   │   ├── ChatPanel.jsx            # Upload gate, live ingest stepper, messages
│   │   ├── App.jsx                  # Shell, auth gate, chat rail
│   │   └── App.css                  # Design tokens & styles
│   ├── Dockerfile                   # Vite build → nginx
│   └── nginx.conf                   # SPA routing
├── .github/workflows/ci.yml         # lint · build · docker · gated deploy
├── docker-compose.yml               # local api + frontend stack
├── ruff.toml
├── results/                         # Ragas metrics (k=6 CSVs; results_old/ = pre-migration baselines)
└── README.md
```

---

## 📜 License

MIT License