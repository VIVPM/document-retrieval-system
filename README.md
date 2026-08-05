# 📄 Advanced Document Retrieval System

A multi-user RAG (Retrieval-Augmented Generation) application for PDF documents. Built with a **React** frontend and a **FastAPI** backend, featuring accounts, persistent per-document conversations, open-source document extraction and hybrid sparse-dense search.

**One account → many chats → exactly one document each.** A chat session's id *is* its Pinecone namespace, so chat ↔ document ↔ namespace is 1:1. Neon/Postgres owns identity, ownership and conversation history; Pinecone holds only vectors.

---

## 🚀 Key Features

*   **Accounts & persistent chats**: JWT auth over bcrypt, DB-backed login lockout, and conversations that survive a restart — including their source citations.
*   **Session rehydration**: the in-process retriever is a pure cache. On a miss it rebuilds from Neon + Pinecone in **~11s** instead of re-ingesting the document (**~180s**), by persisting the fitted BM25 encoder and recomputing centroids from the index.
*   **Asynchronous ingestion**: upload returns `202` and processes in the background with a pollable status, because extraction runs for minutes on a real packet.
*   **Open-Source Extraction**: Leverages **Docling** for high-fidelity, structure-aware PDF parsing.
*   **Hybrid Search Engine**: A single **Pinecone** sparse-dense index holding `gemini-embedding-2` embeddings (768-dim) alongside **BM25** sparse vectors, fused by a tunable `alpha` (0.0 = pure keyword → 1.0 = pure semantic).
*   **Conversational follow-ups**: a follow-up like *"and when does it lock?"* is condensed into a standalone question **before retrieval**, because retrieval runs before any LLM sees a prompt. History is read server-side from Neon.
*   **Answer Generation**: **gemini-2.5-flash** with thinking capped at 2048 — on an ambiguous multi-candidate question it enumerates candidates with sources instead of guessing. Thinking tokens bill at the output rate, so the cap bounds the tail (dynamic permits 24,576) without touching the ~300-token median. Modal/Gemma-2-9B remains as an opt-in fallback (`USE_MODAL_LLM=1`).
*   **Two models, split on measured need**: classification and per-page boundary detection run on **gemini-2.5-flash-lite** (closed-set label, yes/no answer — and boundary detection fires once per *page*, making it the volume driver of ingest cost). Answers and query rewriting stay on flash.
*   **Semantic Routing**: Automatic query routing to specific document sections via embedding centroids — no extra LLM call.

> **Note on reranking:** a cross-encoder reranking stage (BAAI/bge-reranker-base on Modal) was built and evaluated on 250 questions, then removed — it changed answer quality by a statistically indistinguishable amount while costing a 3× over-fetch and a GPU round-trip per query. It remains a reasonable optional addition under conditions this corpus does not meet. See [Design FAQ Q2](#q2-when-is-a-reranker-actually-worth-adding) for the measurements.

---

## 🏗️ Architecture

Six layers, read straight down. Each layer hands off to the one below it; the
ingest pipeline and the retrieval path each stack their own stages vertically.
The two dotted edges are the only cross-cuts: ingest reaches the Modal/Gemini
workers, and the app fans traces out to observability.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 18, 'rankSpacing': 24}}}%%
flowchart TB
    User(["👤 User"])

    subgraph C ["1 · Client — React / Vite"]
        UI["Landing · chat · live ingest · streamed answers"]
    end

    subgraph A ["2 · Application — FastAPI"]
        API["Auth · rate-limit · SSE /message · 202 upload"]
    end

    subgraph I ["3 · Ingest pipeline — background task"]
        direction TB
        i1["Extract · docling / pymupdf"] --> i2["Classify + split · flash-lite"] --> i3["Chunk · tables atomic"] --> i4["Embed · 768d"] --> i5["BM25 fit + Pinecone upsert"]
    end

    subgraph D ["4 · Data — Neon + Pinecone"]
        PG[("Postgres · chats · bm25_params")]
        Pine[("Pinecone · 1 ns/user")]
    end

    subgraph Q ["5 · Retrieval + answer"]
        direction TB
        q1["Rewrite follow-up → standalone"] --> q2["Hybrid query · α-fused"] --> q3["gemini-2.5-flash · streamed, cited"]
    end

    subgraph X ["6 · External AI"]
        Gem["Gemini · flash / embeddings"]
        Mod["Modal · Docling GPU (L4)"]
    end

    User --> C --> A --> I --> D --> Q --> X
    I -.-> X
    A -.-> OBS["📈 Langfuse + Grafana"]
```

---

## 🛠️ Setup & Execution

### 1. Requirements
* **Node.js**: For the React frontend.
* **Python 3.10+**: For the FastAPI backend.
* **Neon (or any Postgres)**: Accounts, chat sessions and messages.
* **Pinecone**: Serverless index, `dimension=768`, `metric=dotproduct`.
* **Google AI API key**: `gemini-embedding-2` embeddings **and** `gemini-2.5-flash` answers.
* **Modal account**: Docling extraction (GPU). The Gemma-2 LLM server is optional.

### 2. Backend Setup
1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Worker Setup (Modal)

**Modal** runs Docling extraction on a GPU. Answers come from the Gemini API directly, so no self-hosted LLM server is required.

#### A. Modal Deployment (Docling, and optionally the LLM)
1.  **Initialize Modal**: `pip install modal && modal setup`.
2.  **Create Secrets**: In the Modal dashboard, create a secret named `huggingface-secret` containing your `HF_TOKEN`.
3.  **Deploy the Stack**:
    ```bash
    # 1. LLM Server (Gemma-2 9B)
    modal run modal/modal_llm_server.py::download_model
    modal deploy modal/modal_llm_server.py

    # 2. Docling Worker (PDF Extraction)
    modal deploy modal/modal_docling_worker.py
    ```
4.  **Finalize .env**: Copy the deployment URLs into your backend `.env`:
    ```text
    LLM_URL=https://your-llm-server.modal.run
    DOCLING_URL=https://your-docling-worker.modal.run
    ```

#### C. Pinecone
Create a serverless index with **`dimension=768`** and **`metric=dotproduct`** — dotproduct is required for sparse-dense hybrid queries, and the dimension must equal `EMBED_DIM` in `llm/llm_router.py`, which also drives the embedding call itself. The backend verifies both at startup and refuses to run on a mismatch, rather than failing minutes later at upsert.

#### D. Neon (Postgres)
Create a database and copy its **pooled** connection string (the `-pooler` host — PgBouncer multiplexes many client connections onto few backends, which a free-tier compute needs). Tables are prefixed `drs_` so this schema can share a database with other projects.

Create them with either:

```bash
# from backend/
python -c "from db.database import engine, Base; import db.models; Base.metadata.create_all(engine)"
# ...or paste migrations.sql into the Neon SQL editor
```

#### E. Complete `backend/.env`

```text
# Vector store
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index
PINECONE_HOST=https://your-index-xxxxx.svc.region.pinecone.io

# Embeddings + LLM
GEMINI_API_KEY=your_gemini_key
LLM_URL=https://your-llm-server.modal.run
DOCLING_URL=https://your-docling-worker.modal.run

# Database + auth
DATABASE_URL=postgresql://user:pass@ep-xxx-pooler.region.aws.neon.tech/dbname?sslmode=require
JWT_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(48))">

# Optional
ALLOWED_ORIGINS=https://your-frontend.onrender.com,http://localhost:5173  # CORS allow-list (comma-separated)
MAX_UPLOAD_MB=3           # upload cap (MB), enforced while streaming
GEMINI_THINKING_BUDGET=2048  # fixed ceiling (default); 0 = off, -1 = dynamic
GEMINI_FAST_MODEL=gemini-2.5-flash-lite   # classification + boundary detection
DOCLING_PIPELINE=classic  # or "vlm" for granite-docling-258M (see below)
USE_MODAL_LLM=0           # 1 enables the Gemma-2 fallback
TOKEN_TTL_HOURS=24

# Observability (optional — all tracing stays off unless these are set)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
GRAFANA_OTLP_ENDPOINT=https://otlp-gateway-prod-<region>.grafana.net/otlp
GRAFANA_OTLP_AUTH=Basic <base64>          # the full Authorization header value
OTEL_SERVICE_NAME=document-retrieval-system
```

> [!IMPORTANT]
> **Deployment Workflow**:
> *   **First Time**: Run `download_model` **then** `deploy`. This ensures the Volume is populated before the server starts.
> *   **Subsequent Changes**: Only run `modal deploy`. You do NOT need to redownload unless you change the `MODEL_NAME` in the script.
> *   **Why Deploy?**: `modal run` gives a temporary development URL. `modal deploy` creates the permanent production URL required for your `.env`.

### 4. Running the Frontend
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

### Evaluation Metrics (k=5, 250 questions, `results/ragas_results_5_hybrid.csv`)

| Metric | Hybrid (shipped) | Vector only |
|---|---|---|
| Faithfulness | **0.892** | 0.804 |
| Answer Correctness | **0.836** | 0.746 |
| Context Precision | **0.856** | 0.697 |
| Context Recall | **0.964** | 0.880 |

Hybrid sparse-dense retrieval beats pure vector search on every metric — which is what justifies the BM25 half of the index.

> [!NOTE]
> Evaluation was performed on a diverse set of complex financial and legal documents to ensure robustness across different domains. Raw per-question output for all configurations lives in `results/`.

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

`docker compose up --build` runs the stack locally: `backend/Dockerfile` (`python:3.12-slim` + uvicorn, non-root, `/api/health` probe) and `frontend/Dockerfile` (Vite build → nginx). Docling runs on Modal, so the backend image needs no GPU/GL libraries.

The backend runs with `--forwarded-allow-ips *` (in the Dockerfile CMD) so slowapi's per-IP rate limits key on the real client (`X-Forwarded-For`) behind a proxy/balancer rather than the proxy's own IP — otherwise every user shares one rate-limit bucket. On a non-Docker deploy, set `FORWARDED_ALLOW_IPS=*` in the service env instead (uvicorn reads it).

---

## ❓ Design FAQ

Questions that come up when reading the retrieval code, answered from the implementation and the evaluation data rather than from general RAG folklore.

### Q1: After the semantic and BM25 candidates are shortlisted, how are they combined?

**They are never shortlisted separately.** This is the most common wrong mental model of this pipeline. There is no "top-N from dense, top-N from sparse, merge, then cut to k". There is **one index, one query, one score, one ranking**.

Each chunk is stored as a single Pinecone record carrying *both* a dense vector and a sparse vector (`retriever.py` → `build_indices`). At query time Pinecone computes one score per chunk, server-side:

```
score(chunk) = α·(q_dense · c_dense) + (1−α)·(q_sparse · c_sparse)
```

`top_k` is applied to **that** number. Fusion happens *before* selection, inside the index — not after two separate retrievals.

Pinecone exposes no `alpha` parameter, because the server only knows how to dot-product. So the weighting is applied by pre-scaling the **query** vectors before they are sent (`retriever.py` → `_scale_vectors`):

```python
scaled_dense  = [v * alpha       for v in dense_vec]
scaled_sparse = [v * (1 - alpha) for v in sparse_vec["values"]]
```

This is exactly Pinecone's documented `hybrid_score_norm` convex-combination helper.

Two consequences worth internalising:

* **Why `alpha` is structurally necessary.** Dense vectors are L2-normalised on both write and query, so the dense term is effectively cosine, bounded in `[-1, 1]`. BM25 sparse weights are **unbounded positive**. Without `alpha`, the sparse term would simply dominate every query. `alpha` is a scale-reconciliation device first and a user preference knob second.
* **Neither modality gets a guaranteed slot.** A chunk that is the #1 BM25 hit but only mediocre semantically can fail to appear in the results at all, because it receives exactly one combined score. This is a real behavioural difference from Reciprocal Rank Fusion — see Q3.

This is also why the index must be created with `metric=dotproduct`: cosine and euclidean indexes cannot serve sparse-dense queries at all. The backend verifies this at startup.

---

### Q2: When is a reranker actually worth adding?

A cross-encoder reranker was built, measured, and removed from this project. The obvious explanation — "the corpus is small, so there was nothing left to fix" — turns out to be **wrong**, and the real reason is more useful.

The corpus is ~67–87 chunks and `recall@5` was **0.964**, distributed almost binary: 241 of 250 questions at perfect recall, 9 at zero. So the apparent story is that there was no headroom. But testing what the reranker actually did to those 9:

| Outcome | Count |
|---|---|
| Questions where hybrid retrieved nothing relevant | 9 |
| **…of those, rescued by the reranker** | **8** |
| **Rankings the reranker newly broke** (perfect → zero) | **9** |

The reranker was doing substantial work in *both directions*. It rescued 8 of 9 genuine misses — its mechanism functioning exactly as designed, promoting a chunk from rank 6–15 into the top 5 — while simultaneously destroying 9 rankings that were already perfect. Net **−1**. The same churn shows up in the paired per-question comparison (answer correctness: 70 wins, 54 losses, 126 ties; overall delta −0.003, p=0.85).

The governing condition is therefore sharper than corpus size:

> **A reranker helps only if it is meaningfully more accurate than your first stage on your data.** If the two are roughly equal in quality, over-fetching merely hands the reranker more opportunities to be wrong, and the result is churn rather than gain.

In this pipeline the first stage is strong and the reranker is not plausibly stronger: modern Gemini embeddings **plus** BM25 exact-term matching, against a corpus of exact-value lookups ("what is Total Loan Costs?"). BM25's lexical precision is near-ideal for that task, while `bge-reranker-base` is a ~278M general-domain cross-encoder with no exposure to mortgage documents.

**Conditions under which a reranker does earn its keep:**

| Condition | Why it matters |
|---|---|
| `recall@k` ≪ `recall@(k×N)` | The entire mechanism is salvaging deep recall into shallow precision. Measure this first — it is a hard ceiling on any possible gain. |
| The reranker genuinely outranks your first stage **on your domain** | Non-negotiable. A general-purpose reranker layered over a strong domain-appropriate retriever frequently loses. |
| Query–document **term interaction** matters | Bi-encoders embed query and document independently and cannot model interaction. Negation, "X but not Y", multi-hop conditions, comparatives. Simple field lookups do not need this. |
| Many **near-duplicate** candidates | Large corpora where dozens of chunks look near-identical to a bi-encoder. Nothing to disambiguate at 67 chunks. |
| The **context window is the binding constraint** | If only 3 chunks fit, they had better be the right 3. |
| Your embedding model is **domain-mismatched** | A cross-encoder can compensate for a weak first stage. |

**The cheap diagnostic, before deploying one:** compare `recall@k` against `recall@(k×3)`. If they are close, a reranker cannot help — it only ever reorders what was already fetched.

---

### Q3: When should RRF be used, and is a reranker still needed alongside it?

**Reciprocal Rank Fusion** — `score = Σᵢ 1/(60 + rankᵢ)` — discards score magnitude and uses only position.

**Use RRF when scores cannot be compared across systems.** That is its entire purpose; it is scale-free by construction. Concretely:

* Retrievers are **physically separate** — e.g. Elasticsearch BM25 alongside a separate vector database. You genuinely have two ranked lists rather than one index, so there is no other option.
* You are fusing **three or more heterogeneous sources** — dense, sparse, graph, a second embedding model. RRF handles N lists uniformly; convex combination becomes awkward past two.
* You have **no labelled data to tune `alpha`**. RRF's only knob is the constant (conventionally 60) and it is famously insensitive to it.
* You want **robustness to one retriever misbehaving**. Each list contributes at most ~1/60, so no single system can dominate the fused ranking.

**Use α-weighted score fusion (this project) when:**

* A single index computes both signals natively — the combined score comes for free and there is no fusion step to implement.
* **Magnitude carries signal.** This is precisely why this project migrated *off* RRF. RRF treats rank 3 as rank 3 whether it scored 0.95 or 0.40; for exact-numeric lookups, "matched this figure exactly" versus "matched it weakly" is exactly the information worth keeping.
* You want a tunable knob exposed to the user (the `alpha` slider).

#### Does RRF remove the need for a reranker?

**No — they are different stages, not competing options.**

* **RRF / α-fusion is a *merge* strategy.** It decides how to combine cheap relevance signals produced by models that never see query and document together.
* **A reranker is a *precision* stage.** A cross-encoder jointly encodes `(query, chunk)` and can model term interaction. It is orders of magnitude slower per pair, which is exactly why it can only run over a shortlist.

The canonical production funnel is `cheap retrievers → RRF merge → cross-encoder rerank → LLM`, with each stage narrowing the candidate set.

There is a non-obvious corollary:

> **RRF *increases* the value of a reranker, relative to score fusion.**

Because RRF deliberately discards magnitude, its ordering is coarse — it knows rank order but not by how much. A reranker restores fine-grained ordering at the top, so an RRF pipeline leaves *more* for a reranker to fix. The α-fusion used here preserves magnitude, so the top-k ordering is already fine-grained. That is a second, independent reason the reranker found little to improve in this project.

**RRF alone suffices when** `recall@k` is already high, latency matters, there is no GPU budget, or the reranker is not stronger than the first stage. **Add a reranker on top when** the corpus is large enough that `recall@k` is genuinely poor, over-fetching to 50–100 candidates is cheap, and you have verified *on your own data* that the cross-encoder actually outranks your first stage. That last clause is the one most often skipped — and it is the one that decided the outcome here.

### Q4: How is multi-column form extraction handled, and why is `DOCLING_PIPELINE=vlm` (granite-docling-258M) available but not the default?

Extraction *was* this project's quality ceiling. A mortgage fee sheet is a two-column **key-value form**, and the classic pipeline returned the value detached from its label — `Interest Rate:` in one block, `4.250 %` in another — so a question naming the field retrieved a chunk that did not contain the number.

**The real cause was a discarded coordinate, not a weak model.** The value was never lost: `Interest Rate:` (x=237) and `4.250 %` (x=298) sit on the *same visual row* (y=842.9 vs 842.8), but the worker exposed only each block's **y**-coordinate, so `pdf_processor` sorted by y alone — dropping every label into one group and every value into another. The fix is one line in the Modal worker (emit `x = bbox.l`) plus `pdf_processor._order_blocks()`, which groups blocks into visual **rows** (y within a tolerance) and orders each row left-to-right by x. `Interest Rate: 4.250 %` is reassembled — verified end-to-end (the question now answers *4.250%*), at classic's speed, with tables untouched.

**Why not the VLM (`granite-docling-258M`) instead?** It looked like the fix — IBM report **TEDS 0.97 structure / 0.96 with-content on FinTabNet**, a *financial* table benchmark — and it *does* keep a label with its value. But measured on the test packets it emitted the fee tables as *empty blocks*, taking `ORIGINATION`, `Underwriting`, `95,641.53` and every other fee figure with them:

| | classic (was) | vlm | **row-grouping (shipped)** |
|---|---|---|---|
| `Interest Rate` value | detached ❌ | `Interest Rate: 4.250 %` ✅ | `Interest Rate: 4.250 %` ✅ |
| fee / funds-to-close tables | kept ✅ | **emptied** ❌ | kept ✅ |
| runtime | ~46s | ~2× (per-page VLM) | ~46s |

Losing the fee table is far worse than a detached field, and the fee sheet is the most-queried document in a packet. **A published benchmark on a public dataset did not transfer to this layout** — the same lesson Q2 records about the reranker. So the VLM stays behind the flag (its text-field win is real if you ever want it), but the row-grouping fix — a coordinate Docling already computed — supersedes the "classic tables + VLM text" hybrid it once pointed at.

---

## 📂 Project Structure

```text
document-retrieval-system/
├── backend/
│   ├── core/                        # Processing & retrieval engine
│   │   ├── document_store.py          # Orchestration + rehydrate()
│   │   ├── retriever.py               # Hybrid search, routing, namespaces
│   │   ├── pdf_processor.py           # Docling integration (classic | vlm)
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
│   ├── modal/                       # Cloud deployment scripts
│   │   ├── modal_llm_server.py        # vLLM hosting (Gemma-2)
│   │   └── modal_docling_worker.py    # Serverless PDF extraction
│   ├── eval/                        # Measurement harnesses
│   │   ├── model_sweep.py             # Generated questions, no judge (trust this)
│   │   ├── model_eval.py              # flash vs flash-lite, LLM-judged
│   │   ├── prompt_eval.py             # Prompt-change A/B
│   │   ├── extract_eval.py            # classic vs granite-docling
│   │   └── hybrid_extract_eval.py     # classic vs vlm vs row-grouping
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
├── notebooks/                       # R&D and evaluation (gitignored)
├── results/                         # Ragas metrics output
└── README.md
```