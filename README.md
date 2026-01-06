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

```mermaid
graph TD
    User[👤 User] -->|JWT| React["⚛️ React Frontend<br>login · chat rail · chat view"]
    React -->|HTTP / JSON| FastAPI["⚡ FastAPI<br>backend/main.py"]

    subgraph Identity [Neon · Postgres]
        FastAPI -->|"owns"| PG[("🐘 accounts · chat_sessions<br>chat_messages · bm25_params")]
    end

    subgraph Ingest ["Ingest — 202, background task"]
        FastAPI -->|"upload"| Docling["📄 Docling on Modal L4<br>OCR + tables"]
        Docling --> Classify["🏷️ Classify + split<br>flash-lite · 15 doc types"]
        Classify --> Chunker["✂️ Structure-aware chunking<br>tables stay atomic · 384 / 48"]
        Chunker --> Embed["🧬 gemini-embedding-2<br>768-dim · RETRIEVAL_DOCUMENT"]
        Embed --> Pine
    end

    subgraph Query ["Query — ownership checked in Postgres first"]
        FastAPI -->|"ask"| Store["📦 Retriever cache<br>miss ⇒ rehydrate"]
        PG -.->|"bm25_params"| Store
        Store --> Router["🎯 Centroid routing<br>no LLM"]
        Router --> Hybrid["🔍 Hybrid query<br>α·dense + (1−α)·sparse"]
        Hybrid --> LLM["🤖 gemini-2.5-flash<br>thinking ≤ 2048"]
        LLM -->|"answer + citations"| PG
    end

    Hybrid <-->|"single call"| Pine[("🌲 Pinecone<br>one namespace per chat")]
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
MAX_UPLOAD_MB=25          # upload cap, enforced while streaming
GEMINI_THINKING_BUDGET=2048  # fixed ceiling (default); 0 = off, -1 = dynamic
GEMINI_FAST_MODEL=gemini-2.5-flash-lite   # classification + boundary detection
DOCLING_PIPELINE=classic  # or "vlm" for granite-docling-258M (see below)
USE_MODAL_LLM=0           # 1 enables the Gemma-2 fallback
TOKEN_TTL_HOURS=24
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

### Q4: Why is `DOCLING_PIPELINE=vlm` (granite-docling-258M) available but not the default?

Extraction, not retrieval or the LLM, is this project's quality ceiling: the classic pipeline flattens multi-column form tables (`label | value | label | value`) with the label cell duplicated across columns. `ibm-granite/granite-docling-258M` looked like the fix — a drop-in Docling VLM pipeline, and IBM report **TEDS 0.97 structure / 0.96 with-content on FinTabNet**, a *financial* table benchmark, against SmolDocling's 0.82 / 0.76.

Measured on the same 7-page test packet (`backend/eval/extract_eval.py`):

| | classic | vlm |
|---|---|---|
| characters extracted | 11,747 | **7,138** |
| page-1 fee table | 22 rows | **empty block** |
| funds-to-close table | 9 rows | **gone** |
| `Interest Rate` | `Interest Rate:` (value detached) | **`Interest Rate: 4.250 %`** ✅ |
| simple grids (pp. 2, 5) | — | byte-identical |
| runtime | 47s | 287s |

It **does** fix text form fields — label and value arrive in one block, where classic splits them with nothing lexically tying them together. But it emitted page 1's fee table as a *single empty table*, taking `ORIGINATION`, `Underwriting`, `Appraisal`, `95,641.53`, `475,000` and every other fee figure with it.

Losing the fee table outright is far worse than having its labels duplicated, and the fee sheet is the most-queried document in a mortgage packet. **A published benchmark on a public dataset did not transfer to this layout** — which is the general lesson, and the same one Q2 records about the reranker. It stays deployed behind the flag because the text-field win is real and a hybrid (classic tables + VLM text blocks) is the obvious next thing to try.

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
│   │   ├── model_eval.py              # flash vs flash-lite, LLM-judged
│   │   └── extract_eval.py            # classic vs granite-docling
│   ├── auth.py                      # JWT + bcrypt
│   ├── main.py                      # API entry point
│   ├── migrations.sql               # Schema + housekeeping queries
│   ├── requirements.txt
│   └── .env                         # Keys, DB URL, worker URLs
├── frontend/
│   └── src/
│       ├── api.js                   # API client, owns the JWT
│       ├── Login.jsx                # Login / signup
│       ├── ChatPanel.jsx            # Upload gate, polling, messages
│       ├── App.jsx                  # Shell, auth gate, chat rail
│       └── App.css                  # Design tokens & styles
├── notebooks/                       # R&D and evaluation (gitignored)
├── results/                         # Ragas metrics output
└── README.md
```