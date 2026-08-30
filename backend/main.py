"""
main.py — FastAPI backend for the Document Retrieval System.

Model: one account → many chat sessions → one document each.
Pinecone holds one namespace per USER, containing every chat they own. Vector
ids are prefixed `{chat_id}#` and carry chat_id in metadata, so a query is
scoped to one chat and a chat is deleted by listing that prefix.

Endpoints
  POST   /api/auth/signup
  POST   /api/auth/login
  GET    /api/chats                     — sidebar list
  POST   /api/chats/new                 — empty chat, awaiting a document
  GET    /api/chats/{id}                — messages + document stats
  POST   /api/chats/{id}/document       — upload; ingests in the background
  GET    /api/chats/{id}/status         — poll while status='processing'
  POST   /api/chats/{id}/message        — ask a question
  PATCH  /api/chats/{id}                — rename
  DELETE /api/chats/{id}                — drop namespace + rows

Ownership is enforced in Postgres before Pinecone is ever touched: a chat that
does not belong to the caller is reported as 404, not 403, so the API does not
leak which chat ids exist.
"""

import asyncio
import json
import os
import re
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Optional

import multiprocess.resource_tracker as rt


# Suppress harmless Windows multiprocess exit error
def _silent_del(self):
    try:
        self._stop()
    except Exception:
        pass


rt.ResourceTracker.__del__ = _silent_del

from dotenv import load_dotenv

# Load .env before importing anything that reads DATABASE_URL / JWT_SECRET at
# import time.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import (Depends, FastAPI, File, HTTPException,
                     Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel, ConfigDict, Field, field_validator

from auth import (LOGIN_LOCKOUT_MINUTES, MAX_LOGIN_FAILURES, REFRESH_TTL_DAYS,
                  create_token, get_current_user, hash_password,
                  hash_refresh_token, make_refresh_token, needs_rehash,
                  verify_password)
from core.answer_generator import (LLM_EMPTY_ANSWER, build_sources,
                                    stream_answer)
from core.document_store import EnhancedDocumentStoreHybrid
from core.query_rewriter import MAX_HISTORY_MESSAGES, rewrite_standalone
from db.database import Base, SessionLocal, engine
from db.models import (Account, ChatMessage, ChatSession, IngestJob,
                       LoginFailure, RefreshToken, now_ist)
import job_queue
from llm.llm_router import embed_model
from observability import (flush as trace_flush, init_http_tracing,
                           init_metrics, init_observability, record_message,
                           set_output, trace_message)

Base.metadata.create_all(bind=engine)


def _recover_orphaned_ingests() -> None:
    """Reconcile chats stuck on 'processing' with the queue, at startup.

    Ingest used to run in a BackgroundTask, so a restart meant the work was
    genuinely gone and the only honest move was to fail every 'processing'
    chat. With a queue the work is a row, so failing them would throw away
    jobs a worker is about to run. Now only chats with NO live job are failed;
    the rest are left for the worker, which owns them.

    Still needed despite the queue: a crash between the upload commit and the
    enqueue is impossible (one transaction), but a chat whose job row was
    manually removed, or predates the queue entirely, would otherwise poll for
    ever and resist deletion.
    """
    db = SessionLocal()
    try:
        stranded = db.query(ChatSession).filter(
            ChatSession.status == "processing").all()
        orphaned = 0
        for chat in stranded:
            live = db.query(IngestJob).filter(
                IngestJob.chat_id == chat.id,
                IngestJob.status.in_(("queued", "running")),
            ).first()
            if live is not None:
                continue
            chat.status = "failed"
            chat.stage = None
            chat.error = "Ingestion was interrupted and has no queued job. Re-upload the document."
            chat.updated_at = now_ist()
            orphaned += 1
        if orphaned:
            db.commit()
            print(f"🧹 Failed {orphaned} chat(s) processing with no queued job.")
        left = len(stranded) - orphaned
        if left:
            print(f"⏳ {left} chat(s) still processing with a live job — the worker owns them.")
    except Exception as e:
        db.rollback()
        print(f"⚠️  Could not reconcile stranded ingests: {type(e).__name__}: {e}")
    finally:
        db.close()


_recover_orphaned_ingests()

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "3"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

def _ns(db, user_id: int) -> str:
    """Pinecone namespace for an account: the account's username."""
    acc = db.query(Account).filter(Account.id == user_id).first()
    return acc.username if acc else f"user_{user_id}"


def _title_from_question(q: str) -> str:
    """A chat title from the user's first question, trimmed to fit the rail."""
    q = " ".join((q or "").split())
    return (q[:48] + ("…" if len(q) > 48 else "")) or "New Chat"


# Pure cache: a miss rehydrates from Postgres + Pinecone. Bounded because each
# entry holds a fitted BM25 encoder in memory.
MAX_CACHED_RETRIEVERS = 20
_retrievers: "OrderedDict[str, EnhancedDocumentStoreHybrid]" = OrderedDict()


def sanitize(obj):
    """
    Recursively convert non-JSON-serializable types to native Python.
    Handles: numpy scalars/arrays, dataclasses, objects with __dict__,
    sets, bytes, and arbitrary objects.
    """
    import numpy as np
    from dataclasses import asdict, is_dataclass

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, set):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if is_dataclass(obj) and not isinstance(obj, type):
        return sanitize(asdict(obj))
    if hasattr(obj, '__dict__'):
        return sanitize(vars(obj))
    # Last resort — stringify
    return str(obj)


# ── App & CORS ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Document Retrieval System API",
    description="Hybrid RAG pipeline for intelligent multi-document Q&A",
    version="2.0.0",
)

# CORS — origins from env (comma-separated); default keeps the deployed frontend
# and local dev working without a code change.
_DEFAULT_ORIGINS = "https://document-retrieval-system-frontend.onrender.com,http://localhost:5173"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Observability ─────────────────────────────────────────────────────────────
# All no-ops unless their env vars are set (see observability.py):
init_observability()      # LLM pipeline (rewrite + answer) -> Langfuse + Grafana
init_http_tracing(app)    # HTTP-layer spans -> Grafana Cloud
init_metrics()            # chat_messages_total counter -> Grafana Cloud

# ── Rate limits ───────────────────────────────────────────────────────────────
# Applied to the four endpoints that cost real money or guard the account, not
# to every route — reads are cheap and limiting them only breaks the sidebar.
#
#   signup/login   an attacker's endpoints. The lockout is per-USERNAME and
#                  DB-backed, so it does nothing against one host spraying many
#                  usernames; these limits are per-IP and cover that gap.
#   document       one upload = Textract per-page + one LLM call per page. By far
#                  the most expensive thing an authenticated user can trigger.
#   message        one LLM call per question, plus a rewrite on follow-ups.
#
# Keyed on request.client.host, which uvicorn fills from X-Forwarded-For only
# for peers listed in --forwarded-allow-ips (default 127.0.0.1). Deploy behind
# a proxy WITHOUT setting that and every user shares the proxy's bucket, so the
# whole app caps at RATE_UPLOAD. Do not re-read the header here — that would be
# a second, ungated trust path. Counters are in-process, so each worker holds
# its own; upgrade_roadmap.txt PART 3 has the Redis version.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

RATE_SIGNUP = os.getenv("RATE_SIGNUP", "5/hour")
RATE_LOGIN = os.getenv("RATE_LOGIN", "10/minute")
RATE_UPLOAD = os.getenv("RATE_UPLOAD", "10/hour")
RATE_MESSAGE = os.getenv("RATE_MESSAGE", "30/minute")

# Daily chat credits. 1 credit = one question and its answer, so this counts
# the user's own turns, not the assistant's. Rate limits above throttle bursts;
# this bounds an account's total spend per day, which is a different job.
# Required with no default — a forgotten deploy setting should fail loudly
# rather than run on a guessed limit.
_daily_cap = os.getenv("DAILY_MESSAGE_CAP")
if not _daily_cap:
    raise RuntimeError(
        "DAILY_MESSAGE_CAP must be set (chat messages each account may "
        "send per day). Set it in backend/.env locally and in the environment "
        "of whatever hosts this in production."
    )
try:
    DAILY_MESSAGE_CAP = int(_daily_cap)
except ValueError:
    raise RuntimeError(
        f"DAILY_MESSAGE_CAP must be a whole number, got {_daily_cap!r}")
if DAILY_MESSAGE_CAP < 1:
    raise RuntimeError(
        f"DAILY_MESSAGE_CAP must be at least 1, got {DAILY_MESSAGE_CAP}")


# ── Schemas ───────────────────────────────────────────────────────────────────
# The username IS a Gmail address. Anchored on both ends so "notgmail.com" and
# "me@gmail.com.attacker.net" are both rejected — a bare endswith("gmail.com")
# accepts the first of those.
GMAIL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._+-]*[a-z0-9])?@gmail\.com$")


def _normalise_email(v: str) -> str:
    """Trim and lowercase. Gmail is case-insensitive, so storing the raw case
    would let Foo@ and foo@ become two accounts for one mailbox."""
    return v.strip().lower()


class SignupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        v = _normalise_email(v)
        if len(v) > 254:
            raise ValueError("Email address is too long.")
        if not GMAIL_RE.match(v):
            raise ValueError("Username must be a Gmail address ending in @gmail.com.")
        return v

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        # Normalised the same way as signup so case does not break login, but
        # deliberately NOT format-checked: a rejected format here would answer
        # "that is not a valid address" instead of "wrong credentials".
        return _normalise_email(v)


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str


class RenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 120:
            raise ValueError("Title must be 1-120 characters.")
        return v


class MessageRequest(BaseModel):
    # alpha is bounded because _scale_vectors multiplies the sparse half by
    # (1 - alpha): outside 0..1 that factor goes negative, which inverts the
    # ranking on a dotproduct index and returns the WORST keyword matches
    # first. It raises nothing -- the answer is simply wrong and looks normal.
    # num_chunks is bounded so one request cannot pull a whole document into a
    # priced prompt; `summarize` is the supported way to ask for that.
    model_config = ConfigDict(extra="forbid")

    question: str
    filter_type: Optional[str] = None
    # 6, not 4: chunks shrank from 512 to 384 tokens, so the same k now
    # retrieves ~25% less text. Coverage is k x chunk_size, not k.
    num_chunks: int = Field(default=6, ge=1, le=20)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    # Whole-document summary: bypass top-k retrieval and feed every chunk.
    summarize: bool = False

    @field_validator("question")
    @classmethod
    def _question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Question is too long (max 2000 characters).")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _owned_chat(db, chat_id: str, user_id: int) -> ChatSession:
    """Load a chat the caller owns, or 404.

    404 rather than 403 for someone else's chat: a 403 would confirm the id
    exists, which is an enumeration oracle.
    """
    chat = db.query(ChatSession).filter(
        ChatSession.id == chat_id, ChatSession.user_id == user_id
    ).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return chat


def _credits_used_today(db, user_id: int) -> int:
    """Chat messages this user has sent since IST midnight.

    This is the whole credit mechanism: remaining = cap - this. Reset is free —
    at midnight the window moves and the count is 0 again, so there is no
    credits table and no nightly restore job. Counts the user's own turns only,
    so one question and its answer together spend exactly one credit. A message
    whose stream failed was never persisted and so costs nothing, which is
    generous by a rewrite call; tighten it only if that is ever abused.
    """
    since = now_ist().replace(hour=0, minute=0, second=0, microsecond=0)
    return db.query(ChatMessage).filter(
        ChatMessage.user_id == user_id,
        ChatMessage.role == "user",
        ChatMessage.created_at >= since,
    ).count()


def _chat_dict(chat: ChatSession) -> dict:
    return {
        "id": chat.id,
        "title": chat.title,
        "status": chat.status,
        "error": chat.error,
        "filename": chat.filename,
        "doc_stats": chat.doc_stats,
        "created_at": chat.created_at.isoformat() if chat.created_at else None,
        "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
    }


def _get_retriever(db, chat: ChatSession) -> EnhancedDocumentStoreHybrid:
    """Cached store for a ready chat, rehydrating from Pinecone on a miss."""
    if chat.status != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Document is not ready (status: {chat.status}). Upload a PDF first.",
        )

    store = _retrievers.get(chat.id)
    if store is not None:
        _retrievers.move_to_end(chat.id)
        return store

    if not chat.bm25_params:
        raise HTTPException(
            status_code=500,
            detail="This chat is missing its search index. Please re-upload the document.",
        )

    while len(_retrievers) >= MAX_CACHED_RETRIEVERS:
        evicted, _ = _retrievers.popitem(last=False)
        print(f"♻️ Evicting cached retriever {evicted} (vectors are untouched)")

    store = EnhancedDocumentStoreHybrid.rehydrate(
        namespace=_ns(db, chat.user_id),
        chat_id=chat.id,
        bm25_params=chat.bm25_params,
        doc_stats=chat.doc_stats or {},
        embed_model=embed_model,
    )
    _retrievers[chat.id] = store
    return store


# ── Auth ──────────────────────────────────────────────────────────────────────

def _issue_refresh_token(db, user_id: int) -> str:
    """Create a refresh token, store only its hash, return the raw value (shown
    to the client once). Deleting the row later is how a session is revoked."""
    raw, token_hash = make_refresh_token()
    db.add(RefreshToken(
        user_id=user_id, token_hash=token_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(days=REFRESH_TTL_DAYS),
    ))
    db.commit()
    return raw


@app.post("/api/auth/signup")
@limiter.limit(RATE_SIGNUP)
def signup(request: Request, body: SignupRequest):
    db = SessionLocal()
    try:
        if db.query(Account).filter(Account.username == body.username).first():
            raise HTTPException(status_code=400, detail="Username already taken.")

        user = Account(username=body.username, hashed_password=hash_password(body.password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return {
            "token": create_token(user.id, user.username),
            "refresh_token": _issue_refresh_token(db, user.id),
            "user_id": user.id,
            "username": user.username,
        }
    finally:
        db.close()


@app.post("/api/auth/login")
@limiter.limit(RATE_LOGIN)
def login(request: Request, body: LoginRequest):
    db = SessionLocal()
    try:
        cutoff = now_ist() - timedelta(minutes=LOGIN_LOCKOUT_MINUTES)
        failures = db.query(LoginFailure).filter(
            LoginFailure.username == body.username,
            LoginFailure.created_at >= cutoff,
        ).count()
        if failures >= MAX_LOGIN_FAILURES:
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed attempts. Try again in {LOGIN_LOCKOUT_MINUTES} minutes.",
            )

        user = db.query(Account).filter(Account.username == body.username).first()
        if not user or not verify_password(body.password, user.hashed_password):
            db.add(LoginFailure(username=body.username))
            db.commit()
            raise HTTPException(status_code=401, detail="Invalid username or password.")

        db.query(LoginFailure).filter(LoginFailure.username == body.username).delete()
        if needs_rehash(user.hashed_password):
            user.hashed_password = hash_password(body.password)
        db.commit()

        return {
            "token": create_token(user.id, user.username),
            "refresh_token": _issue_refresh_token(db, user.id),
            "user_id": user.id,
            "username": user.username,
        }
    finally:
        db.close()


@app.post("/api/auth/refresh")
@limiter.limit(RATE_LOGIN)
def refresh_access_token(request: Request, body: RefreshRequest):
    """Exchange a valid, unexpired, unrevoked refresh token for a new access
    token. The refresh token itself is unchanged and keeps its own expiry."""
    db = SessionLocal()
    try:
        row = db.query(RefreshToken).filter(
            RefreshToken.token_hash == hash_refresh_token(body.refresh_token),
            RefreshToken.expires_at > datetime.now(timezone.utc),
        ).first()
        if row is None:
            raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
        user = db.query(Account).filter(Account.id == row.user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
        return {"token": create_token(user.id, user.username)}
    finally:
        db.close()


@app.post("/api/auth/logout")
def logout(body: RefreshRequest):
    """Revoke a refresh token by deleting it, so it can mint no more access
    tokens. The access token still lives out its short TTL — that is the cost
    of a stateless access token, and why it is short."""
    db = SessionLocal()
    try:
        db.query(RefreshToken).filter(
            RefreshToken.token_hash == hash_refresh_token(body.refresh_token)
        ).delete()
        db.commit()
        return {"success": True}
    finally:
        db.close()


# ── Account ───────────────────────────────────────────────────────────────────

@app.get("/api/account/credits")
def get_credits(current_user: dict = Depends(get_current_user)):
    """Daily chat credits: 1 credit = one question and its answer, resets at IST midnight."""
    db = SessionLocal()
    try:
        used = _credits_used_today(db, current_user["user_id"])
    finally:
        db.close()
    return {"cap": DAILY_MESSAGE_CAP, "used": used,
            "remaining": max(0, DAILY_MESSAGE_CAP - used)}


# ── Chats ─────────────────────────────────────────────────────────────────────

@app.get("/api/chats")
def list_chats(current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        chats = db.query(ChatSession).filter(
            ChatSession.user_id == current_user["user_id"]
        ).order_by(ChatSession.updated_at.desc()).all()
        return {"chats": [_chat_dict(c) for c in chats]}
    finally:
        db.close()


@app.post("/api/chats/new")
def new_chat(current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        uid = current_user["user_id"]
        # Reuse an existing empty chat so repeated clicks don't pile up blanks.
        existing = db.query(ChatSession).filter(
            ChatSession.user_id == uid, ChatSession.status == "awaiting_document"
        ).order_by(ChatSession.created_at.desc()).first()
        if existing is not None:
            return {"chat": _chat_dict(existing)}

        now = now_ist()
        chat = ChatSession(
            id=uuid.uuid4().hex, user_id=uid, title="New Chat",
            status="awaiting_document", created_at=now, updated_at=now,
        )
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return {"chat": _chat_dict(chat)}
    finally:
        db.close()


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        chat = _owned_chat(db, chat_id, current_user["user_id"])
        messages = db.query(ChatMessage).filter(
            ChatMessage.chat_id == chat_id
        ).order_by(ChatMessage.id).all()
        return {
            "chat": _chat_dict(chat),
            "messages": [
                {
                    "id": m.id, "role": m.role, "content": m.content,
                    "sources": m.sources,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in messages
            ],
        }
    finally:
        db.close()


@app.get("/api/chats/{chat_id}/status")
def chat_status(chat_id: str, current_user: dict = Depends(get_current_user)):
    """Cheap poll target while a document is ingesting."""
    db = SessionLocal()
    try:
        chat = _owned_chat(db, chat_id, current_user["user_id"])
        return {
            "status": chat.status,
            "stage": chat.stage,
            "error": chat.error,
            "doc_stats": chat.doc_stats,
            "filename": chat.filename,
            "title": chat.title,
        }
    finally:
        db.close()


@app.post("/api/chats/{chat_id}/document", status_code=202)
@limiter.limit(RATE_UPLOAD)
async def upload_document(
    request: Request,
    chat_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Attach a PDF to a chat and queue it for ingestion.

    Returns 202 immediately — ingestion runs for minutes (Textract per-page,
    one LLM call per page for classification, one embedding call per chunk),
    which no HTTP client should be asked to hold open. Poll /status.

    The work goes on the queue, not into this process: a BackgroundTask dies
    with the API, so a deploy mid-ingest used to lose the document outright.
    A queued job survives, and `worker.py` runs it.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    db = SessionLocal()
    try:
        chat = _owned_chat(db, chat_id, current_user["user_id"])
        if chat.status == "processing":
            raise HTTPException(status_code=409, detail="This chat is already processing a document.")

        # One document per chat. Replacing it means the old vectors must go,
        # or they keep competing for top_k against the new document.
        if chat.status == "ready":
            _retrievers.pop(chat_id, None)
            EnhancedDocumentStoreHybrid(
                namespace=_ns(db, chat.user_id), chat_id=chat_id
            ).retriever.delete_chat()

        # Read with a running size check so an oversized upload is rejected
        # mid-stream rather than after it has all arrived. The cap is small
        # (MAX_UPLOAD_MB), which is what makes holding it in memory reasonable —
        # it goes straight into the job row, so there is no temp file to leak
        # if this process dies between here and the commit.
        buf = bytearray()
        while chunk := await file.read(1024 * 1024):
            buf.extend(chunk)
            if len(buf) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File is larger than the {MAX_UPLOAD_MB} MB limit.",
                )

        if not buf:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        chat.status = "processing"
        chat.stage = None          # cleared; the worker sets each sub-step
        chat.error = None
        chat.filename = file.filename
        chat.doc_stats = None
        chat.bm25_params = None
        chat.updated_at = now_ist()

        # One transaction for both, so a chat can never be left 'processing'
        # with no job to advance it, nor a job exist for a chat that was never
        # marked. That pairing is why enqueue does not commit for itself.
        job_id = job_queue.enqueue(db, chat_id, chat.user_id, file.filename, bytes(buf))
        db.commit()

        print(f"📥 Queued job {job_id[:8]} for chat {chat_id} ({file.filename})")
        return {"chat_id": chat_id, "status": "processing", "filename": file.filename}
    finally:
        db.close()


def _sse(event_type: str, data) -> str:
    """Format one Server-Sent Event line."""
    return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"


def _next_or_stop(gen, sentinel):
    """next(gen) or sentinel on exhaustion — so the blocking call can run in a
    thread without letting StopIteration cross the thread boundary."""
    try:
        return next(gen)
    except StopIteration:
        return sentinel


@app.post("/api/chats/{chat_id}/message")
@limiter.limit(RATE_MESSAGE)
async def send_message(request: Request, chat_id: str, body: MessageRequest,
                       current_user: dict = Depends(get_current_user)):
    """Stream the answer as Server-Sent Events:
      meta  -> sources + the rewritten query, before any token
      token -> answer text as the model produces it
      done  -> stream finished and both turns are persisted
      error -> a message to show in place of the answer

    Retrieval, the rewrite and persistence are synchronous work and run in
    threads; only the token stream lives on the event loop.
    """
    uid = current_user["user_id"]

    def _prepare():
        """Own the chat, rehydrate, rewrite the follow-up, retrieve. Returns
        (search_query, retrieved, sources)."""
        db = SessionLocal()
        try:
            chat = _owned_chat(db, chat_id, uid)

            # Ownership first (a non-owner learns nothing about credits), then
            # the cap, then the expensive work. Checked here rather than in the
            # endpoint because this thread already holds a connection, and the
            # load test showed a message's DB round-trips are what starve the
            # pool — a third acquisition per message would make that worse.
            used = _credits_used_today(db, uid)
            if used >= DAILY_MESSAGE_CAP:
                raise HTTPException(
                    status_code=429,
                    detail=(f"Daily limit reached — {DAILY_MESSAGE_CAP} messages "
                            f"per day. Your credits reset at midnight IST."),
                )

            store = _get_retriever(db, chat)

            if body.summarize:
                # Whole-document summary: no rewrite, no top-k — every chunk in
                # reading order, and one "full document" source not N chunks.
                retrieved = store.all_chunks()
                src = []
                if retrieved:
                    pages = [p for c, _ in retrieved for p in (c.page_start, c.page_end)]
                    src = [{"filename": retrieved[0][0].filename,
                            "doc_type": "Full document",
                            "pages": f"{min(pages)}-{max(pages)}",
                            "relevance": "100%", "preview": ""}]
                return body.question, retrieved, sanitize(src)

            if body.alpha != store.alpha:
                store.set_alpha(body.alpha)

            # History comes from Postgres, not the request: the rewrite drives
            # retrieval, so a client could otherwise steer what gets searched.
            history = [
                {"role": m.role, "content": m.content}
                for m in db.query(ChatMessage)
                          .filter(ChatMessage.chat_id == chat_id)
                          .order_by(ChatMessage.id.desc())
                          .limit(MAX_HISTORY_MESSAGES).all()
            ][::-1]

            search_query = rewrite_standalone(body.question, history)
            filter_type = None if body.filter_type in (None, "All", "") else body.filter_type
            retrieved = store.retrieve_only(
                search_query, filter_type=filter_type, k=body.num_chunks)
            return search_query, retrieved, sanitize(build_sources(retrieved))
        finally:
            db.close()

    def _save(answer, sources):
        """Persist both turns. Returns False if the chat was deleted mid-stream."""
        db = SessionLocal()
        try:
            chat = db.query(ChatSession).filter(
                ChatSession.id == chat_id, ChatSession.user_id == uid).first()
            if chat is None:
                return False
            # Name the chat after the first question, so the rail shows what the
            # conversation is about rather than the filename. Only on the first
            # message, and only if the title is still the default or the
            # filename — never overwrite a name the user set themselves.
            is_first = db.query(ChatMessage).filter(
                ChatMessage.chat_id == chat_id).count() == 0
            if is_first and chat.title in ("New Chat", chat.filename):
                chat.title = _title_from_question(body.question)
            # Store what the user actually typed; the rewrite is a retrieval
            # artefact, not part of the conversation.
            db.add(ChatMessage(chat_id=chat_id, user_id=uid, role="user",
                               content=body.question))
            db.add(ChatMessage(chat_id=chat_id, user_id=uid, role="assistant",
                               content=answer, sources=sources))
            chat.updated_at = now_ist()
            db.commit()
            return True
        finally:
            db.close()

    async def event_stream():
        # One parent span per message; the rewrite and the answer's LLM calls
        # (run in threads, which inherit the OTel context) nest under it. A
        # counter point per message feeds Grafana rate/error alerting.
        ok = False
        with trace_message(body.question, uid, chat_id) as span:
            try:
                try:
                    search_query, retrieved, sources = await asyncio.to_thread(_prepare)
                except HTTPException as e:
                    yield _sse("error", e.detail)
                    return
                except Exception as e:
                    print(f"⚠️  message prepare failed: {type(e).__name__}: {e}")
                    yield _sse("error", "Something went wrong while preparing your answer.")
                    return

                yield _sse("meta", {"sources": sources, "question_asked": body.question,
                                    "question_searched": search_query})

                _STOP = object()
                gen = stream_answer(search_query, retrieved)
                parts = []
                try:
                    while True:
                        tok = await asyncio.to_thread(_next_or_stop, gen, _STOP)
                        if tok is _STOP:
                            break
                        parts.append(tok)
                        yield _sse("token", tok)
                except Exception as e:
                    print(f"⚠️  message stream failed: {type(e).__name__}: {e}")

                answer = "".join(parts).strip()
                if not answer:
                    # Retrieval worked but the model returned nothing — say so rather
                    # than persist a blank bubble that reads as "the document is silent".
                    answer = LLM_EMPTY_ANSWER
                    yield _sse("token", answer)
                set_output(span, answer)

                try:
                    saved = await asyncio.to_thread(_save, answer, sources)
                except Exception as e:
                    print(f"⚠️  message save failed: {type(e).__name__}: {e}")
                    yield _sse("error", "Your answer was generated but could not be saved.")
                    return
                if not saved:
                    yield _sse("error", "This chat no longer exists.")
                    return

                yield _sse("done", {"question_searched": search_query})
                ok = True
            finally:
                trace_flush()
                record_message("ok" if ok else "error")

    # X-Accel-Buffering disables proxy buffering: an nginx-class proxy will
    # otherwise hold the whole SSE body and deliver it in one lump at the end,
    # silently turning token streaming into a slow blocking request. Render's
    # proxy does not do this today; a different host might.
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.patch("/api/chats/{chat_id}")
def rename_chat(chat_id: str, body: RenameRequest,
                current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        chat = _owned_chat(db, chat_id, current_user["user_id"])
        chat.title = body.title
        chat.updated_at = now_ist()
        db.commit()
        db.refresh(chat)
        return {"chat": _chat_dict(chat)}
    finally:
        db.close()


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        chat = _owned_chat(db, chat_id, current_user["user_id"])

        # A queued or running job holds this chat's namespace and id prefix.
        # Deleting now would remove the rows and then let the worker finish
        # writing vectors nothing owns — the UI disables the button, but the
        # API is the boundary that has to enforce it.
        if chat.status == "processing":
            raise HTTPException(
                status_code=409,
                detail="This chat is still processing a document. Wait for it to "
                       "finish before deleting.",
            )

        # Vectors first: vectors whose chat row is gone have no owner left
        # who could ever find or delete them.
        _retrievers.pop(chat_id, None)
        EnhancedDocumentStoreHybrid(
            namespace=_ns(db, chat.user_id), chat_id=chat_id
        ).retriever.delete_chat()

        db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id).delete()
        # Finished job rows outlive the ingest they describe; without this they
        # would outlive the chat too, referencing a chat_id that no longer
        # resolves. In-flight jobs cannot be here — the guard above returned 409.
        db.query(IngestJob).filter(IngestJob.chat_id == chat_id).delete()
        db.delete(chat)
        db.commit()
        return {"success": True}
    finally:
        db.close()


@app.get("/")
def root():
    # So hitting the bare domain returns 200, not FastAPI's default 404 — the
    # real endpoints live under /api. Used as a cheap liveness ping too.
    return {"status": "ok", "service": "document-retrieval-system-api", "version": app.version}


@app.get("/api/health")
def health():
    return {"status": "ok", "cached_retrievers": len(_retrievers)}


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # forwarded_allow_ips="*" so per-IP rate limits use X-Forwarded-For behind a
    # proxy. Harmless locally (no proxy sends the header).
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True,
                forwarded_allow_ips="*")
