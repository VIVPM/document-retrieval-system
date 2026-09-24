"""main.py — FastAPI backend for the Document Retrieval System."""

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

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import (Depends, FastAPI, File, HTTPException,
                     Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter
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
from db.models import (Account, ChatMessage, ChatSession, IngestJob, ReviewDecision, ensure_columns,
                       LoginFailure, RefreshToken, now_ist)
import job_queue
from llm.llm_router import embed_model, estimate_cost_usd, llm as _llm
from observability import (flush as trace_flush, init_http_tracing,
                           record_cost, record_llm_metrics,
                           record_stream_quality,
                           init_metrics, init_observability, record_message,
                           set_output, trace_message)

import contextlib
import time

import logging_setup

logging_setup.configure()
log = logging_setup.get_logger("drs.api")

Base.metadata.create_all(bind=engine)
ensure_columns(engine)


def _recover_orphaned_ingests() -> None:
    """Reconcile chats stuck on 'processing' with the queue, at startup."""
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
            log.warning("failed orphaned chats", extra={"count": orphaned})
        left = len(stranded) - orphaned
        if left:
            log.info("chats left to the worker", extra={"count": left})
    except Exception:
        db.rollback()
        log.exception("could not reconcile stranded ingests")
    finally:
        db.close()


_recover_orphaned_ingests()

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "3"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_UPLOAD_FILES = 20


def _merge_pdfs(parts: list[bytes]) -> bytes:
    """Concatenate PDFs, in order, into one."""
    import fitz

    out = fitz.open()
    try:
        for i, data in enumerate(parts, 1):
            try:
                with fitz.open(stream=data, filetype="pdf") as doc:
                    out.insert_pdf(doc)
            except Exception as e:
                raise ValueError(f"File {i} is not a readable PDF.") from e
        return out.tobytes(garbage=3, deflate=True)
    finally:
        out.close()

def _ns(db, user_id: int) -> str:
    """Pinecone namespace for an account: the account's username."""
    acc = db.query(Account).filter(Account.id == user_id).first()
    return acc.username if acc else f"user_{user_id}"


def _title_from_question(q: str) -> str:
    """A chat title from the user's first question, trimmed to fit the rail."""
    q = " ".join((q or "").split())
    return (q[:48] + ("…" if len(q) > 48 else "")) or "New Chat"


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
    return str(obj)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup/shutdown. The shutdown half is what SIGTERM needs."""
    log.info("api up", extra={"version": _app.version, "llm": _llm.label})
    yield
    log.info("api shutting down; flushing telemetry")
    try:
        trace_flush()
    except Exception:
        log.exception("telemetry flush failed during shutdown")


app = FastAPI(
    title="Document Retrieval System API",
    description="Hybrid RAG pipeline for intelligent multi-document Q&A",
    version="2.0.0",
    lifespan=lifespan,
)

_DEFAULT_ORIGINS = "https://document-retrieval-system-frontend.onrender.com,http://localhost:5173"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_observability()
init_http_tracing(app)
init_metrics()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
def _rate_limited(request: Request, exc: RateLimitExceeded):
    """429 with Retry-After, so a client knows WHEN to retry rather than guessing."""
    limit = getattr(exc, "limit", None)
    window = getattr(getattr(limit, "limit", None), "GRANULARITY", None)
    seconds = getattr(window, "seconds", None) or 60
    retry_after = str(int(seconds))

    log.warning("rate limited", extra={"path": request.url.path,
                                       "client": get_remote_address(request),
                                       "limit": str(getattr(exc, "detail", "")),
                                       "retry_after_s": retry_after})
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}. "
                           f"Retry in about {retry_after}s."},
        headers={"Retry-After": retry_after},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limited)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    """Give every request an id and echo it back."""
    rid = (request.headers.get("X-Request-ID") or "").strip()[:64] or uuid.uuid4().hex[:16]
    logging_setup.set_correlation_id(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response

RATE_SIGNUP = os.getenv("RATE_SIGNUP", "5/hour")
RATE_LOGIN = os.getenv("RATE_LOGIN", "10/minute")
RATE_UPLOAD = os.getenv("RATE_UPLOAD", "10/hour")
RATE_MESSAGE = os.getenv("RATE_MESSAGE", "30/minute")

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
        return _normalise_email(v)


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_key: str
    decision: str
    note: str = ""

    @field_validator("decision")
    @classmethod
    def _decision(cls, v):
        if v not in ("accepted", "confirmed"):
            raise ValueError("decision must be 'accepted' or 'confirmed'")
        return v

    @field_validator("note")
    @classmethod
    def _note(cls, v):
        return v.strip()[:1000]


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
    model_config = ConfigDict(extra="forbid")

    question: str
    filter_type: Optional[str] = None
    num_chunks: int = Field(default=6, ge=1, le=20)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
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


def _owned_chat(db, chat_id: str, user_id: int) -> ChatSession:
    """Load a chat the caller owns, or 404."""
    chat = db.query(ChatSession).filter(
        ChatSession.id == chat_id, ChatSession.user_id == user_id
    ).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return chat


def _credits_used_today(db, user_id: int) -> int:
    """Chat messages this user has sent since IST midnight."""
    since = now_ist().replace(hour=0, minute=0, second=0, microsecond=0)
    return db.query(ChatMessage).filter(
        ChatMessage.user_id == user_id,
        ChatMessage.role == "user",
        ChatMessage.created_at >= since,
    ).count()


def _flag_keys(review) -> list[str]:
    """Keys of the checks an officer must decide: mismatches and reviews."""
    return [f"{c.get('id')}:{i}" for i, c in enumerate(review.get("checks") or [])
            if c.get("status") in ("mismatch", "review")]


def _decisions(db, chat_ids) -> dict:
    """chat_id -> all decision rows, oldest first."""
    out: dict = {}
    if chat_ids:
        rows = db.query(ReviewDecision).filter(ReviewDecision.chat_id.in_(chat_ids)) \
                 .order_by(ReviewDecision.id).all()
        for r in rows:
            out.setdefault(r.chat_id, []).append(r)
    return out


def _latest(review, rows) -> dict:
    """check_key -> the newest decision for THIS review run."""
    run = review.get("run_id") if isinstance(review, dict) else None
    return {r.check_key: r for r in rows or [] if run and r.run_id == run}


def _decision_dict(r: ReviewDecision) -> dict:
    return {"check_key": r.check_key, "check_label": r.check_label,
            "decision": r.decision, "note": r.note, "by": r.username,
            "at": r.created_at.isoformat() if r.created_at else None,
            "run_id": r.run_id}


def _review_summary(review, rows=None) -> dict | None:
    """What the sidebar needs from the file review: status, counts, borrower,
    and how many flags are still open. The full review (fields, evidence) is
    only sent when one chat is opened."""
    if not isinstance(review, dict):
        return None
    flags = _flag_keys(review)
    decided = _latest(review, rows)
    open_ = sum(1 for k in flags if k not in decided)
    return {"status": review.get("status"), "summary": review.get("summary"),
            "borrower": review.get("borrower"),
            "open": open_, "resolved": len(flags) - open_}


def _chat_dict(chat: ChatSession, rows=None) -> dict:
    return {
        "id": chat.id,
        "title": chat.title,
        "status": chat.status,
        "error": chat.error,
        "filename": chat.filename,
        "doc_stats": chat.doc_stats,
        "review_summary": _review_summary(chat.review, rows),
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
        log.info("evicted cached retriever", extra={"chat_id": evicted})

    store = EnhancedDocumentStoreHybrid.rehydrate(
        namespace=_ns(db, chat.user_id),
        chat_id=chat.id,
        bm25_params=chat.bm25_params,
        doc_stats=chat.doc_stats or {},
        embed_model=embed_model,
    )
    _retrievers[chat.id] = store
    return store


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


@app.get("/api/chats")
def list_chats(current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        chats = db.query(ChatSession).filter(
            ChatSession.user_id == current_user["user_id"]
        ).order_by(ChatSession.updated_at.desc()).all()
        dec = _decisions(db, [c.id for c in chats])
        return {"chats": [_chat_dict(c, dec.get(c.id)) for c in chats]}
    finally:
        db.close()


@app.post("/api/chats/new")
def new_chat(current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        uid = current_user["user_id"]
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
        rows = _decisions(db, [chat_id]).get(chat_id, [])
        return {
            "chat": {**_chat_dict(chat, rows), "review": chat.review,
                     "decisions": {k: _decision_dict(r)
                                   for k, r in _latest(chat.review, rows).items()},
                     "decision_history": [_decision_dict(r) for r in rows]},
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


@app.post("/api/chats/{chat_id}/review/decisions", status_code=201)
def decide_flag(chat_id: str, body: DecisionRequest,
                current_user: dict = Depends(get_current_user)):
    """Record an officer's call on one flag. Append-only: a changed mind is a
    new row, so the history keeps both."""
    db = SessionLocal()
    try:
        chat = _owned_chat(db, chat_id, current_user["user_id"])
        review = chat.review if isinstance(chat.review, dict) else {}
        if review.get("status") != "done" or not review.get("run_id"):
            raise HTTPException(status_code=409, detail="This file has no finished review.")
        if body.check_key not in _flag_keys(review):
            raise HTTPException(status_code=422, detail="Unknown flag for this review.")
        if body.decision == "accepted" and not body.note:
            raise HTTPException(status_code=422, detail="A note is required to accept a flag.")
        idx = int(body.check_key.rsplit(":", 1)[1])
        row = ReviewDecision(chat_id=chat_id, run_id=review["run_id"],
                             check_key=body.check_key,
                             check_label=review["checks"][idx].get("label"),
                             decision=body.decision, note=body.note or None,
                             user_id=current_user["user_id"],
                             username=current_user["username"])
        db.add(row)
        db.commit()
        db.refresh(row)
        return {"decision": _decision_dict(row)}
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
    file: list[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """    Attach a PDF to a chat and queue it for ingestion."""
    files = file
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400,
                            detail=f"At most {MAX_UPLOAD_FILES} files per upload.")
    if any(not f.filename or not f.filename.lower().endswith(".pdf") for f in files):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    names = [f.filename for f in files]
    filename = names[0] if len(names) == 1 else f"{names[0]} + {len(names) - 1} more"

    db = SessionLocal()
    try:
        chat = _owned_chat(db, chat_id, current_user["user_id"])
        if chat.status == "processing":
            raise HTTPException(status_code=409, detail="This chat is already processing a document.")

        if chat.status == "ready":
            _retrievers.pop(chat_id, None)
            EnhancedDocumentStoreHybrid(
                namespace=_ns(db, chat.user_id), chat_id=chat_id
            ).retriever.delete_chat()

        parts, total = [], 0
        for f in files:
            buf = bytearray()
            while chunk := await f.read(1024 * 1024):
                buf.extend(chunk)
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload is larger than the {MAX_UPLOAD_MB} MB limit.",
                    )
            if not buf:
                raise HTTPException(status_code=400, detail=f"{f.filename} is empty.")
            parts.append(bytes(buf))

        if len(parts) == 1:
            buf = parts[0]
        else:
            try:
                buf = await asyncio.to_thread(_merge_pdfs, parts)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            if len(buf) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Merged document is larger than the {MAX_UPLOAD_MB} MB limit.",
                )

        chat.status = "processing"
        chat.stage = None
        chat.error = None
        chat.filename = filename
        chat.doc_stats = None
        chat.bm25_params = None
        chat.review = None
        chat.updated_at = now_ist()

        job_id, created = job_queue.enqueue(
            db, chat_id, chat.user_id, filename, bytes(buf),
            idempotency_key=(request.headers.get("Idempotency-Key") or "").strip() or None,
            request_id=logging_setup.correlation_id.get(),
        )
        db.commit()

        log.info("upload queued" if created else "upload deduplicated",
                 extra={"job_id": job_id, "chat_id": chat_id, "user_id": chat.user_id,
                        "doc_filename": filename, "files": len(parts),
                        "bytes": len(buf), "duplicate": not created})
        return {"chat_id": chat_id, "status": "processing",
                "filename": filename, "duplicate": not created}
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
      error -> a message to show in place of the answer"""
    uid = current_user["user_id"]

    def _prepare():
        """Own the chat, rehydrate, rewrite the follow-up, retrieve. Returns
        (search_query, retrieved, sources)."""
        db = SessionLocal()
        try:
            chat = _owned_chat(db, chat_id, uid)

            used = _credits_used_today(db, uid)
            if used >= DAILY_MESSAGE_CAP:
                raise HTTPException(
                    status_code=429,
                    detail=(f"Daily limit reached — {DAILY_MESSAGE_CAP} messages "
                            f"per day. Your credits reset at midnight IST."),
                )

            store = _get_retriever(db, chat)

            if body.summarize:
                retrieved = store.all_chunks()
                src = []
                if retrieved:
                    pages = [p for c, _ in retrieved for p in (c.page_start, c.page_end)]
                    src = [{"filename": retrieved[0][0].filename,
                            "doc_type": "Full document",
                            "pages": f"{min(pages) + 1}-{max(pages) + 1}",
                            "relevance": "100%", "preview": ""}]
                return body.question, retrieved, sanitize(src)

            if body.alpha != store.alpha:
                store.set_alpha(body.alpha)

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
            is_first = db.query(ChatMessage).filter(
                ChatMessage.chat_id == chat_id).count() == 0
            if is_first and chat.title in ("New Chat", chat.filename):
                chat.title = _title_from_question(body.question)
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
        ok = False
        with trace_message(body.question, uid, chat_id) as span:
            try:
                try:
                    search_query, retrieved, sources = await asyncio.to_thread(_prepare)
                except HTTPException as e:
                    yield _sse("error", e.detail)
                    return
                except Exception:
                    log.exception("message prepare failed", extra={"chat_id": chat_id})
                    yield _sse("error", "Something went wrong while preparing your answer.")
                    return

                yield _sse("meta", {"sources": sources, "question_asked": body.question,
                                    "question_searched": search_query})

                _STOP = object()
                gen = stream_answer(search_query, retrieved, summarize=body.summarize)
                parts = []
                stream_started = time.monotonic()
                ttft = None
                try:
                    while True:
                        tok = await asyncio.to_thread(_next_or_stop, gen, _STOP)
                        if tok is _STOP:
                            break
                        if ttft is None:
                            ttft = time.monotonic() - stream_started
                        parts.append(tok)
                        yield _sse("token", tok)
                except Exception:
                    log.exception("message stream failed", extra={"chat_id": chat_id})

                stream_usage = getattr(_llm, "last_stream_usage", None) or {}
                record_stream_quality(span, ttft, len(parts),
                                      time.monotonic() - stream_started,
                                      output_tokens=stream_usage.get("output_tokens"))
                _cost = estimate_cost_usd(stream_usage)
                record_cost(span, stream_usage, _cost)
                record_llm_metrics(stream_usage, _cost, ttft)

                answer = "".join(parts).strip()
                if not answer:
                    answer = LLM_EMPTY_ANSWER
                    yield _sse("token", answer)
                set_output(span, answer)

                try:
                    saved = await asyncio.to_thread(_save, answer, sources)
                except Exception:
                    log.exception("message save failed", extra={"chat_id": chat_id})
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

        if chat.status == "processing":
            raise HTTPException(
                status_code=409,
                detail="This chat is still processing a document. Wait for it to "
                       "finish before deleting.",
            )

        _retrievers.pop(chat_id, None)
        EnhancedDocumentStoreHybrid(
            namespace=_ns(db, chat.user_id), chat_id=chat_id
        ).retriever.delete_chat()

        db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id).delete()
        db.query(IngestJob).filter(IngestJob.chat_id == chat_id).delete()
        db.delete(chat)
        db.commit()
        return {"success": True}
    finally:
        db.close()


@app.get("/")
def root():
    return {"status": "ok", "service": "document-retrieval-system-api", "version": app.version}


@app.get("/api/health")
def health():
    return {"status": "ok", "cached_retrievers": len(_retrievers)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True,
                forwarded_allow_ips="*")
