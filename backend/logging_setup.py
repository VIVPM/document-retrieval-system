"""JSON logging with a correlation id threaded through every line."""

import contextvars
import json
import logging
import os
import sys
import time

correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-")

LOG_FORMAT = os.getenv("LOG_FORMAT", "json").strip().lower()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()

_STD = frozenset((
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
))


class _SafeLogger(logging.Logger):
    """A logger whose `extra=` cannot raise."""

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        if extra:
            extra = {(f"{k}_" if k in _STD else k): v for k, v in extra.items()}
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info,
                                  func, extra, sinfo)


logging.setLoggerClass(_SafeLogger)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the correlation id and any extras."""

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": correlation_id.get(),
        }
        for k, v in record.__dict__.items():
            if k not in _STD and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable, for a local terminal. Same fields, less quoting."""

    def format(self, record: logging.LogRecord) -> str:
        rid = correlation_id.get()
        extras = " ".join(
            f"{k}={v}" for k, v in record.__dict__.items()
            if k not in _STD and not k.startswith("_")
        )
        head = f"{record.levelname:<5} [{rid[:8]}] {record.getMessage()}"
        if extras:
            head = f"{head}  {extras}"
        if record.exc_info:
            head = f"{head}\n{self.formatException(record.exc_info)}"
        return head


def configure() -> None:
    """Install the root handler. Idempotent -- the API and the worker both call it."""
    root = logging.getLogger()
    if any(getattr(h, "_drs_configured", False) for h in root.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if LOG_FORMAT == "json" else TextFormatter())
    handler._drs_configured = True          # type: ignore[attr-defined]

    root.handlers = [handler]
    root.setLevel(LOG_LEVEL)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_correlation_id(value: str) -> None:
    correlation_id.set(value)


if __name__ == "__main__":
    _rec = _SafeLogger("t").makeRecord(
        "t", logging.INFO, "f", 1, "m", (), None,
        extra={"filename": "a.pdf", "module": "m", "job_id": "ok"})
    assert _rec.filename == "f", "record's own field must win"
    assert _rec.filename_ == "a.pdf", "colliding value must survive, suffixed"
    assert _rec.module_ == "m"
    assert _rec.job_id == "ok", "non-colliding keys must pass through unchanged"
    print("logging_setup selftest OK")
