"""
JSON logging with a correlation id threaded through every line.

print() was the only telemetry. That is greppable while there is one process;
it stops being usable the moment there are two, because nothing ties a line to
the request or the job it came from -- and this app now runs an API and a
worker, so a single upload already spans two processes.

The id travels API -> job row -> worker, so `request_id` on an upload and on
the worker line that ingests it are the same string. That is the whole point:
one grep answers "what happened to this upload".

Kept deliberately small -- stdlib logging plus a Formatter. A logging library
would be a dependency for something a JSON dump already does, and the OTel
exporters in observability.py already handle the metrics/traces side.
"""

import contextvars
import json
import logging
import os
import sys
import time

# Set per request (API) or per job (worker); every log line inside that scope
# picks it up automatically. A ContextVar rather than a global because the API
# handles requests concurrently and a global would interleave them.
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-")

LOG_FORMAT = os.getenv("LOG_FORMAT", "json").strip().lower()   # json | text
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()

# Keys the stdlib puts on every record. Anything NOT in here was passed by the
# caller as `extra=` and belongs in the JSON output.
_STD = frozenset((
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
))


class _SafeLogger(logging.Logger):
    """A logger whose `extra=` cannot raise.

    stdlib logging raises KeyError if an `extra` key collides with a field
    LogRecord already owns, and several reserved names -- `filename`,
    `module`, `process`, `name`, `message` -- are exactly what you would call
    a log field. This is not theoretical: `extra={"filename": ...}` on the
    'ingest started' line killed _run_one() before any work ran, so every job
    was claimed and then silently abandoned until its 1800s lease expired.
    The log call, not the ingest, was the failure.

    Colliding keys are suffixed rather than dropped, so the value still
    reaches the output instead of disappearing to make room for the record's.
    """

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        if extra:
            extra = {(f"{k}_" if k in _STD else k): v for k, v in extra.items()}
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info,
                                  func, extra, sinfo)


# Installed at import, before configure() or get_logger() can run, so every
# logger this module hands out is the safe one.
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
        # default=str so a stray UUID, datetime or Decimal cannot make a log
        # line raise. A logging call must never be the thing that fails.
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

    # Replace rather than append: uvicorn installs its own handler, and leaving
    # it would print every line twice, once JSON and once not.
    root.handlers = [handler]
    root.setLevel(LOG_LEVEL)

    # uvicorn's loggers propagate to root once their own handlers are dropped,
    # so access lines come out in the same format as everything else.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_correlation_id(value: str) -> None:
    correlation_id.set(value)


if __name__ == "__main__":
    # python logging_setup.py -- the reserved-key guard, which is the one piece
    # of logic here that can take a request down if it regresses.
    _rec = _SafeLogger("t").makeRecord(
        "t", logging.INFO, "f", 1, "m", (), None,
        extra={"filename": "a.pdf", "module": "m", "job_id": "ok"})
    assert _rec.filename == "f", "record's own field must win"
    assert _rec.filename_ == "a.pdf", "colliding value must survive, suffixed"
    assert _rec.module_ == "m"
    assert _rec.job_id == "ok", "non-colliding keys must pass through unchanged"
    print("logging_setup selftest OK")
