"""Run the frozen fictional packets through extraction and automatic review.

Usage: backend/.venv/Scripts/python.exe backend/eval/run_review_eval.py baseline
       backend/.venv/Scripts/python.exe backend/eval/run_review_eval.py after

Writes one JSON file per packet immediately, then a scored summary. No polling,
Pinecone indexing, retrieval, or account creation is involved: automatic review
consumes the logical documents produced by extract_and_analyze_pdf directly.
"""

import csv
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
load_dotenv(BACKEND / ".env")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core import review  # noqa: E402
from core.document_classifier_jev import MODEL as JEV_MODEL  # noqa: E402
from core.pdf_processor import extract_and_analyze_pdf  # noqa: E402
from llm.llm_router import (GEMINI_FAST_MODEL, LLM_MODEL,  # noqa: E402
                            estimate_cost_usd, llm)

GOLD = ROOT / "results/review_eval/gold_v1/reviewed_review_labels.csv"
PACKETS = ROOT / "output/pdf/review_eval_fictional"
OUT = ROOT / "results/review_eval"
REDUCTO_MINI_USD_PER_PAGE = 0.001  # https://reducto.ai/blog/reducto-r-1-mini-parse-model


class UsageCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.calls = []

    def emit(self, record):
        if record.name == "drs.classifier" and record.msg == "jev call":
            self.calls.append({
                "model": getattr(record, "jev_model", None),
                "decision_type": getattr(record, "decision_type", None),
                "input_tokens": getattr(record, "input_tokens", None),
                "estimated_cost_usd": getattr(record, "cost_usd", None),
            })


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_module(phase):
    if phase == "after":
        return review
    source = OUT / "baseline/review_source.py"
    spec = importlib.util.spec_from_file_location("review_baseline_snapshot", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def meta(phase, rules):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    return {
        "git_commit": commit,
        "review_code_sha256": sha(OUT / "baseline/review_source.py") if phase == "baseline"
                              else sha(BACKEND / "core/review.py"),
        "classifier_code_sha256": sha(BACKEND / "core/document_classifier_jev.py"),
        "extractor_code_sha256": sha(BACKEND / "core/pdf_processor.py"),
        "gold_sha256": sha(GOLD),
        "extract_method": os.getenv("EXTRACT_METHOD", "reducto"),
        "llm_provider": LLM_MODEL,
        "review_model": GEMINI_FAST_MODEL,
        "classifier_model": JEV_MODEL,
        "income_tolerance": rules.INCOME_TOLERANCE,
        "balance_tolerance": rules.BALANCE_TOLERANCE,
        "large_deposit_share": rules.LARGE_DEPOSIT_SHARE,
    }


def run_packet(case_id, labels, run_dir, config, rules):
    packet = PACKETS / f"{case_id}.pdf"
    expected_hashes = {r["packet_sha256"] for r in labels}
    if expected_hashes != {sha(packet)}:
        raise RuntimeError(f"Frozen packet hash mismatch: {packet}")

    collector = UsageCollector()
    logger = logging.getLogger("drs.classifier")
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(collector)
    try:
        start = time.perf_counter()
        pages, logical_docs = extract_and_analyze_pdf(str(packet), filename=packet.name)
        ingestion_seconds = time.perf_counter() - start
    finally:
        logger.removeHandler(collector)
        logger.setLevel(old_level)

    llm_calls = []

    def complete(prompt):
        reply = llm.complete(prompt, temperature=0, fast=True,
                             thinking_budget=0, max_tokens=4096)
        usage = reply.usage or {}
        llm_calls.append({"usage": usage, "estimated_cost_usd": estimate_cost_usd(usage)})
        return reply.text or ""

    start = time.perf_counter()
    result = rules.build_review(logical_docs, complete=complete)
    review_seconds = time.perf_counter() - start
    if result["status"] != "done":
        raise RuntimeError(f"Review did not complete for {case_id}")
    payload = {
        "case_id": case_id,
        "packet_file": packet.name,
        "packet_sha256": sha(packet),
        "config": config,
        "ingested_page_count": len(pages),
        "logical_documents": [{"doc_type": d.doc_type,
                               "pages": [d.page_start + 1, d.page_end + 1]}
                              for d in logical_docs],
        "ingestion_seconds": round(ingestion_seconds, 3),
        "review_seconds_after_ingestion": round(review_seconds, 3),
        "classifier_calls": collector.calls,
        "review_llm_calls": llm_calls,
        "review": result,
    }
    annotate_cost(payload)
    target = run_dir / f"{case_id}.json"
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved {target.name}: {len(result['documents'])} documents, "
          f"{len(result['checks'])} checks, review {review_seconds:.2f}s", flush=True)
    return payload


def annotate_cost(payload):
    """Add list-price estimates; preserve unknown costs as unknown."""
    llm_cost = sum(c.get("estimated_cost_usd") or 0 for c in payload["review_llm_calls"])
    classifier_cost = sum(c.get("estimated_cost_usd") or 0 for c in payload["classifier_calls"])
    reducto_cost = (payload["ingested_page_count"] * REDUCTO_MINI_USD_PER_PAGE
                    if payload["config"]["extract_method"] == "reducto" else None)
    payload["estimated_api_cost_breakdown_usd"] = {
        "review_llm": round(llm_cost, 8),
        "classifier_jev": round(classifier_cost, 8) if payload["classifier_calls"] else None,
        "reducto_list_price": round(reducto_cost, 8) if reducto_cost is not None else None,
    }
    payload["known_estimated_api_cost_usd"] = round(
        llm_cost + classifier_cost + (reducto_cost or 0), 8)
    payload["unpriced_api_components"] = ([] if payload["classifier_calls"]
                                          else ["Jev classifier usage was not captured"])
    return payload


def score(labels, runs):
    field_rows = []
    check_rows = []
    by_case = defaultdict(list)
    for row in labels:
        by_case[row["case_id"]].append(row)
    for case_id, gold_rows in by_case.items():
        actual = runs[case_id]["review"]
        docs = {d["doc_type"]: d for d in actual["documents"]}
        checks = {c["id"]: c for c in actual["checks"]}
        for row in gold_rows:
            doc = docs.get(row["expected_document_type"])
            prediction = (doc or {}).get("fields", {}).get(row["field_key"])
            expected = row["expected_field_value"]
            raw = (prediction or {}).get("raw")
            value_ok = (raw == expected) if expected else prediction is None
            page = (prediction or {}).get("page")
            page_ok = page == int(row["source_page"]) if row["source_page"] else page is None
            field_rows.append({
                "case_id": case_id, "check_key": row["check_key"],
                "field_key": row["field_key"], "expected_document_type": row["expected_document_type"],
                "expected_value": expected or None, "predicted_value": raw,
                "expected_page": int(row["source_page"]) if row["source_page"] else None,
                "predicted_page": page, "predicted_source": (prediction or {}).get("source"),
                "value_correct": value_ok, "page_correct": page_ok,
                "document_detected": doc is not None,
            })
        for check_key in dict.fromkeys(r["check_key"] for r in gold_rows):
            gold = next(r for r in gold_rows if r["check_key"] == check_key)
            predicted = checks.get(check_key)
            missing_type = None
            if gold["expected_review_status"] == "missing_document":
                missing_type = "Pay Slip" if case_id == "05_missing_pay_slip" else "Bank Statement"
            missing_indicator = checks.get(f"not_in_packet_{missing_type.lower().replace(' ', '_')}") if missing_type else None
            check_rows.append({
                "case_id": case_id, "check_key": check_key,
                "expected_status": gold["expected_review_status"],
                "predicted_status": (predicted or {}).get("status"),
                "flag_decision": (predicted or {}).get("status") in ("mismatch", "review"),
                "missing_document_reported": missing_indicator is not None,
                "evidence": (predicted or {}).get("evidence", []),
            })

    present = [r for r in field_rows if r["expected_value"] is not None]
    missing_fields = [r for r in field_rows if r["expected_value"] is None]
    tp = sum(r["expected_status"] == "mismatch" and r["predicted_status"] == "mismatch" for r in check_rows)
    fp = sum(r["expected_status"] != "mismatch" and r["predicted_status"] == "mismatch" for r in check_rows)
    fn = sum(r["expected_status"] == "mismatch" and r["predicted_status"] != "mismatch" for r in check_rows)
    missing_checks = [r for r in check_rows if r["expected_status"] == "missing_document"]
    def status_correct(row):
        expected = "missing" if row["expected_status"] == "missing_document" else row["expected_status"]
        return row["predicted_status"] == expected
    summary = {
        "field_extraction_accuracy": {"correct": sum(r["value_correct"] for r in present),
                                      "denominator": len(present)},
        "absent_field_not_invented": {"correct": sum(r["value_correct"] for r in missing_fields),
                                      "denominator": len(missing_fields)},
        "page_citation_accuracy": {"correct": sum(r["page_correct"] and r["predicted_value"] is not None for r in present),
                                   "denominator": len(present)},
        "discrepancy_precision": {"true_positives": tp, "false_positives": fp,
                                  "denominator": tp + fp},
        "discrepancy_recall": {"true_positives": tp, "false_negatives": fn,
                               "denominator": tp + fn},
        "false_flags_on_missing_document_checks": {
            "false_flags": sum(r["flag_decision"] for r in missing_checks),
            "denominator": len(missing_checks)},
        "missing_document_indicators": {
            "reported": sum(r["missing_document_reported"] for r in missing_checks),
            "denominator": len(missing_checks)},
        "comparison_status_accuracy": {"correct": sum(status_correct(r) for r in check_rows),
                                       "denominator": len(check_rows)},
        "missing_comparison_check_coverage": {
            "explicit_missing_checks": sum(r["predicted_status"] == "missing" for r in missing_checks),
            "denominator": len(missing_checks)},
        "review_latency_seconds": {"per_case": {k: v["review_seconds_after_ingestion"] for k, v in runs.items()}},
        "known_estimated_api_cost_usd": sum(v["known_estimated_api_cost_usd"] for v in runs.values()),
        "unpriced_api_components": sorted({x for v in runs.values() for x in v["unpriced_api_components"]}),
        "field_predictions": field_rows,
        "check_predictions": check_rows,
    }
    return summary


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("baseline", "after"):
        raise SystemExit("Usage: run_review_eval.py baseline|after")
    phase = sys.argv[1]
    run_dir = OUT / phase
    run_dir.mkdir(parents=True, exist_ok=True)
    with GOLD.open(encoding="utf-8", newline="") as fh:
        labels = list(csv.DictReader(fh))
    if len(labels) != 60 or any(r["label_status"] != "reviewer_approved" for r in labels):
        raise RuntimeError("Frozen reviewed labels missing or incomplete")
    rules = review_module(phase)
    config = meta(phase, rules)
    runs = {}
    for case_id in dict.fromkeys(r["case_id"] for r in labels):
        path = run_dir / f"{case_id}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing["config"] != config:
                raise RuntimeError(f"Existing {phase} result has different config: {path}")
            runs[case_id] = annotate_cost(existing)
            path.write_text(json.dumps(runs[case_id], indent=2) + "\n", encoding="utf-8")
        else:
            runs[case_id] = run_packet(case_id, [r for r in labels if r["case_id"] == case_id], run_dir, config, rules)
    scored = score(labels, runs)
    (run_dir / "summary.json").write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
