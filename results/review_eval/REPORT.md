# Fictional loan-file automatic review evaluation

Status: baseline saved, one focused fix applied, same six frozen packets rerun. The 60 field labels were approved in the user-supplied review CSV; `gold_v1` preserves that reviewed copy and packet hashes. All packet content is fictional.

## Scope and method

- Six packets, 28 total pages per run: one clean, three with one intentional mismatch each, one without a pay slip, and one without a bank statement.
- Five labeled comparisons per packet: income, employer, checking balance, loan amount, and purchase price. Each comparison has two labeled field rows, for 30 comparisons and 60 rows. Of the 60 fields, 57 have a printed value and 3 are absent because the supporting document is missing.
- The runner calls the production Reducto adapter, TypeSafe Jev document classifier/boundary detector, and `build_review` with the production review LLM. It bypasses Pinecone indexing because file review consumes the classified logical documents before retrieval. Review latency starts **after** extraction and classification finish.
- Predictions are saved separately from labels. The baseline's original `review.py` is saved as `baseline/review_source.py`; the after run uses the current code. Each packet JSON includes field predictions, review checks, evidence pages, document types, model/config names, code/packet hashes, latency, usage, and estimated costs. `summary.json` contains all scored per-field and per-check predictions.
- Values use exact printed text. Page accuracy requires the packet's 1-based source page. A missing document must not cause a mismatch or review flag for an impossible comparison. `missing` is the app status corresponding to the gold set's `missing_document` outcome.

## Results

| Metric | Baseline | After | Denominator / meaning |
| --- | ---: | ---: | --- |
| Field extraction accuracy | 57/57 (100%) | 57/57 (100%) | Present labeled fields only |
| Absent fields not invented | 3/3 (100%) | 3/3 (100%) | Fields in absent supporting documents |
| Page-citation accuracy | 57/57 (100%) | 57/57 (100%) | Present labeled fields with a prediction on the correct packet page |
| Discrepancy precision | 3/3 (100%) | 3/3 (100%) | True mismatches / all predicted mismatch checks (0 false positives) |
| Discrepancy recall | 3/3 (100%) | 3/3 (100%) | Found mismatches / 3 expected mismatches (0 false negatives) |
| False flags on missing-document comparisons | 0/3 | 0/3 | `mismatch` or `review` on an uncheckable comparison |
| General missing-document indicator | 3/3 | 3/3 | Comparison-level opportunities with `not_in_packet_*` notice |
| Explicit missing comparison checks | 0/3 | 3/3 | Affected income, employer, or balance row has `missing` status |
| Overall comparison-status accuracy | 27/30 (90%) | 30/30 (100%) | Five labeled comparisons in each packet |

Mean review time after ingestion was **1.490 s** at baseline (range 1.310-1.636 s) and **1.807 s** after (range 1.354-3.067 s), across six packets per phase. This small sequential sample does not establish a latency regression or improvement.

## Cost estimate

Each phase processed 28 pages. Reducto r-1 mini list price is **$0.001/page**, giving **$0.028** estimated parsing cost per phase. Review Gemini usage, priced by the app's `PRICE_PER_MTOK`, adds **$0.0027692** per phase. The **known-price subtotal is $0.0307692 per phase** (about $0.0615384 for both runs). TypeSafe Jev calls occurred during classification, but their token usage was not captured in these runs; their cost is **excluded**, so the subtotal is not a complete bill. Network, discounts, credits, retries, and taxes are also excluded.

Pricing sources: [Reducto r-1 mini](https://reducto.ai/blog/reducto-r-1-mini-parse-model), [TypeSafe Jev](https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost). TypeSafe lists $0.042 per million input tokens, but no token count was recorded for this evaluation. The runner now captures Jev usage for future runs; it does not invent historical token counts.

## Largest observed failures and fix

1. **Missing-document comparison rows were omitted at baseline (3/3).** The review did give a general `not_in_packet_*` notice, so it did not falsely assert a discrepancy, but it did not show which income, employer, or balance comparison could not be verified. `run_rules` now emits a `missing` check with the application evidence page for those three cases. The after run confirms all three.
2. **One application value was rejected by the grounding guard in every packet (6/6).** The application printed `Undisclosed Funds From Another Party?: NO`, but the declaration guard searches a narrower phrase. This field was outside the approved 60-field gold set, so it is a documented unscored failure and was not changed in this one-fix comparison.
3. The labeled packets are simple digital-text fixtures. They do not test scanned pages, handwriting, multi-page statements, duplicate amounts, ambiguous names, multiple pay slips, or adverse title records. Perfect scores on 3 positive discrepancies have wide uncertainty and should not be generalized to real loan files.

## Reproduction and files

From the repository root, with the existing credentials in `backend/.env`:

```powershell
backend/.venv/Scripts/python.exe backend/core/review.py
backend/.venv/Scripts/python.exe backend/eval/run_review_eval.py baseline
backend/.venv/Scripts/python.exe backend/eval/run_review_eval.py after
```

The runner reuses matching saved packet JSON. To generate a fresh run, copy the frozen packets and reviewed labels to a new evaluation directory and use a new output phase; do not overwrite these baseline and after files. The PDFs and approved labels are in `output/pdf/review_eval_fictional/` and `results/review_eval/gold_v1/`. Run artifacts are in `results/review_eval/baseline/` and `results/review_eval/after/`.

No manual review time was measured, so this evaluation makes **no manual-time reduction claim**.
