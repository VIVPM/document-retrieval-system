"""Offline checks for the current evaluation artifacts and their links."""
import csv
import json
import re
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "results" / "current"


def load(name):
    return json.loads((CURRENT / name).read_text(encoding="utf-8"))


class EvaluationArtifactsTests(unittest.TestCase):
    def test_current_question_coverage(self):
        quality = load("answers.json")["results"]
        safety = load("safety_answers.json")["results"]
        self.assertEqual(len(quality), 120)
        self.assertEqual(len(safety), 20)
        self.assertEqual(len({r["id"] for r in quality + safety}), 140)
        for row in quality + safety:
            self.assertEqual(row["gen_status"], "complete")
            self.assertEqual(len(row["retrieved_contexts"]), 6)
            self.assertEqual(len(row["retrieved_chunks"]), 6)
            self.assertTrue(all(c["score"] > 0 and not c["linked"] for c in row["retrieved_chunks"]))

    def test_summary_keeps_quality_and_safety_separate(self):
        summary = load("summary.json")
        self.assertEqual(summary["cases"], 140)
        with (CURRENT / "ragas.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 120)
        for metric, result in summary["answer_quality"]["ragas"].items():
            self.assertEqual(result["n"], 120)
            self.assertAlmostEqual(result["mean"], sum(float(r[metric]) for r in rows) / 120)
        for category, result in summary["safety"]["response_checks"].items():
            rows = [r for r in load("safety_scores.json")["results"] if r["category"] == category]
            applicable = [r for r in rows if r["safety_pass"] is not None]
            self.assertEqual(result["n"], len(applicable))
            self.assertEqual(result["passed"], sum(r["safety_pass"] for r in applicable))
            self.assertEqual(result["not_tested"], len(rows) - len(applicable))
        self.assertEqual(summary["latency"], load("latency.json")["summary"])
        self.assertEqual(len(load("latency.json")["results"]), 140)

    def test_accuracy_review(self):
        review = load("accuracy_review.json")
        counts = Counter(r["verdict"] for r in review["decisions"])
        self.assertEqual(sum(counts.values()), 120)
        self.assertEqual(counts["correct"], 109)
        self.assertAlmostEqual(review["summary"]["answer_accuracy"], counts["correct"] / 120)

    def test_safety_references_are_in_dataset(self):
        questions = json.loads((ROOT / "backend" / "eval" / "safety_questions.json").read_text(encoding="utf-8"))
        by_id = {q["id"]: q for q in questions["questions"]}
        for row in load("safety_answers.json")["results"]:
            self.assertEqual(row["expected_answer"], by_id[row["id"]]["expected_answer"])

    def test_current_links_and_filenames(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for path in re.findall(r"\]\((results/[^)]+)\)", readme):
            self.assertTrue(path.startswith("results/current/"), path)
            self.assertTrue((ROOT / path).is_file(), path)
        for path in load("summary.json")["files"].values():
            self.assertTrue((ROOT / path).is_file(), path)
        self.assertFalse(any(re.search(r"(?:^|_)v\d", p.name) for p in CURRENT.iterdir()))


if __name__ == "__main__":
    unittest.main()
