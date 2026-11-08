"""The answer-quality set is well-formed, every key point is grounded in the real provision
text (when the corpus is available), and the grading logic behaves."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
QUALITY = ROOT / "eval" / "answer_quality.jsonl"
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"

spec = importlib.util.spec_from_file_location("run_quality", ROOT / "eval" / "run_quality.py")
rq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq)


def _items():
    return [json.loads(line) for line in QUALITY.read_text().splitlines() if line.strip()]


def test_the_set_is_well_formed():
    items = _items()
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)) and len(items) >= 25
    for i in items:
        assert i["question"].strip().endswith("?")
        if i.get("expect_refusal"):
            assert not i["points"] and not i["gold"]
        else:
            assert i["gold"] and len(i["points"]) >= 2
            assert all(p["text"] and p["ground"] for p in i["points"])
    assert sum(1 for i in items if i.get("expect_refusal")) >= 3   # out-of-corpus traps are present


@pytest.mark.skipif(not CHUNKS.exists(), reason="needs the ingested corpus (python -m lexeuropa.cli ingest)")
def test_every_key_point_is_grounded_in_the_provision_text():
    chunks = [json.loads(line) for line in CHUNKS.read_text().splitlines()]
    for item in _items():
        if not item["gold"]:
            continue
        blob = " ".join(
            c["text"] for c in chunks
            for g in item["gold"]
            if c["celex"] == g["celex"] and (g["article"] is None or c["article"] == g["article"])
        ).lower()
        assert blob, f"no source text for {item['id']}"
        for point in item["points"]:
            for phrase in point["ground"]:
                assert phrase.lower() in blob, f"{item['id']}: '{phrase}' not in the gold provision"


def test_grade_parses_the_judges_checklist_verdict(monkeypatch):
    monkeypatch.setattr(rq, "chat", lambda *a, **k: ('```json\n{"points": [{"n": 1, "covered": true}, {"n": 2, "covered": false}], "contradiction": true, "contradiction_note": "x"}\n```', "stop"))
    g = rq.grade("Q?", ["A statement."], [{"text": "p1"}, {"text": "p2"}, {"text": "p3"}])
    assert g["covered"] == [True, False, False]      # an item the judge omitted counts as not covered
    assert g["contradiction"] is True and g["error"] is None


def test_grade_failure_counts_as_nothing_covered_and_is_reported(monkeypatch):
    def boom(*a, **k):
        raise rq.GenerationError("HTTP 402")

    monkeypatch.setattr(rq, "chat", boom)
    g = rq.grade("Q?", ["A."], [{"text": "p1"}, {"text": "p2"}])
    assert g["covered"] == [False, False] and g["error"]


def test_summary_separates_traps_errors_and_retrieval_misses():
    rows = [
        {"trap": False, "completeness": 1.0, "retrieved_gold": True},
        {"trap": False, "completeness": 0.5, "retrieved_gold": False, "contradiction": True},
        {"trap": False, "error": "boom"},
        {"trap": True, "declined": True},
        {"trap": True, "declined": False},
    ]
    s = rq.summarize(rows)
    assert s["n"] == 2 and s["mean_completeness"] == 0.75 and s["fully_complete"] == 0.5
    assert s["mean_completeness_when_gold_retrieved"] == 1.0 and s["n_gold_retrieved"] == 1
    assert s["contradictions"] == 1 and s["trap_pass"] == "1/2" and s["errors"] == 1
