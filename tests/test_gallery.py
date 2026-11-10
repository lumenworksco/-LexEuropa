"""The recorded-answers gallery: fixed questions, trimmed results."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_gallery as bg


def test_the_question_list_is_fixed_unique_and_includes_refusals():
    data = json.loads((ROOT / "eval" / "gallery_questions.json").read_text(encoding="utf-8"))
    qs = data["questions"]
    assert len(qs) == len({q["id"] for q in qs}) == len({q["question"] for q in qs})
    assert 20 <= len(qs) <= 30 and all(q["question"].endswith("?") for q in qs)
    assert {"tax_france", "prediction"} <= {q["id"] for q in qs}      # the deliberate out-of-scope refusals


def test_trim_keeps_every_cited_passage_and_only_a_few_others():
    retrieved = [{"id": f"c{i}", "text": "x"} for i in range(12)]
    result = {"claims": [{"citations": ["c9"]}, {"citations": ["c2", "c9"]}], "retrieved": retrieved, "question": "q"}
    out = bg.trim(result)
    ids = [c["id"] for c in out["retrieved"]]
    assert "c9" in ids and "c2" in ids
    assert len(ids) == 2 + bg.MAX_EXTRA_RETRIEVED
    assert out["question"] == "q" and len(result["retrieved"]) == 12      # the input is not mutated
