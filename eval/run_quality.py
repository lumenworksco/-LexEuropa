"""Answer-quality evaluation: is the answer COMPLETE and CORRECT, not just well cited?

`eval/answer_quality.jsonl` has questions where completeness is at stake (rules with
exceptions, lists, multi-part conditions). Each has a checklist of key points a complete
answer must convey; every point was grounded in the real provision text (see
`tests/test_quality_set.py`). A stronger model grades each answer against the checklist
(reference-guided grading is far more reliable than open-ended grading). Four "trap"
questions are outside the corpus (national law, US law, outcome prediction, off-topic);
the right behaviour is to decline, and answering them with citations is a failure.

    python eval/run_quality.py --out data/finetune/quality_run.json         # the system as configured
    python eval/run_quality.py --calibrate                                  # check the judge itself

Metrics: mean completeness, share of fully complete answers, completeness when the gold
provision was retrieved (separates retrieval misses from generation gaps), answers the
judge found to contradict the provision, and the trap pass rate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lexeuropa.config import RELEVANCE_LLM_BASE_URL, RELEVANCE_LLM_MODEL
from lexeuropa.generation.client import GenerationError, chat

DEFAULT_FILE = Path(__file__).parent / "answer_quality.jsonl"

QUALITY_PROMPT = """You are grading an answer to a legal question against a checklist.

Question: {question}

Answer (numbered statements):
{statements}

Checklist of key points a complete answer must convey:
{points}

For each checklist item, decide whether the answer conveys it. The meaning must be present
in the answer; the wording may differ. A passing reference that omits the substance does not
count. Also decide whether any statement in the answer contradicts the checklist or states
something clearly wrong about the provision.

Respond with ONLY a JSON object:
{{"points": [{{"n": 1, "covered": true}}, ...], "contradiction": false, "contradiction_note": ""}}"""


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def grade(question: str, statements: list[str], points: list[dict]) -> dict:
    """Judge one answer. Returns {"covered": [bool...], "contradiction": bool, "note": str, "error": str|None}."""
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, start=1)) or "(no statements)"
    checklist = "\n".join(f"{i}. {p['text']}" for i, p in enumerate(points, start=1))
    prompt = QUALITY_PROMPT.format(question=question, statements=numbered, points=checklist)
    last_error = ""
    for attempt in range(4):  # a flaky gateway (HTTP 504/5xx) must not become "the answer covered nothing"
        try:
            raw, _ = chat(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=900,
                model=RELEVANCE_LLM_MODEL, base_url=RELEVANCE_LLM_BASE_URL,
            )
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
            data, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
            by_n = {int(p["n"]): bool(p["covered"]) for p in data.get("points", []) if "n" in p}
            return {
                "covered": [by_n.get(i, False) for i in range(1, len(points) + 1)],
                "contradiction": bool(data.get("contradiction", False)),
                "note": str(data.get("contradiction_note", "")),
                "error": None,
            }
        except (GenerationError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            last_error = f"{type(e).__name__}: {e}"
            time.sleep((2, 5, 10, 0)[attempt])
    return {"covered": [False] * len(points), "contradiction": False, "note": "", "error": last_error}


def is_gold(chunk: dict, gold: list[dict]) -> bool:
    return any(chunk["celex"] == g["celex"] and (g["article"] is None or chunk["article"] == g["article"]) for g in gold)


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if not r.get("trap") and not r.get("error") and not r.get("grade_error")]
    traps = [r for r in rows if r.get("trap")]
    comp = [r["completeness"] for r in scored]
    with_gold = [r["completeness"] for r in scored if r.get("retrieved_gold")]
    return {
        "n": len(scored),
        "mean_completeness": round(sum(comp) / len(comp), 3) if comp else None,
        "fully_complete": round(sum(1 for c in comp if c == 1.0) / len(comp), 3) if comp else None,
        "mean_completeness_when_gold_retrieved": round(sum(with_gold) / len(with_gold), 3) if with_gold else None,
        "n_gold_retrieved": len(with_gold),
        "contradictions": sum(1 for r in scored if r.get("contradiction")),
        "trap_pass": f"{sum(1 for r in traps if r.get('declined'))}/{len(traps)}" if traps else None,
        "errors": sum(1 for r in rows if r.get("error")),
        "grader_failures_excluded": sum(1 for r in rows if r.get("grade_error")),
    }


def run_system(items: list[dict], out_path: Path | None = None) -> list[dict]:
    from lexeuropa.answer import answer_question

    rows = []
    for item in items:
        out = answer_question(item["question"])
        row = {"id": item["id"], "question": item["question"], "trap": bool(item.get("expect_refusal")),
               "generator": out.get("generator"), "answers_question": out.get("answers_question"), "error": out.get("error")}
        claims = out.get("claims", [])
        statements = [c["text"] for c in claims]
        row["statements"] = statements
        row["declined"] = not any(c["citations"] for c in claims)
        if not row["trap"]:
            row["retrieved_gold"] = any(is_gold(c, item["gold"]) for c in out.get("retrieved", []))
            if not out.get("error"):
                g = grade(item["question"], statements, item["points"])
                row.update(covered=g["covered"], contradiction=g["contradiction"], contradiction_note=g["note"], grade_error=g["error"])
                row["completeness"] = round(sum(g["covered"]) / len(g["covered"]), 3)
        status = f"trap declined={row['declined']}" if row["trap"] else f"completeness={row.get('completeness')} gold_retrieved={row.get('retrieved_gold')}"
        print(f"[{item['id']}] {status}{' ERROR ' + row['error'] if row['error'] else ''}", flush=True)
        rows.append(row)
        if out_path:  # partial results survive a stalled provider
            out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return rows


def regrade(items: list[dict], path: Path, everything: bool) -> int:
    """Re-grade the saved answers in `path` (no answering model involved). By default only
    rows whose earlier grading failed; with --all, every row."""
    by_id = {i["id"]: i for i in items}
    rows = json.loads(path.read_text())
    n = 0
    for row in rows:
        item = by_id.get(row["id"])
        if not item or row.get("trap") or row.get("error"):
            continue
        if not everything and not row.get("grade_error"):
            continue
        g = grade(item["question"], row["statements"], item["points"])
        row.update(covered=g["covered"], contradiction=g["contradiction"], contradiction_note=g["note"], grade_error=g["error"])
        row["completeness"] = round(sum(g["covered"]) / len(g["covered"]), 3)
        n += 1
        print(f"[{row['id']}] regraded: completeness={row['completeness']}{' STILL FAILING' if g['error'] else ''}", flush=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nregraded {n} rows\n" + json.dumps(summarize(rows), indent=2))
    return 0


def calibrate(items: list[dict]) -> int:
    """Grade synthetic answers made from the real provision text, no answering model involved."""
    from lexeuropa.ingest.pipeline import load_chunks

    chunks = load_chunks()
    scored = [i for i in items if not i.get("expect_refusal")]

    def texts(item):
        g = item["gold"][0]
        cs = [c for c in chunks if c.celex == g["celex"] and (g["article"] is None or c.article == g["article"])]
        cs.sort(key=lambda c: int(c.para_index) if c.para_index.isdigit() else 0)
        return [c.text for c in cs][:40]

    def variants(idx: int, item: dict):
        full = texts(item)
        half = [" ".join(full)[: max(60, len(" ".join(full)) * 2 // 5)]]          # first ~40% of the provision
        other = texts(scored[(idx + 1) % len(scored)])[:3]                         # an unrelated provision
        return {"verbatim": full, "first_40pct": half, "unrelated": other}

    def one(args):
        idx, item = args
        return item["id"], {k: grade(item["question"], v, item["points"]) for k, v in variants(idx, item).items()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, enumerate(scored)))
    totals = {"verbatim": [], "first_40pct": [], "unrelated": []}
    for qid, per in results:
        line = []
        for k in totals:
            g = per[k]
            c = sum(g["covered"]) / len(g["covered"]) if g["covered"] else 0
            totals[k].append(c)
            line.append(f"{k}={c:.2f}{'!' if g['error'] else ''}")
        print(f"[{qid}] " + "  ".join(line))
    print()
    for k, v in totals.items():
        print(f"{k:12} mean completeness {sum(v) / len(v):.3f}  (n={len(v)})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--calibrate", action="store_true", help="grade synthetic answers to check the judge")
    parser.add_argument("--regrade", type=Path, default=None, help="re-grade the saved answers in this results file")
    parser.add_argument("--all", action="store_true", help="with --regrade: every row, not only failed gradings")
    parser.add_argument("--only", default=None, help="comma-separated question ids")
    args = parser.parse_args()
    items = load(args.file)
    if args.only:
        keep = set(args.only.split(","))
        items = [i for i in items if i["id"] in keep]
    if args.calibrate:
        return calibrate(items)
    if args.regrade:
        return regrade(items, args.regrade, args.all)
    rows = run_system(items, args.out)
    print("\n" + json.dumps(summarize(rows), indent=2))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
