"""Record real answers for the static demo: site/demo/gallery.json.

    python scripts/build_gallery.py --resume-endpoint            # all questions, one command
    python scripts/build_gallery.py --only breach,erasure         # just those (merged into the existing file)

The questions are fixed in eval/gallery_questions.json before anything is generated; every answer is stored
as the system produced it (not edited, not cherry-picked), including refusals and any failure. Uses the product's
default configuration (fine-tuned model on its Inference Endpoint, fallback, relevance check, citation judge), so
it needs credits: roughly $1 including the endpoint's time. With --resume-endpoint the endpoint is woken first
(a real request, not just resume(): a scale-to-zero endpoint answers 503 until warm) and scaled to zero afterwards.
Re-running skips questions that already have a successful answer.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "gallery_questions.json"
OUT = ROOT / "site" / "demo" / "gallery.json"
ENDPOINT_NAME = "lexeuropa-eval"
MAX_EXTRA_RETRIEVED = 5       # besides the cited passages, keep this many "also retrieved" ones


def trim(result: dict) -> dict:
    """Keep what the page needs: all cited passages and a few others (the rest of the retrieved list is dropped)."""
    cited = {cid for c in result.get("claims", []) for cid in c.get("citations", [])}
    kept, extra = [], 0
    for chunk in result.get("retrieved", []):
        if chunk["id"] in cited:
            kept.append(chunk)
        elif extra < MAX_EXTRA_RETRIEVED:
            kept.append(chunk); extra += 1
    out = dict(result)
    out["retrieved"] = kept
    return out


def warm_endpoint() -> None:
    from huggingface_hub import get_inference_endpoint

    sys.path.insert(0, str(ROOT))
    from lexeuropa.generation.client import GenerationError, chat

    ep = get_inference_endpoint(ENDPOINT_NAME)
    ep.resume()
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            chat([{"role": "user", "content": "Say OK."}], max_tokens=5, timeout=120)
            print("endpoint warm", flush=True)
            return
        except GenerationError as e:
            print(f"  waiting for the endpoint ({str(e)[:60]})", flush=True)
            time.sleep(20)
    raise SystemExit("the endpoint did not become ready in 15 minutes")


def scale_to_zero() -> None:
    from huggingface_hub import get_inference_endpoint

    get_inference_endpoint(ENDPOINT_NAME).scale_to_zero()
    print("endpoint scaled to zero", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="comma-separated question ids")
    parser.add_argument("--resume-endpoint", action="store_true", help="wake the endpoint first and scale it to zero afterwards")
    parser.add_argument("--redo", action="store_true", help="re-answer questions that already have an answer")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from lexeuropa.answer import answer_question
    from lexeuropa.config import FALLBACK_LLM_MODEL, JUDGE_MODEL, LLM_MODEL

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"items": []}
    by_id = {it["id"]: it for it in existing["items"]}
    todo = [q for q in questions if args.redo or q["id"] not in by_id or by_id[q["id"]]["result"].get("error")]
    print(f"{len(todo)} of {len(questions)} questions to answer")
    if not todo:
        return 0
    if args.resume_endpoint:
        warm_endpoint()
    today = datetime.date.today().isoformat()
    try:
        for n, q in enumerate(todo, start=1):
            result = None
            for attempt in range(3):
                started = time.time()
                result = answer_question(q["question"])
                if not result.get("error") or "503" not in str(result.get("error")):
                    break
                print("  503, waiting for the model ...", flush=True); time.sleep(45)
            result["elapsed_s"] = round(time.time() - started, 1)
            result["recorded"], result["recorded_at"] = True, today
            by_id[q["id"]] = {"id": q["id"], "question": q["question"], "result": trim(result)}
            status = "ERROR " + str(result["error"])[:80] if result.get("error") else f"{len(result.get('claims', []))} claims, generator={result.get('generator')}"
            print(f"[{n}/{len(todo)}] {q['id']}: {status} ({result['elapsed_s']} s)", flush=True)
            order = [x["id"] for x in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]]
            items = sorted(by_id.values(), key=lambda it: order.index(it["id"]) if it["id"] in order else 999)
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps({
                "generated_at": today, "commit": commit,
                "models": {"generator": LLM_MODEL, "fallback_and_relevance": FALLBACK_LLM_MODEL, "citation_judge": JUDGE_MODEL},
                "items": items,
            }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")      # saved after every question
    finally:
        if args.resume_endpoint:
            scale_to_zero()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
