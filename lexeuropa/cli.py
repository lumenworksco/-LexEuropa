"""Command-line entry point: `python -m lexeuropa.cli <ingest|ask> ...`"""
from __future__ import annotations

import argparse
import json
import sys

from lexeuropa.answer import NotIngestedError, answer_question
from lexeuropa.ingest.pipeline import run_ingest


def _cmd_ingest(args: argparse.Namespace) -> int:
    run_ingest(build_dense=not args.no_dense)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    try:
        result = answer_question(
            args.question,
            default_instrument=args.instrument,
            k_candidates=args.k_candidates,
            k_final=args.k_final,
        )
    except NotIngestedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"\nQuestion: {result['question']}")
    if result.get("error"):
        print(f"\nerror: {result['error']}")
        return 1

    print(f"Answer language: {result['language']}")
    print(f"Reranked: {result['reranked']}")

    print("\nAuthorities retrieved:")
    for c in result["retrieved"]:
        print(f"  [{c['id']}] {c['citation']} ({c['lang']})")

    print("\nAnswer:")
    for claim in result["claims"]:
        cites = ", ".join(claim["citations"]) or "(none)"
        print(f"\n  [{claim['label'].upper()}] {claim['text']}")
        print(f"    citations: {cites}")
        if claim["reason"]:
            print(f"    reason: {claim['reason']}")
        per_citation = claim.get("citation_labels", {})
        distinct_labels = {v["label"] for v in per_citation.values()}
        if len(per_citation) > 1 and len(distinct_labels) > 1:
            print("    per-citation breakdown:")
            for cid, v in per_citation.items():
                print(f"      [{v['label'].upper()}] {cid}: {v['reason']}")

    print(f"\n{result['disclaimer']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lexeuropa")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="fetch, parse, chunk and index the seed corpus")
    p_ingest.add_argument(
        "--no-dense", action="store_true", help="skip building the dense (bge-m3) index"
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a legal research question against the ingested corpus")
    p_ask.add_argument("question")
    p_ask.add_argument(
        "--instrument",
        default=None,
        help="default instrument alias (e.g. gdpr) to resolve bare article references against",
    )
    p_ask.add_argument("--k-candidates", type=int, default=20)
    p_ask.add_argument("--k-final", type=int, default=8)
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
