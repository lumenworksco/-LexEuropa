"""Ties retrieval, generation and verification into one `ask` call.

Kept separate from cli.py so the eval harness can call `answer_question`
directly instead of shelling out.
"""
from __future__ import annotations

from lexeuropa.config import BM25_INDEX_PATH, DENSE_INDEX_PATH, INSTRUMENT_ALIASES
from lexeuropa.generation.client import GenerationError, chat
from lexeuropa.generation.prompts import ParseError, build_messages, parse_response
from lexeuropa.ingest.pipeline import load_chunks
from lexeuropa.retrieval.bm25_index import BM25Index
from lexeuropa.retrieval.dense_index import DenseIndex
from lexeuropa.retrieval.pipeline import retrieve
from lexeuropa.retrieval.reference_resolver import ReferenceResolver
from lexeuropa.verification.verifier import verify_claims

DISCLAIMER = (
    "This is AI-assisted legal research output, not legal advice. "
    "It does not replace a qualified legal professional."
)


class NotIngestedError(RuntimeError):
    pass


def _load_indices():
    try:
        chunks = load_chunks()
    except FileNotFoundError as e:
        raise NotIngestedError(
            "no ingested corpus found — run `python -m lexeuropa.cli ingest` first"
        ) from e

    chunks_by_id = {c.id: c for c in chunks}
    bm25 = BM25Index.load(BM25_INDEX_PATH)
    dense = DenseIndex.load(DENSE_INDEX_PATH, chunks_by_id)
    resolver = ReferenceResolver(chunks)
    return bm25, dense, resolver


def answer_question(
    question: str,
    *,
    default_instrument: str | None = None,
    k_candidates: int = 20,
    k_final: int = 8,
) -> dict:
    bm25, dense, resolver = _load_indices()

    default_celex = None
    if default_instrument:
        default_celex = INSTRUMENT_ALIASES.get(default_instrument.lower())

    result = retrieve(
        question,
        bm25_index=bm25,
        dense_index=dense,
        resolver=resolver,
        k_candidates=k_candidates,
        k_final=k_final,
        default_celex=default_celex,
    )

    if not result.chunks:
        return {
            "question": question,
            "language": None,
            "retrieved": [],
            "reranked": result.reranked,
            "claims": [],
            "disclaimer": DISCLAIMER,
            "error": "no relevant sources were retrieved for this question",
        }

    messages, handle_to_id = build_messages(question, result.chunks)
    try:
        raw, finish_reason = chat(messages)
    except GenerationError as e:
        return {
            "question": question,
            "language": None,
            "retrieved": [c.to_dict() for c in result.chunks],
            "reranked": result.reranked,
            "claims": [],
            "disclaimer": DISCLAIMER,
            "error": str(e),
        }

    try:
        # Skip the bracket-repair path on a real length cutoff: that failure
        # mode can truncate mid-string, which isn't safely repairable by
        # appending closing brackets the way a missing-final-brace slip is.
        parsed = parse_response(raw, allow_repair=finish_reason != "length")
    except ParseError as e:
        truncation_note = (
            " (generation stopped due to finish_reason=length — the output was cut "
            "off mid-response; see the context-window notes in CLAUDE.md)"
            if finish_reason == "length"
            else ""
        )
        return {
            "question": question,
            "language": None,
            "retrieved": [c.to_dict() for c in result.chunks],
            "reranked": result.reranked,
            "claims": [],
            "disclaimer": DISCLAIMER,
            "error": f"could not parse model output: {e}{truncation_note}",
        }

    # Map source handles ("S1") back to real chunk ids before verifying. An
    # unrecognized handle is left as-is — it then simply won't match any
    # retrieved chunk id, so the verifier's existence gate still catches it.
    for claim in parsed["claims"]:
        claim["citations"] = [handle_to_id.get(h, h) for h in claim["citations"]]

    verifications = verify_claims(parsed["claims"], result.chunks)

    return {
        "question": question,
        "language": parsed.get("language"),
        "retrieved": [c.to_dict() for c in result.chunks],
        "reranked": result.reranked,
        "claims": [
            {
                "text": v.text,
                "citations": v.citations,
                "label": v.label,
                "reason": v.reason,
                "citation_labels": {
                    cid: {"label": lbl, "reason": r} for cid, (lbl, r) in v.citation_labels.items()
                },
            }
            for v in verifications
        ],
        "disclaimer": DISCLAIMER,
    }
