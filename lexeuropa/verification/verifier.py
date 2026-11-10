"""Citation and evidence verification.

For every claim the generator makes, this module checks:
  1. Existence gate: does every cited source id actually appear in the set
     of chunks that were retrieved for this question? A citation to
     anything else is treated as fabricated, full stop — no judge call
     needed, because a citation the retriever never surfaced cannot have
     been read by the model in good faith.
  2. Support check: for claims that pass the gate, an LLM-judge call
     classifies how strongly the cited text supports the claim.

The judge is injectable (`judge_fn`) specifically so tests can verify the
existence gate deterministically without needing a live LLM.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from lexeuropa.auth_context import map_in_context
from dataclasses import dataclass, field
from typing import Callable

from lexeuropa.config import CASE_NUMBER_ALIASES, JUDGE_BASE_URL, JUDGE_MODEL
from lexeuropa.generation.client import GenerationError, chat
from lexeuropa.retrieval.reference_resolver import _CASE_RE
from lexeuropa.schema import Chunk

LABELS = {"supported", "interpretative", "analysis", "unsupported", "uncertain"}


@dataclass
class ClaimVerification:
    text: str
    citations: list[str]
    label: str
    reason: str
    # Per-citation judge outcome (chunk id -> (label, reason)), judged
    # independently of the other citations on the same claim. `label`
    # above is the best (most supportive) of these — see
    # `_aggregate_citation_labels` — so a weak co-citation is still
    # visible here even when it doesn't drag the claim-level label down.
    citation_labels: dict[str, tuple[str, str]] = field(default_factory=dict)


_LABEL_RANK = {"supported": 0, "interpretative": 1, "analysis": 2, "unsupported": 3, "uncertain": 4}


def _aggregate_citation_labels(citation_labels: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """Picks the most supportive per-citation outcome as the claim's
    overall label — a claim holds if ANY of its citations backs it. The
    reason flags any other citation that was individually weaker, so a
    co-citation that didn't actually pull weight stays visible instead of
    disappearing behind the one that did (the bug this replaces: judging
    every cited chunk together let one relevant chunk launder an
    unrelated one into "supported")."""
    best_id = min(citation_labels, key=lambda cid: _LABEL_RANK[citation_labels[cid][0]])
    best_label, best_reason = citation_labels[best_id]
    weaker = [
        cid
        for cid, (label, _) in citation_labels.items()
        if cid != best_id and _LABEL_RANK[label] > _LABEL_RANK[best_label]
    ]
    if weaker:
        best_reason = f"{best_reason} (note: did not independently support: {', '.join(weaker)})"
    return best_label, best_reason


JudgeFn = Callable[[str, list[Chunk]], tuple[str, str]]


def _parse_json_object(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in: {raw!r}")
    return json.loads(text[start : end + 1])


JUDGE_PROMPT = """Claim: {claim}

Cited source excerpts:
{sources}

Classify the relationship between the claim and the cited excerpts as exactly one of:
- supported: the excerpts directly and explicitly state this
- interpretative: the excerpts support this only through legal/judicial interpretation, not a literal statement
- analysis: this is a reasoned conclusion combining multiple excerpts, not stated directly in any one of them
- unsupported: the excerpts do not actually support this claim
- uncertain: it is not possible to tell from the excerpts alone

Respond with ONLY a JSON object: {{"label": "<one of the above>", "reason": "<one sentence>"}}"""


def default_llm_judge(claim_text: str, cited_chunks: list[Chunk]) -> tuple[str, str]:
    sources = "\n".join(f"- [{c.id}] {c.citation}: {c.text}" for c in cited_chunks)
    prompt = JUDGE_PROMPT.format(claim=claim_text, sources=sources)
    try:
        raw, _finish_reason = chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            model=JUDGE_MODEL,
            base_url=JUDGE_BASE_URL,
        )
    except GenerationError as e:
        # A judge outage must not take down the whole `ask` call — surface
        # it as "uncertain" (we genuinely can't tell) with the real cause
        # in the reason, rather than letting the exception propagate.
        return "uncertain", f"judge unavailable: {e}"
    try:
        data = _parse_json_object(raw)
        label = data.get("label", "uncertain")
        if label not in LABELS:
            label = "uncertain"
        return label, data.get("reason", "")
    except Exception:
        return "uncertain", "judge response could not be parsed"


def _source_identity_mismatch(claim_text: str, cited_chunks: list[Chunk]) -> str | None:
    """Catches a claim that explicitly names one case but cites a
    different judgment — observed live: a claim about C-131/12 cited only
    C-136/17 paragraphs that happen to quote C-131/12's holding verbatim,
    and the judge approved it on text content alone, since nothing checks
    whether the cited source is the authority actually named. Faithful
    text doesn't mean it's attributed to the right case."""
    mentioned_celexes = set()
    for m in _CASE_RE.finditer(claim_text):
        key = f"c-{m.group(1)}/{m.group(2)}"
        celex = CASE_NUMBER_ALIASES.get(key)
        if celex:
            mentioned_celexes.add(celex)
    if len(mentioned_celexes) != 1:
        return None  # no case named, or ambiguous — don't guess
    (mentioned,) = mentioned_celexes
    cited_celexes = {c.celex for c in cited_chunks}
    other_known_judgments = cited_celexes & set(CASE_NUMBER_ALIASES.values())
    if mentioned not in cited_celexes and other_known_judgments:
        wrong = ", ".join(sorted(other_known_judgments))
        return (
            f"claim names a case resolving to CELEX {mentioned}, but the cited source(s) "
            f"are from a different judgment ({wrong})"
        )
    return None


def verify_claim(
    claim: dict, retrieved_by_id: dict[str, Chunk], judge_fn: JudgeFn = default_llm_judge
) -> ClaimVerification:
    text = claim["text"]
    citations = claim.get("citations", [])

    if not citations:
        return ClaimVerification(
            text, citations, "uncertain", "the model gave no citation for this claim"
        )

    missing = [cid for cid in citations if cid not in retrieved_by_id]
    if missing:
        return ClaimVerification(
            text,
            citations,
            "unsupported",
            f"cited source id(s) were not among the retrieved evidence: {missing}",
        )

    cited_chunks = [retrieved_by_id[cid] for cid in citations]

    mismatch_reason = _source_identity_mismatch(text, cited_chunks)
    if mismatch_reason:
        return ClaimVerification(text, citations, "unsupported", mismatch_reason)

    # Judge every cited chunk on its own, not all together — see
    # `_aggregate_citation_labels` for why.
    # Calls are independent network round-trips, so run them concurrently.
    with ThreadPoolExecutor(max_workers=min(len(cited_chunks), 8)) as pool:
        results = map_in_context(pool, lambda chunk: judge_fn(text, [chunk]), cited_chunks)
    citation_labels = {chunk.id: r for chunk, r in zip(cited_chunks, results)}
    label, reason = _aggregate_citation_labels(citation_labels)
    return ClaimVerification(text, citations, label, reason, citation_labels)


def verify_claims(
    claims: list[dict], retrieved: list[Chunk], judge_fn: JudgeFn = default_llm_judge
) -> list[ClaimVerification]:
    retrieved_by_id = {c.id: c for c in retrieved}
    if len(claims) <= 1:
        return [verify_claim(c, retrieved_by_id, judge_fn) for c in claims]
    # Claims are independent network round-trips (an answer that quotes a long
    # article can have 14 of them), so check them concurrently; order is kept.
    with ThreadPoolExecutor(max_workers=min(len(claims), 6)) as pool:
        return map_in_context(pool, lambda c: verify_claim(c, retrieved_by_id, judge_fn), claims)
