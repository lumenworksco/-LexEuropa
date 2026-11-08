"""Does the answer address the question that was asked?

The per-claim judge (verifier.py) only checks claim-vs-cited-text, so a
correct quotation of the wrong paragraph passes as "supported". This is the
missing question-level check: one call to a stronger model, with the question
and the answer's statements. It returns (True|False|None, reason); None means
"could not tell" (call failed or output unusable) and never triggers anything.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from lexeuropa.config import RELEVANCE_LLM_BASE_URL, RELEVANCE_LLM_MODEL
from lexeuropa.generation.client import GenerationError, chat

RELEVANCE_PROMPT = """You are checking an AI-generated answer to a legal question.

Question: {question}

Statements in the answer:
{statements}

Do these statements, taken together, directly answer the question that was asked?
Answer false if they address only a different point, only a side issue, an exception
without the main rule, or otherwise do not respond to what was asked.
Answer true if they give a direct answer to the question, even if it is incomplete.

Respond with ONLY a JSON object: {{"addresses_question": true or false, "reason": "<one sentence>"}}"""


def check_relevance(
    question: str, statements: list[str], *, chat_fn: Callable = chat
) -> tuple[bool | None, str]:
    if not statements:
        return None, "no statements to check"
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, start=1))
    prompt = RELEVANCE_PROMPT.format(question=question, statements=numbered)
    try:
        raw, _finish = chat_fn(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            model=RELEVANCE_LLM_MODEL,
            base_url=RELEVANCE_LLM_BASE_URL,
        )
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
        data, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
        verdict = data.get("addresses_question")
        if not isinstance(verdict, bool):
            return None, "unusable relevance verdict"
        return verdict, str(data.get("reason", "")).strip()
    except (GenerationError, ValueError, json.JSONDecodeError) as e:
        return None, f"relevance check unavailable: {type(e).__name__}"
