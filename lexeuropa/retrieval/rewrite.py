"""Rewrite a lay question in the vocabulary of the legal texts, as an EXTRA search query.

"What information can a person get when they ask an organisation for access to their data?"
matches other laws' data-access rules; "right of access by the data subject, GDPR Article 15,
confirmation of processing, purposes, recipients, retention period" matches the article that
answers it. The rewrite is only added as a second query (the original still searches), and any
failure just means no extra query.
"""
from __future__ import annotations

from typing import Callable

from lexeuropa.config import RELEVANCE_LLM_BASE_URL, RELEVANCE_LLM_MODEL
from lexeuropa.generation.client import GenerationError, chat

REWRITE_PROMPT = """Rewrite this question about EU law as ONE short search query written in the terminology of the legal texts themselves: the legal concept (use the term the legislation uses, e.g. "data subject", "controller", "gatekeeper", "undertaking", "provider"), the most likely instrument by its usual name, and the typical headings of the articles that would answer it.

Example: "Can my boss read my work emails?" -> "employer monitoring of employees' electronic communications processing of personal data legitimate interests proportionality GDPR"

Output ONLY the query on one line, no quotes or explanation.

Question: {question}"""


def rewrite_query(question: str, *, chat_fn: Callable = chat) -> str | None:
    try:
        raw, _finish = chat_fn(
            [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
            temperature=0.0, max_tokens=80, model=RELEVANCE_LLM_MODEL, base_url=RELEVANCE_LLM_BASE_URL,
        )
    except GenerationError:
        return None
    line = raw.strip().splitlines()[0].strip().strip('"\'') if raw.strip() else ""
    return line if 8 <= len(line) <= 400 else None
