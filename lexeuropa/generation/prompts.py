"""Prompt construction and structured-output parsing for answer generation.

Citations use short handles ("S1") instead of full chunk ids, and
parse_response repairs a specific observed truncation pattern — see
CLAUDE.md's "Known limitations" for what was tried and what the evidence
showed.
"""
from __future__ import annotations

import json
import re
import sys

from lexeuropa.schema import Chunk

SYSTEM_PROMPT = """You are a European Union legal research assistant.

Rules:
- Ground every factual claim ONLY in the source excerpts given to you (each has a handle).
- Each claim must list the exact source handle(s) (e.g. "S1") it is based on — not the citation text, not the id, just the handle.
- NEVER invent, guess, or reuse a handle that was not given to you.
- Keep each claim's "text" to 1-2 sentences. Split a longer answer into several short claims instead of one long one.
- If the sources do not answer part of the question, add a claim describing
  that gap with an empty "citations" list instead of guessing.
- Always answer in English, regardless of what language the question is asked in.
- Output ONLY a single JSON object, no markdown fences, no commentary, matching:
  {"language": "en", "claims": [{"text": "<string>", "citations": ["S1", ...]}, ...]}

This assists legal research. It is not legal advice and does not replace a qualified lawyer."""


def build_messages(question: str, sources: list[Chunk]) -> tuple[list[dict], dict[str, str]]:
    """Returns (messages, handle_to_chunk_id) — the caller must map citation
    handles back to real chunk ids (see module docstring) before verifying.
    """
    lines = []
    handle_to_id: dict[str, str] = {}
    for i, s in enumerate(sources, start=1):
        handle = f"S{i}"
        handle_to_id[handle] = s.id
        lines.append(f'- handle: "{handle}"\n  citation: {s.citation}\n  text: {s.text}')
    sources_block = "\n".join(lines) if lines else "(no sources retrieved)"

    user = f"Sources:\n{sources_block}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return messages, handle_to_id


class ParseError(RuntimeError):
    pass


def _repair_missing_closing_brackets(text: str) -> str | None:
    """Append missing closing brackets/braces; None if unsafe to (see CLAUDE.md)."""
    stack = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if in_string or not stack:
        return None
    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[c] for c in reversed(stack))


def parse_response(raw: str, *, allow_repair: bool = True) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    start = text.find("{")
    if start == -1:
        raise ParseError(f"no JSON object found in model output: {raw!r}")
    body = text[start:]

    end = text.rfind("}")
    candidate = text[start : end + 1] if end >= start else body
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        repaired = _repair_missing_closing_brackets(body) if allow_repair else None
        if repaired is None:
            raise ParseError(f"model output was not valid JSON: {e}\nraw: {raw!r}") from e
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            raise ParseError(
                f"model output was not valid JSON, even after attempting to repair "
                f"unterminated brackets: {e}\nraw: {raw!r}"
            ) from e
        print(
            "warning: model output was missing its closing JSON brackets — repaired and parsed anyway",
            file=sys.stderr,
        )

    if "claims" not in data or not isinstance(data["claims"], list):
        raise ParseError(f"model output missing a 'claims' list: {data!r}")
    for claim in data["claims"]:
        if "text" not in claim or "citations" not in claim:
            raise ParseError(f"malformed claim (needs text + citations): {claim!r}")
        citations = claim["citations"]
        if isinstance(citations, str):
            # A smaller model will sometimes emit a single citation as a
            # bare string instead of a one-element list. Iterating over
            # that string character-by-character downstream would make a
            # correct citation look like dozens of fabricated one-letter
            # ones — normalize here instead of exploding into that mess.
            claim["citations"] = [citations]
        elif not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
            raise ParseError(f"claim 'citations' must be a string or list of strings: {claim!r}")
    return data
