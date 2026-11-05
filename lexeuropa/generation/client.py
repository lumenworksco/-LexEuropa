"""OpenAI-compatible chat client, pointed at HF Inference Providers by default — see CLAUDE.md."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import NamedTuple

from lexeuropa.config import HF_TOKEN, HTTP_USER_AGENT, LLM_BASE_URL, LLM_MODEL


class GenerationError(RuntimeError):
    pass


class ChatResult(NamedTuple):
    content: str
    finish_reason: str | None  # "length" means the completion was cut off


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.1,
    model: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 3000,
    # 240s, not 120s: the default LLM_BASE_URL is a scale-to-zero
    # Endpoint, and a cold request after it's scaled down needs time to
    # restart the container before it can answer at all.
    timeout: int = 240,
) -> ChatResult:
    model = model or LLM_MODEL
    base_url = base_url or LLM_BASE_URL
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json", "User-Agent": HTTP_USER_AGENT}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        hint = "HF_TOKEN is not set (see CLAUDE.md Setup)" if not HF_TOKEN else ""
        raise GenerationError(
            f"generation request to {base_url} failed: HTTP {e.code}: {body} {hint}".strip()
        ) from e
    except Exception as e:
        hint = (
            "HF_TOKEN is not set (see CLAUDE.md Setup)"
            if not HF_TOKEN and "huggingface.co" in base_url
            else f"is `ollama serve` running, and does `ollama list` show {model!r}?"
        )
        raise GenerationError(f"generation request to {base_url} failed: {e} ({hint})") from e

    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError) as e:
        raise GenerationError(f"unexpected response shape from LLM: {data}") from e

    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        print(
            "warning: generation stopped due to finish_reason=length — the model's "
            "output was cut off mid-response, which is likely why JSON parsing will fail",
            file=sys.stderr,
        )

    return ChatResult(content=content, finish_reason=finish_reason)
