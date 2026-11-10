"""The public demo's credential rules: no owner token, no call without a visitor token, no mixing of visitors."""
import json
import threading
import time
import urllib.request

import pytest

import lexeuropa.auth_context as ac
import lexeuropa.generation.client as client
import lexeuropa.answer as answer_mod
from lexeuropa.generation.client import ChatResult, GenerationError
from lexeuropa.schema import Chunk
from lexeuropa.verification.verifier import verify_claims


def _chunk(n: int) -> Chunk:
    return Chunk(id=f"X:en:art_{n}:001", celex="X", lang="en", instrument_title="t", citation=f"Article {n} X",
                 unit_type="article_paragraph", article=str(n), para_index="1", text="text", source_url="")


class _FakeResponse:
    def __init__(self, body): self._b = body
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _ok_response():
    return _FakeResponse(json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode())


@pytest.fixture
def public(monkeypatch):
    """Public-demo mode with an owner token that must never be used."""
    monkeypatch.setattr(ac, "PUBLIC_MODE", True)
    monkeypatch.setattr(client, "PUBLIC_MODE", True)
    monkeypatch.setattr(ac, "HF_TOKEN", "hf_OWNER_SECRET")


def test_public_mode_makes_no_request_at_all_without_a_visitor_token(public, monkeypatch):
    def boom(*a, **k): raise AssertionError("a network request was made without a visitor token")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ac.SignInRequired):
        client.chat([{"role": "user", "content": "q"}])


def test_public_mode_uses_only_the_visitor_token_never_the_owner_token(public, monkeypatch):
    seen = []
    def fake_urlopen(req, timeout=None):
        seen.append(req.get_header("Authorization"))
        return _ok_response()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with ac.bound_token("hf_visitor"):
        client.chat([{"role": "user", "content": "q"}])
    assert seen == ["Bearer hf_visitor"]
    assert ac.current_token() is None      # nothing leaks out of the `with`; still no owner fallback


def test_local_mode_still_falls_back_to_the_owner_token(monkeypatch):
    monkeypatch.setattr(ac, "PUBLIC_MODE", False)
    monkeypatch.setattr(ac, "HF_TOKEN", "hf_owner")
    assert ac.current_token() == "hf_owner"
    with ac.bound_token("hf_other"):
        assert ac.current_token() == "hf_other"


def test_concurrent_visitors_never_share_a_token_through_the_verifier_fan_out(public):
    """Two requests at once, each fanning out over thread pools (claims, then citations): every judge call
    must carry its own request's token. (A module-level token, or a ContextVar that does not follow the
    pool workers, would fail this.)"""
    chunks = [_chunk(i) for i in range(1, 5)]
    claims = [{"text": f"claim {i}", "citations": [c.id for c in chunks[:3]]} for i in range(6)]
    seen = {"A": set(), "B": set()}
    barrier = threading.Barrier(2)

    def judge_for(who):
        def judge(text, cited):
            time.sleep(0.01)                        # interleave the two requests' workers
            seen[who].add(ac.current_token())
            return ("supported", "ok")
        return judge

    def request(who, token):
        with ac.bound_token(token):
            barrier.wait()
            verify_claims(claims, chunks, judge_fn=judge_for(who))

    threads = [threading.Thread(target=request, args=("A", "hf_visitor_A")), threading.Thread(target=request, args=("B", "hf_visitor_B"))]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert seen == {"A": {"hf_visitor_A"}, "B": {"hf_visitor_B"}}


def test_stock_generator_answers_in_one_pass_with_the_guidance_and_no_second_pass(monkeypatch):
    from lexeuropa.retrieval.pipeline import RetrievalResult
    from lexeuropa.retrieval.reference_resolver import ReferenceResolver
    chunk = _chunk(17)
    monkeypatch.setattr(answer_mod, "STOCK_GENERATOR", True)
    monkeypatch.setattr(answer_mod, "RELEVANCE_CHECK", False)
    monkeypatch.setattr(answer_mod, "QUERY_REWRITE", False)
    monkeypatch.setattr(answer_mod, "_load_indices", lambda: (None, None, ReferenceResolver([])))
    monkeypatch.setattr(answer_mod, "retrieve", lambda *a, **k: RetrievalResult(chunks=[chunk], scores={chunk.id: 1.0}, reranked=False))
    monkeypatch.setattr(answer_mod, "verify_claims", lambda claims, chunks: [])
    calls = []
    def fake_chat(messages, *, base_url=None, model=None, **kw):
        calls.append((base_url, model, messages[0]["content"]))
        return ChatResult(content='{"claims": [{"text": "Nothing here.", "citations": []}]}', finish_reason="stop")   # a refusal
    monkeypatch.setattr(answer_mod, "chat", fake_chat)
    out = answer_mod.answer_question("A question?")
    assert len(calls) == 1                                        # a refusal does not trigger a second generation
    assert calls[0][0] == answer_mod.FALLBACK_LLM_BASE_URL and calls[0][1] == answer_mod.FALLBACK_LLM_MODEL
    assert "Additional instructions" in calls[0][2]
    assert out["generator"] == "stock" and out["generator_model"] == answer_mod.FALLBACK_LLM_MODEL
