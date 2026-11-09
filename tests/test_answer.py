"""answer_question's error paths, with the LLM and indices mocked out."""
import json

import lexeuropa.answer as answer_mod
from lexeuropa.generation.client import ChatResult
from lexeuropa.retrieval.pipeline import RetrievalResult
from lexeuropa.retrieval.reference_resolver import ReferenceResolver
from lexeuropa.schema import Chunk

_EMPTY_RESOLVER = ReferenceResolver([])   # the real flow always has a resolver

CHUNK = Chunk(
    id="32016R0679:en:art_17:001",
    celex="32016R0679",
    lang="en",
    instrument_title="GDPR",
    citation="Article 17(1) GDPR",
    unit_type="article_paragraph",
    article="17",
    para_index="1",
    text="The data subject shall have the right to obtain erasure of personal data.",
    source_url="http://publications.europa.eu/resource/celex/32016R0679",
)


import pytest


@pytest.fixture(autouse=True)
def _no_network_relevance_check(monkeypatch):
    """The relevance check calls a model; keep every test hermetic unless it opts in."""
    monkeypatch.setattr(answer_mod, "RELEVANCE_CHECK", False)
    monkeypatch.setattr(answer_mod, "QUERY_REWRITE", False)


def test_truncated_generation_names_finish_reason_in_the_error(monkeypatch):
    monkeypatch.setattr(answer_mod, "_load_indices", lambda: (None, None, _EMPTY_RESOLVER))
    monkeypatch.setattr(
        answer_mod,
        "retrieve",
        lambda *a, **k: RetrievalResult(chunks=[CHUNK], scores={CHUNK.id: 1.0}, reranked=True),
    )
    # A JSON object with an unterminated claims list, exactly what a
    # length-limited completion produces.
    monkeypatch.setattr(
        answer_mod, "chat", lambda *a, **k: ChatResult(content='{"claims": [', finish_reason="length")
    )

    result = answer_mod.answer_question("Some question?")

    assert result["claims"] == []
    assert "finish_reason=length" in result["error"]


def test_parse_failure_without_length_cutoff_has_no_truncation_note(monkeypatch):
    monkeypatch.setattr(answer_mod, "_load_indices", lambda: (None, None, _EMPTY_RESOLVER))
    monkeypatch.setattr(
        answer_mod,
        "retrieve",
        lambda *a, **k: RetrievalResult(chunks=[CHUNK], scores={CHUNK.id: 1.0}, reranked=True),
    )
    monkeypatch.setattr(
        answer_mod, "chat", lambda *a, **k: ChatResult(content="not json at all", finish_reason="stop")
    )

    result = answer_mod.answer_question("Some question?")

    assert "finish_reason=length" not in result["error"]


def test_citation_handles_are_mapped_back_to_real_chunk_ids(monkeypatch):
    import lexeuropa.verification.verifier as verifier_mod

    monkeypatch.setattr(answer_mod, "_load_indices", lambda: (None, None, _EMPTY_RESOLVER))
    monkeypatch.setattr(
        answer_mod,
        "retrieve",
        lambda *a, **k: RetrievalResult(chunks=[CHUNK], scores={CHUNK.id: 1.0}, reranked=True),
    )
    # The model cites "S1" (the handle build_messages assigned to CHUNK,
    # since it's the only source) and a handle that was never given to it.
    model_output = {
        "language": "en",
        "claims": [
            {"text": "A real claim.", "citations": ["S1"]},
            {"text": "A fabricated claim.", "citations": ["S99"]},
        ],
    }
    monkeypatch.setattr(
        answer_mod, "chat", lambda *a, **k: ChatResult(content=json.dumps(model_output), finish_reason="stop")
    )
    # Only the existence gate is under test here, not the LLM-judge itself —
    # stub the judge's chat() call so this test doesn't hit a live model.
    monkeypatch.setattr(
        verifier_mod, "chat", lambda *a, **k: ChatResult(content='{"label": "supported", "reason": "stub"}', finish_reason="stop")
    )

    result = answer_mod.answer_question("Some question?")

    claims = {c["text"]: c for c in result["claims"]}
    assert claims["A real claim."]["citations"] == [CHUNK.id]
    assert claims["A real claim."]["label"] != "unsupported"
    assert claims["A fabricated claim."]["citations"] == ["S99"]
    assert claims["A fabricated claim."]["label"] == "unsupported"


def _cited_answer():
    return json.dumps({"language": "en", "claims": [{"text": "Erasure applies.", "citations": ["S1"]}]})


_REFUSAL = json.dumps(
    {"language": "en", "claims": [{"text": "The provided sources do not contain information.", "citations": []}]}
)


def _patch_retrieval(monkeypatch):
    monkeypatch.setattr(answer_mod, "_load_indices", lambda: (None, None, _EMPTY_RESOLVER))
    monkeypatch.setattr(
        answer_mod,
        "retrieve",
        lambda *a, **k: RetrievalResult(chunks=[CHUNK], scores={CHUNK.id: 1.0}, reranked=False),
    )
    monkeypatch.setattr(answer_mod, "verify_claims", lambda claims, chunks: [
        type("V", (), {"text": c["text"], "citations": c["citations"], "label": "supported",
                       "reason": "", "citation_labels": {}})() for c in claims
    ])


def test_refusal_from_primary_model_falls_back_to_a_stronger_model(monkeypatch):
    _patch_retrieval(monkeypatch)
    monkeypatch.setattr(answer_mod, "FALLBACK_LLM_MODEL", "big-stock-model")
    calls = []

    def fake_chat(messages, **kw):
        calls.append(kw.get("model"))
        return ChatResult(content=_REFUSAL if kw.get("model") is None else _cited_answer(), finish_reason="stop")

    monkeypatch.setattr(answer_mod, "chat", fake_chat)
    result = answer_mod.answer_question("Q?")
    assert calls == [None, "big-stock-model"]
    assert result["generator"] == "fallback" and result["generator_model"] == "big-stock-model"
    assert result["claims"][0]["citations"] == [CHUNK.id]


def test_a_cited_primary_answer_never_triggers_the_fallback(monkeypatch):
    _patch_retrieval(monkeypatch)
    monkeypatch.setattr(answer_mod, "FALLBACK_LLM_MODEL", "big-stock-model")
    calls = []
    monkeypatch.setattr(
        answer_mod, "chat",
        lambda messages, **kw: calls.append(kw.get("model")) or ChatResult(content=_cited_answer(), finish_reason="stop"),
    )
    result = answer_mod.answer_question("Q?")
    assert calls == [None] and result["generator"] == "primary" and result["generator_model"] is None


def test_fallback_can_be_disabled_and_a_double_refusal_stays_a_refusal(monkeypatch):
    _patch_retrieval(monkeypatch)
    monkeypatch.setattr(answer_mod, "chat", lambda messages, **kw: ChatResult(content=_REFUSAL, finish_reason="stop"))
    monkeypatch.setattr(answer_mod, "FALLBACK_LLM_MODEL", "")
    assert answer_mod.answer_question("Q?")["generator"] == "primary"
    monkeypatch.setattr(answer_mod, "FALLBACK_LLM_MODEL", "big-stock-model")
    result = answer_mod.answer_question("Q?")
    assert result["generator"] == "primary" and not result["claims"][0]["citations"]


def _enable_relevance(monkeypatch, verdicts):
    """verdicts: list of (bool|None, reason) returned by successive relevance checks."""
    monkeypatch.setattr(answer_mod, "RELEVANCE_CHECK", True)
    monkeypatch.setattr(answer_mod, "FALLBACK_LLM_MODEL", "big-stock-model")
    seen = []

    def fake_check(question, statements):
        seen.append(statements)
        return verdicts[len(seen) - 1]

    monkeypatch.setattr(answer_mod, "check_relevance", fake_check)
    return seen


def test_cited_but_off_topic_primary_answer_triggers_the_fallback_with_extra_guidance(monkeypatch):
    _patch_retrieval(monkeypatch)
    _enable_relevance(monkeypatch, [(False, "it covers a different point"), (True, "direct")])
    systems = []

    def fake_chat(messages, **kw):
        systems.append((kw.get("model"), messages[0]["content"]))
        return ChatResult(content=_cited_answer(), finish_reason="stop")

    monkeypatch.setattr(answer_mod, "chat", fake_chat)
    result = answer_mod.answer_question("Q?")
    assert [m for m, _ in systems] == [None, "big-stock-model"]
    assert "state the general rule FIRST" not in systems[0][1]       # primary prompt untouched
    assert "state the general rule FIRST" in systems[1][1]           # fallback gets the guidance
    assert result["generator"] == "fallback" and result["answers_question"] is True


def test_on_topic_primary_answer_keeps_the_primary_and_reports_the_verdict(monkeypatch):
    _patch_retrieval(monkeypatch)
    _enable_relevance(monkeypatch, [(True, "direct")])
    calls = []
    monkeypatch.setattr(
        answer_mod, "chat",
        lambda messages, **kw: calls.append(kw.get("model")) or ChatResult(content=_cited_answer(), finish_reason="stop"),
    )
    result = answer_mod.answer_question("Q?")
    assert calls == [None] and result["generator"] == "primary" and result["answers_question"] is True


def test_an_unavailable_relevance_check_never_triggers_the_fallback(monkeypatch):
    _patch_retrieval(monkeypatch)
    _enable_relevance(monkeypatch, [(None, "relevance check unavailable")])
    calls = []
    monkeypatch.setattr(
        answer_mod, "chat",
        lambda messages, **kw: calls.append(kw.get("model")) or ChatResult(content=_cited_answer(), finish_reason="stop"),
    )
    result = answer_mod.answer_question("Q?")
    assert calls == [None] and result["answers_question"] is None


def test_a_still_off_topic_fallback_answer_is_flagged_not_hidden(monkeypatch):
    _patch_retrieval(monkeypatch)
    _enable_relevance(monkeypatch, [(False, "a"), (False, "still a different point")])
    monkeypatch.setattr(answer_mod, "chat", lambda messages, **kw: ChatResult(content=_cited_answer(), finish_reason="stop"))
    result = answer_mod.answer_question("Q?")
    assert result["generator"] == "fallback" and result["answers_question"] is False
    assert result["relevance_reason"] == "still a different point"


def _capture_retrieve(monkeypatch, instruments):
    """Patch the indices/retrieval and record the restrict_celex the answer flow chooses."""
    from lexeuropa.retrieval.reference_resolver import ReferenceResolver

    def mk(celex):
        return Chunk(id=f"{celex}:en:art_1:001", celex=celex, lang="en", instrument_title="t", citation="c", unit_type="article_paragraph",
                     article="1", para_index="1", text="x", source_url="")

    resolver = ReferenceResolver([mk(c) for c in instruments])
    monkeypatch.setattr(answer_mod, "_load_indices", lambda: (None, None, resolver))
    seen = {}

    def fake_retrieve(*a, **k):
        seen.update(k)
        return RetrievalResult(chunks=[CHUNK], scores={CHUNK.id: 1.0}, reranked=False)

    monkeypatch.setattr(answer_mod, "retrieve", fake_retrieve)
    monkeypatch.setattr(answer_mod, "chat", lambda *a, **k: ChatResult(content="not json", finish_reason="stop"))
    monkeypatch.setattr(answer_mod, "FALLBACK_LLM_MODEL", "")
    return seen


def test_a_question_naming_exactly_one_instrument_is_restricted_to_it(monkeypatch):
    seen = _capture_retrieve(monkeypatch, ["32016R0679", "32018R1725"])
    answer_mod.answer_question("When must a personal data breach be reported under the GDPR?")
    assert seen["restrict_celex"] == "32016R0679"


def test_a_question_naming_two_instruments_or_none_is_not_restricted(monkeypatch):
    seen = _capture_retrieve(monkeypatch, ["32016R0679", "32022R2065"])
    answer_mod.answer_question("How do the GDPR and the DSA treat notices?")
    assert seen["restrict_celex"] is None
    answer_mod.answer_question("When must a breach be reported?")
    assert seen["restrict_celex"] is None


def test_the_law_menu_choice_restricts_even_when_the_question_names_nothing(monkeypatch):
    seen = _capture_retrieve(monkeypatch, ["32016R0679", "32018R1725"])
    answer_mod.answer_question("When must a breach be reported?", default_instrument="eudpr")
    assert seen["restrict_celex"] == "32018R1725"
