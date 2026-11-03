"""answer_question's error paths, with the LLM and indices mocked out."""
import json

import lexeuropa.answer as answer_mod
from lexeuropa.generation.client import ChatResult
from lexeuropa.retrieval.pipeline import RetrievalResult
from lexeuropa.schema import Chunk

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


def test_truncated_generation_names_finish_reason_in_the_error(monkeypatch):
    monkeypatch.setattr(answer_mod, "_load_indices", lambda: (None, None, None))
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
    monkeypatch.setattr(answer_mod, "_load_indices", lambda: (None, None, None))
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

    monkeypatch.setattr(answer_mod, "_load_indices", lambda: (None, None, None))
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
