import pytest

from lexeuropa.generation.prompts import ParseError, build_messages, parse_response
from lexeuropa.schema import Chunk

CHUNK_A = Chunk(
    id="32016R0679:en:art_17:001",
    celex="32016R0679",
    lang="en",
    instrument_title="GDPR",
    citation="Article 17(1) GDPR",
    unit_type="article_paragraph",
    article="17",
    para_index="1",
    text="erasure text",
    source_url="http://publications.europa.eu/resource/celex/32016R0679",
)
CHUNK_B = Chunk(
    id="32016R0679:en:art_6:001",
    celex="32016R0679",
    lang="en",
    instrument_title="GDPR",
    citation="Article 6(1) GDPR",
    unit_type="article_paragraph",
    article="6",
    para_index="1",
    text="lawfulness text",
    source_url="http://publications.europa.eu/resource/celex/32016R0679",
)


def test_build_messages_assigns_short_handles_and_returns_the_mapping():
    messages, handle_to_id = build_messages("q?", [CHUNK_A, CHUNK_B])
    assert handle_to_id == {"S1": CHUNK_A.id, "S2": CHUNK_B.id}
    # the long chunk id should not appear in what the model is asked to copy
    user_content = messages[1]["content"]
    assert "S1" in user_content and "S2" in user_content
    assert CHUNK_A.id not in user_content


def test_parses_well_formed_claims():
    raw = '{"language": "en", "claims": [{"text": "A", "citations": ["x:1"]}]}'
    data = parse_response(raw)
    assert data["claims"][0]["citations"] == ["x:1"]


def test_normalizes_bare_string_citation_to_a_list():
    # Smaller models sometimes emit a single citation as a bare string
    # instead of a one-element list — this must not be iterated character
    # by character downstream (which would turn a valid citation into a
    # pile of fabricated one-letter ones).
    raw = '{"language": "en", "claims": [{"text": "A", "citations": "32016R0679:en:art_17:001"}]}'
    data = parse_response(raw)
    assert data["claims"][0]["citations"] == ["32016R0679:en:art_17:001"]


def test_rejects_non_string_citations():
    raw = '{"language": "en", "claims": [{"text": "A", "citations": [1, 2]}]}'
    with pytest.raises(ParseError):
        parse_response(raw)


def test_strips_markdown_fences():
    raw = '```json\n{"language": "en", "claims": []}\n```'
    data = parse_response(raw)
    assert data["claims"] == []


def test_rejects_missing_claims_key():
    with pytest.raises(ParseError):
        parse_response('{"language": "en"}')


# Recorded live output from qwen2.5:3b-instruct (trimmed for length but with
# the exact original ending intact): every inner list/object closes
# correctly, but the model stopped one token before the outermost `}`.
_RECORDED_MISSING_FINAL_BRACE = (
    '{"language": "en", "claims": [{"text": "The CJEU characterized the obligations '
    'of a search engine operator.", "citations": ["S3", "S1"]}]'
)


def test_repairs_a_missing_final_closing_brace():
    data = parse_response(_RECORDED_MISSING_FINAL_BRACE)
    assert data["claims"][0]["citations"] == ["S3", "S1"]


def test_does_not_repair_when_repair_is_disabled():
    # answer.py disables repair on a real finish_reason=length cutoff,
    # since that can truncate mid-string rather than just missing a brace.
    with pytest.raises(ParseError):
        parse_response(_RECORDED_MISSING_FINAL_BRACE, allow_repair=False)


def test_does_not_repair_text_truncated_inside_an_open_string():
    truncated_mid_string = '{"language": "en", "claims": [{"text": "cut off mid'
    with pytest.raises(ParseError):
        parse_response(truncated_mid_string)
