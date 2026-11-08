from lexeuropa.generation.client import ChatResult
from lexeuropa.schema import Chunk
from lexeuropa.verification.verifier import default_llm_judge, verify_claim, verify_claims

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


C136_CHUNK = Chunk(
    id="62017CJ0136:en:pt_19",
    celex="62017CJ0136",
    lang="en",
    instrument_title="GC and Others v CNIL",
    citation="C-136/17, para. 19",
    unit_type="judgment_paragraph",
    article=None,
    para_index="19",
    text="Article 17 of Regulation 2016/679 reads as follows: ...",
    source_url="http://publications.europa.eu/resource/celex/62017CJ0136",
)


def _stub_judge_always(label, reason="stub reason"):
    def judge(claim_text, cited_chunks):
        return label, reason

    return judge


def test_claim_with_no_citations_is_uncertain():
    claim = {"text": "Some unsupported statement.", "citations": []}
    result = verify_claim(claim, {CHUNK.id: CHUNK})
    assert result.label == "uncertain"


def test_claim_citing_a_chunk_not_in_retrieved_set_is_unsupported():
    claim = {"text": "Article 99 says something.", "citations": ["nonexistent:chunk:id"]}
    result = verify_claim(claim, {CHUNK.id: CHUNK}, judge_fn=_stub_judge_always("supported"))
    assert result.label == "unsupported"
    assert "nonexistent:chunk:id" in result.reason


def test_claim_with_partially_fabricated_citation_is_unsupported():
    claim = {
        "text": "Right to erasure applies broadly.",
        "citations": [CHUNK.id, "made-up:citation"],
    }
    result = verify_claim(claim, {CHUNK.id: CHUNK}, judge_fn=_stub_judge_always("supported"))
    assert result.label == "unsupported"


def test_claim_with_valid_citation_defers_to_judge():
    claim = {
        "text": "Data subjects can request erasure of their personal data.",
        "citations": [CHUNK.id],
    }
    result = verify_claim(claim, {CHUNK.id: CHUNK}, judge_fn=_stub_judge_always("supported"))
    assert result.label == "supported"


def test_judge_always_calls_its_own_configured_model_and_endpoint(monkeypatch):
    """Guards against a real failure mode: if LLM_MODEL/LLM_BASE_URL ever
    point at a fine-tuned adapter, the judge must not follow — it has its
    own JUDGE_MODEL/JUDGE_BASE_URL regardless."""
    import lexeuropa.verification.verifier as verifier_mod

    seen = {}

    def _fake_chat(messages, *, temperature=0.1, model=None, base_url=None, **kwargs):
        seen["model"] = model
        seen["base_url"] = base_url
        return ChatResult(content='{"label": "supported", "reason": "ok"}', finish_reason="stop")

    monkeypatch.setattr(verifier_mod, "chat", _fake_chat)
    # Simulate the generator pointing at a fine-tuned adapter elsewhere.
    monkeypatch.setattr(verifier_mod, "JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct:featherless-ai")
    monkeypatch.setattr(verifier_mod, "JUDGE_BASE_URL", "https://router.huggingface.co/v1")

    default_llm_judge("some claim", [CHUNK])

    assert seen["model"] == "Qwen/Qwen2.5-7B-Instruct:featherless-ai"
    assert seen["base_url"] == "https://router.huggingface.co/v1"


def test_claim_naming_one_case_but_citing_a_different_one_is_unsupported():
    """Reproduces the exact live bug: a claim about C-131/12 citing only
    C-136/17 paragraphs (even though C-136/17 happens to quote C-131/12's
    holding verbatim, so the text itself is faithful). The judge is never
    even called here — this is a deterministic source-identity check that
    must catch it before a text-only judge call could be fooled by the
    faithful-but-wrong-case text."""

    def _judge_that_would_wrongly_approve(claim_text, cited_chunks):
        return "supported", "the text does match"  # what the real judge did live

    claim = {
        "text": "In C-131/12, the Court characterized the obligations of a search engine operator.",
        "citations": [C136_CHUNK.id],
    }
    result = verify_claim(
        claim,
        {C136_CHUNK.id: C136_CHUNK},
        judge_fn=_judge_that_would_wrongly_approve,
    )
    assert result.label == "unsupported"
    assert "62017CJ0136" in result.reason


def test_claim_naming_the_same_case_it_cites_still_goes_to_the_judge():
    claim = {
        "text": "In C-136/17, the Court discussed Article 17.",
        "citations": [C136_CHUNK.id],
    }
    result = verify_claim(claim, {C136_CHUNK.id: C136_CHUNK}, judge_fn=_stub_judge_always("supported"))
    assert result.label == "supported"


def test_claim_naming_no_case_still_goes_to_the_judge():
    claim = {"text": "Article 17 of Regulation 2016/679 reads as follows.", "citations": [C136_CHUNK.id]}
    result = verify_claim(claim, {C136_CHUNK.id: C136_CHUNK}, judge_fn=_stub_judge_always("supported"))
    assert result.label == "supported"


def test_irrelevant_co_citation_does_not_hide_behind_a_relevant_one():
    """Reproduces the documented limitation: a claim citing one clearly
    relevant chunk and one irrelevant one previously got a single joint
    judge call, so the relevant chunk alone could make the whole set
    "supported" with the irrelevant co-citation never independently
    checked. Each citation must now be judged on its own and show up
    separately in citation_labels, even though the claim-level label
    still reflects the best of the two."""

    def judge_per_chunk(claim_text, cited_chunks):
        (chunk,) = cited_chunks
        if chunk.id == CHUNK.id:
            return "supported", "directly on point"
        return "unsupported", "unrelated to the claim"

    claim = {
        "text": "Data subjects can request erasure of their personal data.",
        "citations": [CHUNK.id, C136_CHUNK.id],
    }
    result = verify_claim(
        claim, {CHUNK.id: CHUNK, C136_CHUNK.id: C136_CHUNK}, judge_fn=judge_per_chunk
    )
    assert result.label == "supported"
    assert result.citation_labels[CHUNK.id] == ("supported", "directly on point")
    assert result.citation_labels[C136_CHUNK.id] == ("unsupported", "unrelated to the claim")
    assert C136_CHUNK.id in result.reason


def test_verify_claims_runs_over_a_list():
    claims = [
        {"text": "A", "citations": []},
        {"text": "B", "citations": [CHUNK.id]},
    ]
    results = verify_claims(claims, [CHUNK], judge_fn=_stub_judge_always("interpretative"))
    assert results[0].label == "uncertain"
    assert results[1].label == "interpretative"


def test_claims_are_verified_concurrently_and_in_order():
    import threading
    import time

    from lexeuropa.schema import Chunk
    from lexeuropa.verification.verifier import verify_claims

    chunks = [
        Chunk(id=f"X:{i}", celex="X", lang="en", instrument_title="t", citation=f"c{i}", unit_type="article_paragraph",
              article=str(i), para_index="1", text=f"text {i}", source_url="")
        for i in range(6)
    ]
    claims = [{"text": f"claim {i}", "citations": [f"X:{i}"]} for i in range(6)]
    active = peak = 0
    lock = threading.Lock()

    def slow_judge(claim_text, cited):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.15)
        with lock:
            active -= 1
        return "supported", claim_text

    started = time.time()
    results = verify_claims(claims, chunks, slow_judge)
    assert [r.text for r in results] == [f"claim {i}" for i in range(6)]      # order preserved
    assert peak > 1 and time.time() - started < 0.6                            # not 6 x 0.15 s in series
