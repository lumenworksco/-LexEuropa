"""check_relevance parsing and failure handling, with the model call faked."""
from lexeuropa.generation.client import GenerationError
from lexeuropa.verification.relevance import check_relevance


def _fake(content):
    return lambda messages, **kw: (content, "stop")


def test_true_and_false_verdicts_are_parsed():
    assert check_relevance("Q?", ["A."], chat_fn=_fake('{"addresses_question": true, "reason": "direct"}')) == (True, "direct")
    assert check_relevance("Q?", ["A."], chat_fn=_fake('{"addresses_question": false, "reason": "other point"}')) == (False, "other point")


def test_fenced_and_chatty_output_is_still_parsed():
    out = 'Sure!\n```json\n{"addresses_question": false, "reason": "x"}\n```\nHope that helps {see above}'
    assert check_relevance("Q?", ["A."], chat_fn=_fake(out))[0] is False


def test_garbage_or_non_boolean_verdicts_mean_unknown_not_false():
    assert check_relevance("Q?", ["A."], chat_fn=_fake("no idea"))[0] is None
    assert check_relevance("Q?", ["A."], chat_fn=_fake('{"addresses_question": "yes"}'))[0] is None
    assert check_relevance("Q?", ["A."], chat_fn=_fake("{broken"))[0] is None


def test_a_failing_model_call_means_unknown_not_false():
    def boom(messages, **kw):
        raise GenerationError("HTTP 402")

    verdict, reason = check_relevance("Q?", ["A."], chat_fn=boom)
    assert verdict is None and "unavailable" in reason


def test_no_statements_means_unknown():
    assert check_relevance("Q?", [], chat_fn=_fake("{}"))[0] is None
