"""The local web interface: routing, input validation and the loopback-only guards.

The real answer pipeline is mocked out; these tests only exercise the HTTP layer.
"""
import http.client
import json
import threading

import pytest

import lexeuropa.answer as answer_mod
import lexeuropa.web as web


@pytest.fixture()
def server(monkeypatch):
    web.Handler.info_cache = None
    monkeypatch.setattr(
        web, "build_info",
        lambda: {"instruments": [{"celex": "32016R0679", "title": "GDPR", "alias": "gdpr", "doc_type": "regulation",
                                  "date": "2016-04-27", "case_number": None, "chunks": 3}], "chunks": 3},
    )
    srv = web.make_server(0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, resp, data


def _post_json(port, payload, host=None, content_type="application/json"):
    body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
    headers = {"Content-Type": content_type}
    if host:
        headers["Host"] = host
    return _request(port, "POST", "/api/ask", body=body, headers=headers)


def test_serves_the_page_with_a_restrictive_content_security_policy(server):
    status, resp, data = _request(server, "GET", "/")
    assert status == 200 and b"<title>LexEuropa</title>" in data
    csp = resp.getheader("Content-Security-Policy")
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp
    assert resp.getheader("Cache-Control") == "no-store"


def test_info_endpoint_lists_the_corpus(server):
    status, _, data = _request(server, "GET", "/api/info")
    info = json.loads(data)
    assert status == 200 and info["chunks"] == 3 and info["instruments"][0]["alias"] == "gdpr"


def test_requests_with_a_non_loopback_host_header_are_rejected(server):
    status, _, _ = _request(server, "GET", "/", headers={"Host": "evil.example.com"})
    assert status == 403
    status, _, _ = _post_json(server, {"question": "What is Article 17?"}, host="evil.example.com:8765")
    assert status == 403


@pytest.mark.parametrize("host,ok", [
    ("localhost", True), ("localhost:8765", True), ("127.0.0.1:8765", True), ("[::1]:8765", True),
    ("LOCALHOST", True), ("127.0.0.1.evil.com", False), ("evil.com:80", False), ("", False), (None, False),
])
def test_host_is_loopback(host, ok):
    assert web.host_is_loopback(host) is ok


def test_ask_requires_a_json_content_type(server):
    status, _, _ = _post_json(server, {"question": "What is Article 17?"}, content_type="text/plain")
    assert status == 415


def test_ask_rejects_oversized_and_malformed_bodies(server):
    status, _, _ = _post_json(server, "x" * (web.MAX_BODY_BYTES + 1))
    assert status == 413
    status, _, _ = _post_json(server, "{not json")
    assert status == 400


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "ab"}, {"question": 5}, {"question": "x" * (web.MAX_QUESTION_CHARS + 1)},
    {"question": "What is Article 17?", "instrument": "not-a-law"}, ["list"],
])
def test_ask_validates_the_payload(server, payload):
    status, _, data = _post_json(server, payload)
    assert status == 400 and json.loads(data)["error"]


def test_ask_returns_the_answer_and_passes_the_question_and_instrument_through(server, monkeypatch):
    seen = {}

    def fake_answer(question, *, default_instrument=None, **kw):
        seen.update(question=question, instrument=default_instrument)
        return {"question": question, "claims": [], "retrieved": [], "disclaimer": "d"}

    monkeypatch.setattr(answer_mod, "answer_question", fake_answer)
    status, _, data = _post_json(server, {"question": "  What does Article 17 say?  ", "instrument": "gdpr"})
    body = json.loads(data)
    assert status == 200 and "elapsed_s" in body
    assert seen == {"question": "What does Article 17 say?", "instrument": "gdpr"}


def test_ask_turns_a_pipeline_failure_into_a_json_error_not_a_dead_connection(server, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("endpoint exploded")

    monkeypatch.setattr(answer_mod, "answer_question", boom)
    status, _, data = _post_json(server, {"question": "What is Article 17?"})
    assert status == 500 and "endpoint exploded" in json.loads(data)["error"]


def test_the_server_only_binds_to_loopback(server):
    srv = web.make_server(0)
    try:
        assert srv.server_address[0] == "127.0.0.1"
    finally:
        srv.server_close()


def test_unknown_routes_are_404(server):
    assert _request(server, "GET", "/etc/passwd")[0] == 404
    assert _request(server, "POST", "/api/other", body="{}", headers={"Content-Type": "application/json"})[0] == 404
