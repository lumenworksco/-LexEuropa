"""The public demo server: sign-in flow, sessions, per-visitor tokens, limits. A real server in a thread;
the Hugging Face OAuth calls and the answer pipeline are mocked."""
import http.client
import json
import threading
import urllib.parse

import pytest

import lexeuropa.answer as answer_mod
import lexeuropa.web_public as wp
from lexeuropa.auth_context import current_token

ORIGIN = "https://demo.hf.space"


@pytest.fixture
def server(monkeypatch):
    cfg = wp.PublicConfig(client_id="cid", client_secret="csecret", origin=ORIGIN, max_questions_per_hour=3, max_concurrent=2)
    srv = wp.make_public_server(cfg, port=0, host="127.0.0.1")
    srv.RequestHandlerClass.ready.set()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(wp, "exchange_code", lambda cfg, code: {"access_token": f"hf_oauth_{code}"})
    monkeypatch.setattr(wp, "fetch_username", lambda cfg, tok: "alice")
    yield srv
    srv.shutdown(); srv.server_close()


def call(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    h = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body); h.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=h)
    r = conn.getresponse(); raw = r.read(); conn.close()
    return r.status, r.headers, raw.decode("utf-8", "replace")


def sign_in(srv, code="abc"):
    """Run the whole sign-in flow; returns the Cookie header value for the new session."""
    _, hdr, _ = call(srv, "GET", "/login")
    state = urllib.parse.parse_qs(urllib.parse.urlparse(hdr["Location"]).query)["state"][0]
    state_cookie = hdr["Set-Cookie"].split(";")[0]
    status, hdr2, _ = call(srv, "GET", f"/login/callback?code={code}&state={state}", headers={"Cookie": state_cookie})
    assert status == 302
    sid = [v for k, v in hdr2.items() if k == "Set-Cookie" and v.startswith("lex_sid=")][0].split(";")[0]
    return sid


def ask(srv, cookie, q="What is Article 17 GDPR?", **extra):
    return call(srv, "POST", "/api/ask", {"question": q}, {"Cookie": cookie, **extra})


def events(raw):
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


def test_pages_and_host_check(server):
    status, _, html = call(server, "GET", "/")
    assert status == 200 and "<html" in html
    assert call(server, "GET", "/", headers={"Host": "evil.example"})[0] == 403
    assert call(server, "GET", "/healthz")[0] == 200
    st = json.loads(call(server, "GET", "/auth/status")[2])
    assert st["enabled"] is True and st["signed_in"] is False and st["name"] is None and st["model"]


def test_login_redirects_to_hugging_face_with_the_inference_scope_and_a_bound_state(server):
    status, hdr, _ = call(server, "GET", "/login")
    assert status == 302
    url = urllib.parse.urlparse(hdr["Location"])
    q = urllib.parse.parse_qs(url.query)
    assert url.netloc == "huggingface.co" and url.path == "/oauth/authorize"
    assert q["client_id"] == ["cid"] and q["redirect_uri"] == [f"{ORIGIN}/login/callback"] and q["response_type"] == ["code"]
    assert "inference-api" in q["scope"][0].split()
    cookie = hdr["Set-Cookie"]
    assert cookie.startswith(f"lex_state={q['state'][0]};") and "HttpOnly" in cookie and "Secure" in cookie and "SameSite=Lax" in cookie


def test_callback_rejects_a_missing_or_mismatched_state(server):
    assert call(server, "GET", "/login/callback?code=x&state=nope")[0] == 400                      # no state cookie
    assert call(server, "GET", "/login/callback?code=x&state=nope", headers={"Cookie": "lex_state=other"})[0] == 400
    assert call(server, "GET", "/login/callback?state=s", headers={"Cookie": "lex_state=s"})[0] == 400   # no code


def test_sign_in_creates_an_httponly_session_and_never_exposes_the_token(server):
    _, hdr, _ = call(server, "GET", "/login")
    state = urllib.parse.parse_qs(urllib.parse.urlparse(hdr["Location"]).query)["state"][0]
    status, hdr2, body = call(server, "GET", f"/login/callback?code=thecode&state={state}", headers={"Cookie": f"lex_state={state}"})
    assert status == 302 and hdr2["Location"] == "/"
    sid_header = [v for k, v in hdr2.items() if k == "Set-Cookie" and v.startswith("lex_sid=")][0]
    assert "HttpOnly" in sid_header and "Secure" in sid_header and "SameSite=Lax" in sid_header
    everything = body + "".join(f"{k}: {v}" for k, v in hdr2.items())
    assert "hf_oauth_thecode" not in everything
    cookie = sid_header.split(";")[0]
    status_body = call(server, "GET", "/auth/status", headers={"Cookie": cookie})[2]
    st = json.loads(status_body)
    assert st["signed_in"] is True and st["name"] == "alice" and st["model"]
    assert "hf_oauth_thecode" not in status_body


def test_a_failed_token_exchange_is_a_plain_error(server, monkeypatch):
    def boom(cfg, code): raise RuntimeError("secret detail csecret")
    monkeypatch.setattr(wp, "exchange_code", boom)
    _, hdr, _ = call(server, "GET", "/login")
    state = urllib.parse.parse_qs(urllib.parse.urlparse(hdr["Location"]).query)["state"][0]
    status, _, body = call(server, "GET", f"/login/callback?code=c&state={state}", headers={"Cookie": f"lex_state={state}"})
    assert status == 502 and "csecret" not in body


def test_asking_requires_a_session_a_same_origin_json_post(server):
    assert ask(server, "")[0] == 401 and "sign_in_required" in ask(server, "")[2]
    cookie = sign_in(server)
    assert ask(server, cookie, Origin="https://evil.example")[0] == 403
    assert call(server, "POST", "/api/ask", headers={"Cookie": cookie, "Content-Type": "text/plain"})[0] == 415
    assert ask(server, cookie, q="x")[0] == 400                      # shorter than the 3-character minimum


def test_the_visitors_own_token_pays_for_their_question_and_is_scrubbed_from_the_reply(server, monkeypatch):
    seen = {}
    def fake(question, *, default_instrument=None, progress=None, **kw):
        seen["token"] = current_token()
        progress("retrieving")
        return {"question": question, "claims": [], "error": None, "note": f"echo {current_token()}"}
    monkeypatch.setattr(answer_mod, "answer_question", fake)
    cookie = sign_in(server, "xyz")
    status, _, raw = ask(server, cookie)
    assert status == 200 and seen["token"] == "hf_oauth_xyz"
    evs = events(raw)
    assert evs[0] == {"stage": "retrieving"} and "result" in evs[-1]
    assert "hf_oauth_xyz" not in raw and "[redacted]" in raw          # the token never goes back to the browser


def test_two_visitors_at_once_never_share_a_token(server, monkeypatch):
    tokens, barrier = {}, threading.Barrier(2)
    def fake(question, *, default_instrument=None, progress=None, **kw):
        barrier.wait(timeout=5)                                       # both requests are inside answer_question together
        tokens[question] = current_token()
        return {"question": question, "claims": []}
    monkeypatch.setattr(answer_mod, "answer_question", fake)
    a, b = sign_in(server, "A"), sign_in(server, "B")
    out = {}
    ts = [threading.Thread(target=lambda c=c, q=q: out.__setitem__(q, ask(server, c, q=q)[0])) for c, q in ((a, "question for A"), (b, "question for B"))]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert out == {"question for A": 200, "question for B": 200}
    assert tokens == {"question for A": "hf_oauth_A", "question for B": "hf_oauth_B"}


def test_limits_per_visitor_hourly_one_at_a_time_and_global(server, monkeypatch):
    release = threading.Event(); inside = threading.Semaphore(0)
    def slow(question, *, default_instrument=None, progress=None, **kw):
        inside.release(); release.wait(timeout=10)                    # signal "I hold a slot", then wait to be released
        return {"question": question, "claims": []}
    monkeypatch.setattr(answer_mod, "answer_question", slow)
    a, b, c = sign_in(server, "A"), sign_in(server, "B"), sign_in(server, "C")
    ta = threading.Thread(target=lambda: ask(server, a)); ta.start()
    assert inside.acquire(timeout=5)
    status, _, body = ask(server, a)                                   # same visitor while their question is running
    assert status == 429 and "previous question is still being answered" in body
    tb = threading.Thread(target=lambda: ask(server, b)); tb.start()
    assert inside.acquire(timeout=5)                                   # now both slots (max_concurrent=2) are occupied
    status, _, body = ask(server, c)                                   # a third visitor: the server is full
    assert status == 429 and "other questions" in body
    release.set(); ta.join(); tb.join()
    # Hourly cap is 3 per visitor. A has been granted 1 (a refused request does not count), so two more fit.
    results = [ask(server, a, q=f"question number {i}")[0] for i in range(4)]
    assert results == [200, 200, 429, 429]


def test_not_ready_gives_503(server):
    server.RequestHandlerClass.ready.clear()
    cookie = sign_in(server)
    assert ask(server, cookie)[0] == 503


def test_logout_ends_the_session(server):
    cookie = sign_in(server)
    status, hdr, _ = call(server, "POST", "/auth/logout", {}, {"Cookie": cookie})
    assert status == 200 and "Max-Age=0" in hdr["Set-Cookie"]
    assert json.loads(call(server, "GET", "/auth/status", headers={"Cookie": cookie})[2])["signed_in"] is False


def test_errors_become_advice_without_provider_details_or_the_token():
    msg = "generation request to https://router.huggingface.co/v1 failed: HTTP 402: {'error': 'Payment Required hf_tok'} "
    out = wp.friendly_error(msg, "hf_tok")
    assert "no inference credit" in out and "hf_tok" not in out and "router.huggingface.co" not in out
    assert "sign in again" in wp.friendly_error("HTTP 401: Unauthorized")
    assert "try again" in wp.friendly_error("something odd")


def test_serve_public_refuses_to_start_without_the_public_flag(monkeypatch):
    import lexeuropa.config as config
    monkeypatch.setattr(config, "PUBLIC_MODE", False)
    with pytest.raises(SystemExit) as e:
        wp.serve_public(port=0)
    assert "LEXEUROPA_PUBLIC=1" in str(e.value)
