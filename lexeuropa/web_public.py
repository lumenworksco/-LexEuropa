"""The public demo server (the Hugging Face Space): the same interface as `serve`, but live for anyone.

It is a separate, opt-in mode (`python -m lexeuropa.cli serve-public`, which refuses to start unless
`LEXEUROPA_PUBLIC=1`). The local `serve` keeps its loopback bind and Host check untouched.

Who pays: the Space holds NO Hugging Face credentials of its own. A visitor signs in with Hugging Face
("Sign in with HF", scope `inference-api`) and every model call for their questions uses their token, so
their own free inference credit pays; see `auth_context.py` for how that is enforced.

The OAuth token never reaches the browser's JavaScript. After the code exchange it is kept in server memory,
keyed by a random session id in an HttpOnly cookie. Other safeguards: the sign-in `state` is bound to a
cookie; POSTs must be same-origin JSON; one question at a time per visitor, an hourly cap per visitor, a
global concurrency limit (excess requests get a plain 429, they do not queue); the token is scrubbed from
anything sent back; question text and tokens are never logged.

Flow: GET /login -> huggingface.co/oauth/authorize -> GET /login/callback?code&state -> session cookie.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer

from lexeuropa.auth_context import SignInRequired, bound_token
from lexeuropa.config import FALLBACK_LLM_MODEL
from lexeuropa.web import STATIC_DIR, Handler, host_is_loopback, validate_ask_payload

SESSION_COOKIE = "lex_sid"
STATE_COOKIE = "lex_state"
OAUTH_SCOPE = "openid profile inference-api"


@dataclass
class PublicConfig:
    client_id: str = ""
    client_secret: str = ""
    provider_url: str = "https://huggingface.co"
    origin: str = ""                      # e.g. https://ipwnds-lexeuropa-demo.hf.space (no trailing slash)
    session_ttl: int = 3600               # seconds
    max_questions_per_hour: int = 20      # per signed-in visitor
    max_concurrent: int = 3               # questions being answered at once, whole server
    max_sessions: int = 500
    allowed_origins: set = field(default_factory=set)

    @property
    def host(self) -> str:
        return urllib.parse.urlparse(self.origin).netloc

    @property
    def redirect_uri(self) -> str:
        return f"{self.origin}/login/callback"

    @classmethod
    def from_env(cls) -> "PublicConfig":
        space_host = os.environ.get("SPACE_HOST", "")
        origin = os.environ.get("LEXEUROPA_PUBLIC_ORIGIN") or (f"https://{space_host}" if space_host else "")
        return cls(
            client_id=os.environ.get("OAUTH_CLIENT_ID", ""),
            client_secret=os.environ.get("OAUTH_CLIENT_SECRET", ""),
            provider_url=os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co").rstrip("/"),
            origin=origin.rstrip("/"),
        )


class SessionStore:
    """In-memory sessions: sid -> {token, name, expires}. Lost on restart, which just means signing in again."""

    def __init__(self, ttl: int, max_sessions: int):
        self._ttl, self._max = ttl, max_sessions
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, token: str, name: str | None) -> str:
        sid = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._data = {k: v for k, v in self._data.items() if v["expires"] > now}
            while len(self._data) >= self._max:          # full: drop the oldest
                self._data.pop(min(self._data, key=lambda k: self._data[k]["expires"]))
            self._data[sid] = {"token": token, "name": name, "expires": now + self._ttl,
                               "asked": [], "busy": False}
        return sid

    def get(self, sid: str | None) -> dict | None:
        if not sid:
            return None
        with self._lock:
            s = self._data.get(sid)
            if s and s["expires"] <= time.time():
                del self._data[sid]
                return None
            return s

    def delete(self, sid: str | None) -> None:
        with self._lock:
            self._data.pop(sid or "", None)

    def try_begin(self, session: dict, per_hour: int) -> str | None:
        """Reserve the visitor's slot for one question. Returns an error code, or None when granted."""
        now = time.time()
        with self._lock:
            session["asked"] = [t for t in session["asked"] if now - t < 3600]
            if session["busy"]:
                return "busy_self"
            if len(session["asked"]) >= per_hour:
                return "rate_limited"
            session["busy"] = True
            session["asked"].append(now)
            return None

    def end(self, session: dict) -> None:
        with self._lock:
            session["busy"] = False


# --- OAuth calls (separate functions so tests can replace them) -------------------------------------------
def exchange_code(cfg: PublicConfig, code: str) -> dict:
    basic = base64.b64encode(f"{cfg.client_id}:{cfg.client_secret}".encode()).decode()
    body = urllib.parse.urlencode(
        {"client_id": cfg.client_id, "code": code, "grant_type": "authorization_code", "redirect_uri": cfg.redirect_uri}
    ).encode()
    req = urllib.request.Request(
        f"{cfg.provider_url}/oauth/token", data=body, method="POST",
        headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def fetch_username(cfg: PublicConfig, access_token: str) -> str | None:
    req = urllib.request.Request(f"{cfg.provider_url}/oauth/userinfo", headers={"Authorization": f"Bearer {access_token}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read()).get("preferred_username")
    except Exception:
        return None


# --- turning failures into advice -------------------------------------------------------------------------
_HTTP_CODE = re.compile(r"HTTP (\d{3})")


def friendly_error(message: str, token: str | None = None) -> str:
    """A visitor-facing explanation for a failed model call; never echoes the token or provider details."""
    if token:
        message = message.replace(token, "[redacted]")
    m = _HTTP_CODE.search(message)
    code = m.group(1) if m else None
    if code == "402":
        return ("Your Hugging Face account has no inference credit left for this month, so the model could not "
                "be called. Free accounts get a small monthly allowance; you can add credit or try again next month.")
    if code in ("401", "403"):
        return "Your Hugging Face sign-in no longer works for this request. Please sign out and sign in again."
    if code == "429":
        return "The model provider is busy or you hit its rate limit. Please wait a minute and try again."
    if code and code.startswith("5"):
        return "The model provider had a temporary problem. Please try again."
    return "The question could not be answered right now. Please try again."


def _scrub(text: str, token: str | None) -> str:
    return text.replace(token, "[redacted]") if token else text


class PublicHandler(Handler):
    server_version = "LexEuropaDemo"
    cfg: PublicConfig = PublicConfig()
    sessions: SessionStore = SessionStore(3600, 500)
    slots = threading.BoundedSemaphore(3)
    ready = threading.Event()

    # -- helpers ----------------------------------------------------------
    def _guard_host(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if host_is_loopback(host) or (self.cfg.host and host == self.cfg.host.lower()):
            return True
        self._json(403, {"error": "forbidden host"})
        return False

    def _cookie(self, name: str) -> str | None:
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == name:
                return v
        return None

    def _set_cookie(self, name: str, value: str, max_age: int) -> tuple:
        return ("Set-Cookie", f"{name}={value}; Max-Age={max_age}; Path=/; HttpOnly; Secure; SameSite=Lax")

    def _redirect(self, location: str, extra: tuple = ()) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True            # not a browser cross-site request (curl, tests, same-origin fetch in some browsers)
        return origin == self.cfg.origin or origin in self.cfg.allowed_origins

    def _session(self) -> dict | None:
        return self.sessions.get(self._cookie(SESSION_COOKIE))

    # -- GET ---------------------------------------------------------------
    def do_GET(self) -> None:
        if not self._guard_host():
            return
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/info":
            try:
                if Handler.info_cache is None:
                    from lexeuropa.web import build_info
                    Handler.info_cache = build_info()
                self._json(200, Handler.info_cache)
            except FileNotFoundError:
                self._json(500, {"error": "the corpus is not available on this server"})
        elif path == "/healthz":
            self._json(200, {"ok": True, "ready": self.ready.is_set()})
        elif path == "/auth/status":
            s = self._session()
            self._json(200, {"enabled": bool(self.cfg.client_id), "signed_in": bool(s), "name": s["name"] if s else None,
                             "model": FALLBACK_LLM_MODEL.split("/")[-1]})
        elif path == "/login":
            self._login()
        elif path == "/login/callback":
            self._callback(urllib.parse.parse_qs(query))
        else:
            self._json(404, {"error": "not found"})

    def _login(self) -> None:
        if not (self.cfg.client_id and self.cfg.origin):
            self._json(503, {"error": "sign-in is not configured on this server"})
            return
        state = secrets.token_urlsafe(24)
        params = urllib.parse.urlencode({
            "client_id": self.cfg.client_id, "redirect_uri": self.cfg.redirect_uri,
            "scope": OAUTH_SCOPE, "state": state, "response_type": "code",
        }, quote_via=urllib.parse.quote)
        self._redirect(f"{self.cfg.provider_url}/oauth/authorize?{params}", (self._set_cookie(STATE_COOKIE, state, 600),))

    def _callback(self, q: dict) -> None:
        state_cookie, state_param = self._cookie(STATE_COOKIE), (q.get("state") or [""])[0]
        code = (q.get("code") or [""])[0]
        clear_state = self._set_cookie(STATE_COOKIE, "", 0)
        if not code or not state_cookie or not hmac.compare_digest(state_cookie, state_param):
            self._json(400, {"error": "the sign-in could not be verified; please try again"}, (clear_state,))
            return
        try:
            token_data = exchange_code(self.cfg, code)
            access_token = token_data["access_token"]
        except Exception:
            self._json(502, {"error": "Hugging Face did not accept the sign-in; please try again"}, (clear_state,))
            return
        sid = self.sessions.create(access_token, fetch_username(self.cfg, access_token))
        self._redirect("/", (clear_state, self._set_cookie(SESSION_COOKIE, sid, self.cfg.session_ttl)))

    # -- POST --------------------------------------------------------------
    def do_POST(self) -> None:
        if not self._guard_host():
            return
        path = self.path.split("?", 1)[0]
        if path not in ("/api/ask", "/auth/logout"):
            self._json(404, {"error": "not found"})
            return
        if not self._same_origin():
            self._json(403, {"error": "cross-origin request refused"})
            return
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            self._json(415, {"error": "content type must be application/json"})
            return
        if path == "/auth/logout":
            self.sessions.delete(self._cookie(SESSION_COOKIE))
            self._json(200, {"ok": True}, (self._set_cookie(SESSION_COOKIE, "", 0),))
            return
        self._ask()

    def _ask(self) -> None:
        session = self._session()
        if not session:
            self._json(401, {"error": "sign_in_required"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > 16 * 1024:
            self._json(413, {"error": "request body too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"error": "request body is not valid JSON"})
            return
        question, instrument, error = validate_ask_payload(payload)
        if error:
            self._json(400, {"error": error})
            return
        if not self.ready.is_set():
            self._json(503, {"error": "the demo is still starting up (loading the library); try again in a minute"})
            return

        denied = self.sessions.try_begin(session, self.cfg.max_questions_per_hour)
        if denied == "busy_self":
            self._json(429, {"error": "busy", "message": "Your previous question is still being answered."})
            return
        if denied == "rate_limited":
            self._json(429, {"error": "rate_limited", "message": f"The demo allows {self.cfg.max_questions_per_hour} questions per hour per person."})
            return
        if not self.slots.acquire(blocking=False):
            self.sessions.end(session)
            self._json(429, {"error": "busy", "message": "The demo is answering other questions right now. Please try again in a minute."})
            return

        token = session["token"]
        try:
            self._answer_streaming(question, instrument, token)
        finally:
            self.slots.release()
            self.sessions.end(session)

    def _answer_streaming(self, question: str, instrument: str | None, token: str) -> None:
        from lexeuropa.answer import NotIngestedError, answer_question   # lazy: loads the indexes

        started = time.time()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(event: dict) -> None:
            line = _scrub(json.dumps(event, ensure_ascii=False), token)
            self.wfile.write((line + "\n").encode("utf-8"))
            self.wfile.flush()

        def safe_emit(event: dict) -> None:
            try:
                emit(event)
            except (BrokenPipeError, ConnectionResetError):
                pass

        try:
            with bound_token(token):
                result = answer_question(question, default_instrument=instrument, progress=lambda s: emit({"stage": s}))
            if result.get("error"):
                result["error"] = friendly_error(str(result["error"]), token)
            result["elapsed_s"] = round(time.time() - started, 1)
            emit({"result": result})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except SignInRequired:
            safe_emit({"error": "Please sign in with Hugging Face to ask a question."})
        except NotIngestedError:
            safe_emit({"error": "the corpus is not available on this server"})
        except Exception as e:
            safe_emit({"error": friendly_error(str(e), token)})


def make_public_server(cfg: PublicConfig, port: int = 7860, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """A server bound to `host` (0.0.0.0 inside the Space container; tests use 127.0.0.1 and port 0)."""
    handler = type("BoundPublicHandler", (PublicHandler,), {
        "cfg": cfg,
        "sessions": SessionStore(cfg.session_ttl, cfg.max_sessions),
        "slots": threading.BoundedSemaphore(cfg.max_concurrent),
        "ready": threading.Event(),
    })
    return ThreadingHTTPServer((host, port), handler)


def _warm_up(server: ThreadingHTTPServer) -> None:
    """Load the indexes and the local embedding model before the first visitor, then flip /healthz to ready."""
    try:
        from lexeuropa.answer import _load_indices
        from lexeuropa.retrieval.dense_index import embed_texts

        _load_indices()
        embed_texts(["warm up"])
        print("ready: indexes and embedding model loaded", flush=True)
    except Exception as e:   # surface it in the logs; /healthz stays not-ready
        print(f"warm-up failed: {type(e).__name__}: {e}", flush=True)
        return
    server.RequestHandlerClass.ready.set()


def serve_public(port: int = 7860) -> int:
    from lexeuropa.config import PUBLIC_MODE

    if not PUBLIC_MODE:
        raise SystemExit("refusing to start: set LEXEUROPA_PUBLIC=1 so that no owner credentials can be used")
    cfg = PublicConfig.from_env()
    if not (cfg.client_id and cfg.client_secret and cfg.origin):
        print("warning: OAUTH_CLIENT_ID / OAUTH_CLIENT_SECRET / SPACE_HOST are not all set; sign-in is disabled", flush=True)
    server = make_public_server(cfg, port=port)
    threading.Thread(target=_warm_up, args=(server,), daemon=True).start()
    print(f"LexEuropa public demo listening on :{port} (origin {cfg.origin or 'unset'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
