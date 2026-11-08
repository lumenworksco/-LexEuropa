"""A small local web interface: `python -m lexeuropa.cli serve`.

Standard library only (no web framework), so the base install stays light.
It serves one static page and two JSON endpoints and runs on this machine:

    GET  /            the chat-style page (lexeuropa/web_static/index.html)
    GET  /api/info    the corpus: instruments, chunk counts, selectable laws
    POST /api/ask     {"question": "...", "instrument": "gdpr"|null, "stream": bool}
                      -> answer_question(). With "stream": true the response is
                      newline-delimited JSON: {"stage": ...} events as the pipeline
                      advances, then {"result": {...}} or {"error": "..."}.

Safety: it binds to loopback only (there is no login, so it must never be
reachable from the network); requests whose Host header is not a loopback name
are rejected (DNS-rebinding); POSTs must be application/json (which browsers
only send cross-origin after a preflight that this server never grants);
request bodies and questions are size-limited; question text is never logged.
"""
from __future__ import annotations

import json
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lexeuropa.config import INSTRUMENT_ALIASES
from lexeuropa.ingest.pipeline import load_chunks, load_instruments

STATIC_DIR = Path(__file__).parent / "web_static"
MAX_BODY_BYTES = 16 * 1024
MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 2000
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
_HOST_RE = re.compile(r"^(?P<host>\[[0-9a-f:]+\]|[^:]+)(?::\d+)?$", re.IGNORECASE)

# Serialize generation: one question at a time keeps cost and endpoint load
# predictable for a single-user local tool.
_ask_lock = threading.Lock()


def host_is_loopback(host_header: str | None) -> bool:
    if not host_header:
        return False
    m = _HOST_RE.match(host_header.strip())
    return bool(m) and m.group("host").lower() in LOOPBACK_HOSTS


def _shortest_alias_by_celex() -> dict[str, str]:
    """celex -> the shortest alias that maps to it (e.g. 32016R0679 -> "gdpr")."""
    best: dict[str, str] = {}
    for alias, celex in INSTRUMENT_ALIASES.items():
        if celex not in best or len(alias) < len(best[celex]):
            best[celex] = alias
    return best


def build_info() -> dict:
    """What the corpus contains, for the page's coverage panel and law selector."""
    counts: dict[str, int] = {}
    for chunk in load_chunks():
        counts[chunk.celex] = counts.get(chunk.celex, 0) + 1
    alias_by_celex = _shortest_alias_by_celex()
    instruments = [
        {
            "celex": i.celex,
            "title": i.title,
            "date": i.date,
            "doc_type": i.doc_type,
            "case_number": i.case_number,
            "chunks": counts.get(i.celex, 0),
            "alias": alias_by_celex.get(i.celex),
        }
        for i in load_instruments()
    ]
    return {"instruments": instruments, "chunks": sum(counts.values())}


def validate_ask_payload(payload: object) -> tuple[str | None, str | None, str | None]:
    """(question, instrument, error). `instrument` is an alias or None."""
    if not isinstance(payload, dict):
        return None, None, "request body must be a JSON object"
    question = payload.get("question")
    if not isinstance(question, str):
        return None, None, "'question' must be a string"
    question = question.strip()
    if len(question) < MIN_QUESTION_CHARS:
        return None, None, "please enter a question"
    if len(question) > MAX_QUESTION_CHARS:
        return None, None, f"question is too long (limit {MAX_QUESTION_CHARS} characters)"
    instrument = payload.get("instrument") or None
    if instrument is not None:
        if not isinstance(instrument, str) or instrument.lower() not in INSTRUMENT_ALIASES:
            return None, None, "unknown instrument"
    return question, instrument, None


class Handler(BaseHTTPRequestHandler):
    server_version = "LexEuropa"
    info_cache: dict | None = None

    def log_message(self, fmt: str, *args) -> None:  # never log questions
        pass

    # -- helpers ---------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: dict) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _guard_host(self) -> bool:
        if host_is_loopback(self.headers.get("Host")):
            return True
        self._json(403, {"error": "forbidden host"})
        return False

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        if not self._guard_host():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            page = (STATIC_DIR / "index.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
        elif path == "/api/info":
            try:
                if Handler.info_cache is None:
                    Handler.info_cache = build_info()
                self._json(200, Handler.info_cache)
            except FileNotFoundError:
                self._json(500, {"error": "no ingested corpus found - run `python -m lexeuropa.cli ingest` first"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._guard_host():
            return
        if self.path.split("?", 1)[0] != "/api/ask":
            self._json(404, {"error": "not found"})
            return
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            self._json(415, {"error": "content type must be application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
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

        from lexeuropa.answer import NotIngestedError, answer_question  # lazy: loads the indexes

        stream = payload.get("stream") is True
        started = time.time()

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def emit(event: dict) -> None:
                # A closed tab raises here, which also stops the pipeline from
                # spending on stages that nobody is waiting for.
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()

            try:
                with _ask_lock:
                    result = answer_question(
                        question, default_instrument=instrument, progress=lambda s: emit({"stage": s})
                    )
                result["elapsed_s"] = round(time.time() - started, 1)
                emit({"result": result})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except NotIngestedError as e:
                _try_emit(emit, {"error": str(e)})
            except Exception as e:
                _try_emit(emit, {"error": f"{type(e).__name__}: {e}"})
            return

        try:
            with _ask_lock:
                result = answer_question(question, default_instrument=instrument)
        except NotIngestedError as e:
            self._json(500, {"error": str(e)})
            return
        except Exception as e:  # surface it in the page instead of a dead connection
            self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return
        result["elapsed_s"] = round(time.time() - started, 1)
        self._json(200, result)


def _try_emit(emit, event: dict) -> None:
    try:
        emit(event)
    except (BrokenPipeError, ConnectionResetError):
        pass


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    """Bound to loopback only. Port 0 picks a free port (used by the tests)."""
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(port: int = 8765, open_browser: bool = True) -> int:
    server = make_server(port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"LexEuropa is running at {url}  (Ctrl+C to stop)")
    print("Questions are processed by models hosted on Hugging Face; this page is only reachable from this computer.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0
