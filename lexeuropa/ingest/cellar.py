"""Fetch authoritative legal texts from the Publications Office CELLAR service.

EUR-Lex's own HTML pages sit behind a CloudFront WAF bot challenge (verified
by hand: a plain GET on eur-lex.europa.eu returns HTTP 202 with
`x-amzn-waf-action: challenge` and no body). CELLAR's content-negotiation
endpoint at publications.europa.eu resolves a CELEX number straight to the
underlying document via a 303 redirect and returns real XHTML, so we use
that instead of scraping EUR-Lex.
"""
from __future__ import annotations

import urllib.request
import urllib.error
from pathlib import Path

from lexeuropa.config import CELLAR_BASE, LANG_TO_CELLAR, RAW_DIR

USER_AGENT = "LexEuropa-research/0.1 (legal research prototype; contact via repository)"


class FetchError(RuntimeError):
    pass


def raw_path(celex: str, lang: str) -> Path:
    return Path(RAW_DIR) / celex / f"{lang}.xhtml"


def fetch(celex: str, lang: str, *, force: bool = False) -> str:
    """Return cached or freshly-fetched XHTML for a CELEX number in a language.

    `lang` is a two-letter code (only "en" is configured — see config.py).
    """
    if lang not in LANG_TO_CELLAR:
        raise ValueError(f"unsupported language code: {lang!r}")

    path = raw_path(celex, lang)
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")

    url = f"{CELLAR_BASE}{celex}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/xhtml+xml",
            "Accept-Language": LANG_TO_CELLAR[lang],
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        raise FetchError(
            f"CELLAR fetch failed for CELEX {celex} ({lang}): HTTP {e.code}"
        ) from e
    except urllib.error.URLError as e:
        raise FetchError(f"CELLAR fetch failed for CELEX {celex} ({lang}): {e}") from e

    if "xml" not in content_type and "html" not in content_type:
        raise FetchError(
            f"CELLAR returned unexpected content-type {content_type!r} for "
            f"CELEX {celex} ({lang}); this language may not exist for this "
            f"document, or the document may not be available as XHTML"
        )

    text = body.decode("utf-8")
    if not text.strip():
        raise FetchError(
            f"CELLAR returned an empty body for CELEX {celex} ({lang}) — this "
            f"usually means that language version does not exist for this document"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
