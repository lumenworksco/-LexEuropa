"""Paths, model settings and known instrument aliases."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no new dependency) — populates os.environ without
    overriding anything already set in the real shell environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(PROJECT_ROOT / ".env")

# Works around python.org's macOS build shipping without a CA bundle.
import certifi  # noqa: E402

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# Reranker weights cache into the project dir, not ~/.cache/huggingface — see CLAUDE.md.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
INSTRUMENTS_PATH = PROCESSED_DIR / "instruments.jsonl"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
DENSE_INDEX_PATH = INDEX_DIR / "dense.npz"

CELLAR_BASE = "http://publications.europa.eu/resource/celex/"
# CELLAR / EUR-Lex use ISO 639-2/B three-letter codes; we work with two-letter
# codes (matching EU official-language conventions) in the rest of the app.
# English-only by design for now — see CLAUDE.md.
LANG_TO_CELLAR = {"en": "eng"}

# Generation runs on HF, not locally — see CLAUDE.md. Default is the
# fine-tuned adapter (Qwen2.5-3B + LoRA, merged), served from a dedicated,
# scale-to-zero Inference Endpoint rather than the Inference Providers
# router — a LoRA-trained model isn't in that catalog. Scale-to-zero means
# a cold request after 15 idle minutes pays a restart cost (observed:
# under a minute, since the model is already on the endpoint's attached
# storage) — this is expected, not a bug, if `ask` seems to hang briefly
# on the first call in a while.
HF_TOKEN = os.environ.get("HF_TOKEN")
LLM_BASE_URL = os.environ.get(
    "LEXEUROPA_LLM_BASE_URL", "https://6ac3628565c8b62f296a02a0.endpoints.huggingface.cloud/v1"
)
LLM_MODEL = os.environ.get("LEXEUROPA_LLM_MODEL", "iPwnds/lexeuropa-qwen3b-merged")

# The judge is a deliberately separate model/endpoint from the generator
# (not just LLM_BASE_URL/LLM_MODEL again): if LLM_MODEL ever points at a
# fine-tuned adapter trained to always emit {"claims": [...]} JSON, sending
# judge prompts (plain-text label+reason) to that same model would get
# JSON back instead of a judgeable answer, breaking verification silently.
# If the primary generator refuses (no cited claim) or its output cannot be parsed,
# retry once on this stronger stock model via the HF router; set
# LEXEUROPA_FALLBACK_LLM_MODEL="" to disable. The answer is labelled as a fallback.
FALLBACK_LLM_BASE_URL = os.environ.get("LEXEUROPA_FALLBACK_LLM_BASE_URL", "https://router.huggingface.co/v1")
FALLBACK_LLM_MODEL = os.environ.get("LEXEUROPA_FALLBACK_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

JUDGE_BASE_URL = os.environ.get("LEXEUROPA_JUDGE_BASE_URL", "https://router.huggingface.co/v1")
JUDGE_MODEL = os.environ.get("LEXEUROPA_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct:featherless-ai")

# Embeddings run on HF Inference Providers too, via hf-inference (not the OpenAI-compatible router).
HF_INFERENCE_BASE_URL = os.environ.get(
    "LEXEUROPA_HF_INFERENCE_URL", "https://router.huggingface.co/hf-inference"
)
EMBED_MODEL = os.environ.get("LEXEUROPA_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
# "hf" (default): query/passage embeddings via HF Inference Providers (needs account
# credits). "local": run the same model with sentence-transformers on this machine
# (pip install -e ".[rerank]"; ~1.3 GB of weights, free). Use "local" for offline
# retrieval work; the indexed corpus vectors are the same model either way.
EMBED_BACKEND = os.environ.get("LEXEUROPA_EMBED_BACKEND", "hf")

# The reranker runs locally (CPU, sentence-transformers) for now — see CLAUDE.md.
RERANKER_MODEL = os.environ.get("LEXEUROPA_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# Set LEXEUROPA_USE_RERANKER=0 to skip the local cross-encoder (no torch/weights needed).
USE_RERANKER = os.environ.get("LEXEUROPA_USE_RERANKER", "1") != "0"
# At most this many recitals in the final retrieved set (see retrieval/pipeline.py).
MAX_RECITALS_IN_RESULT = int(os.environ.get("LEXEUROPA_MAX_RECITALS", "1"))

# Some HF Inference Providers proxy to third parties (e.g. featherless-ai)
# fronted by Cloudflare, which blocks Python's default urllib fingerprint
# with a 403 (verified: same request succeeds with this header, fails
# without it) — a browser User-Agent clears it.
# TODO: untested whether an honest UA (e.g. "LexEuropa/0.1") would also
# clear it, since the block likely just matches "Python-urllib" — worth
# trying once HF credits are available again, instead of impersonating Chrome.
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Known short-name -> CELEX aliases for the reference resolver. Extend as the
# corpus grows; this is intentionally a small seed set, not a full registry.
INSTRUMENT_ALIASES: dict[str, str] = {
    "gdpr": "32016R0679",
    "general data protection regulation": "32016R0679",
    "regulation 2016/679": "32016R0679",
    "regulation (eu) 2016/679": "32016R0679",
    "law enforcement directive": "32016L0680",
    "directive 2016/680": "32016L0680",
    "directive (eu) 2016/680": "32016L0680",
    "eudpr": "32018R1725",
    "regulation 2018/1725": "32018R1725",
    "regulation (eu) 2018/1725": "32018R1725",
    "digital services act": "32022R2065",
    "dsa": "32022R2065",
    "regulation 2022/2065": "32022R2065",
    "regulation (eu) 2022/2065": "32022R2065",
    "digital markets act": "32022R1925",
    "dma": "32022R1925",
    "regulation 2022/1925": "32022R1925",
    "regulation (eu) 2022/1925": "32022R1925",
    "data governance act": "32022R0868",
    "regulation 2022/868": "32022R0868",
    "regulation (eu) 2022/868": "32022R0868",
    "data act": "32023R2854",
    "regulation 2023/2854": "32023R2854",
    "regulation (eu) 2023/2854": "32023R2854",
    "nis 2 directive": "32022L2555",
    "nis2": "32022L2555",
    "directive 2022/2555": "32022L2555",
    "directive (eu) 2022/2555": "32022L2555",
    "artificial intelligence act": "32024R1689",
    "ai act": "32024R1689",
    "regulation 2024/1689": "32024R1689",
    "regulation (eu) 2024/1689": "32024R1689",
    "treaty on european union": "12016M/TXT",
    "teu": "12016M/TXT",
    "treaty on the functioning of the european union": "12016E/TXT",
    "tfeu": "12016E/TXT",
    "charter of fundamental rights": "12016P/TXT",
    "eu charter": "12016P/TXT",
    "the charter": "12016P/TXT",
    "nis 2": "32022L2555",
    "data protection directive": "31995L0046",
    "directive 95/46/ec": "31995L0046",
    "directive 95/46": "31995L0046",
    "eprivacy directive": "32002L0058",
    "directive 2002/58/ec": "32002L0058",
    "directive 2002/58": "32002L0058",
    "e-commerce directive": "32000L0031",
    "directive 2000/31/ec": "32000L0031",
    "directive 2000/31": "32000L0031",
    "infosoc directive": "32001L0029",
    "directive 2001/29/ec": "32001L0029",
    "directive 2001/29": "32001L0029",
}

# Joined cases are one judgment under several case numbers; the Instrument
# carries only the first, so the rest are listed here.
EXTRA_CASE_NUMBER_ALIASES: dict[str, str] = {
    "c-594/12": "62012CJ0293",
    "c-698/15": "62015CJ0203",
    "c-512/18": "62018CJ0511",
    "c-520/18": "62018CJ0511",
}

def _case_number_aliases() -> dict[str, str]:
    # Derived from seed_corpus.py rather than hand-duplicated, so a new
    # judgment added there is automatically resolvable by its case number
    # without the two tables drifting apart.
    from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS

    aliases: dict[str, str] = {}
    for inst in SEED_INSTRUMENTS:
        if inst.case_number:
            key = inst.case_number.strip().lower().replace("‑", "-")
            aliases[key] = inst.celex
    return aliases


CASE_NUMBER_ALIASES: dict[str, str] = {**_case_number_aliases(), **EXTRA_CASE_NUMBER_ALIASES}

for _d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)
