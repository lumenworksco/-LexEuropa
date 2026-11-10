"""Dense retrieval via HF Inference Providers' bge-large-en-v1.5 embeddings.

Brute-force cosine search over NumPy, not FAISS — the corpus is small.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import numpy as np

from lexeuropa.auth_context import current_token
from lexeuropa.config import EMBED_BACKEND, EMBED_MODEL, HF_INFERENCE_BASE_URL, HTTP_USER_AGENT, PUBLIC_MODE
from lexeuropa.schema import Chunk

# Per the model card: add to queries only, never to indexed passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingError(RuntimeError):
    pass


_local_model = None


def _embed_locally(texts: list[str]) -> np.ndarray:
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                'LEXEUROPA_EMBED_BACKEND=local needs sentence-transformers (pip install -e ".[rerank]")'
            ) from e
        _local_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return np.asarray(
        _local_model.encode(texts, normalize_embeddings=True, batch_size=8, show_progress_bar=False),
        dtype=np.float32,
    )


def embed_texts(texts: list[str], *, retries: int = 3) -> np.ndarray:
    """Retries transient failures (timeouts, connection resets) — a single
    stalled request shouldn't be fatal on its own, especially for a batch
    job making hundreds of calls. HTTP errors (4xx/5xx with a real
    response) are not retried, since those are not transient."""
    if EMBED_BACKEND == "local" and texts:
        return _embed_locally(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    headers = {"Content-Type": "application/json", "User-Agent": HTTP_USER_AGENT}
    token = current_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{HF_INFERENCE_BASE_URL}/models/{EMBED_MODEL}/pipeline/feature-extraction",
        data=json.dumps({"inputs": texts}).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                embeddings = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            hint = " (HF_TOKEN is not set — see CLAUDE.md Setup)" if not token else ""
            raise EmbeddingError(
                f"embedding request to {HF_INFERENCE_BASE_URL} failed: HTTP {e.code}: {body}{hint}"
            ) from e
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2**attempt)  # 1s, 2s, 4s, ...
                continue
            hint = " (HF_TOKEN is not set — see CLAUDE.md Setup)" if not token else ""
            raise EmbeddingError(
                f"embedding request to {HF_INFERENCE_BASE_URL} failed after {retries} attempts: "
                f"{last_error}{hint}"
            ) from last_error

    if not isinstance(embeddings, list) or not embeddings:
        raise EmbeddingError(f"unexpected embeddings response shape: {embeddings}")
    return np.array(embeddings, dtype=np.float32)


class DenseIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = vectors / norms

    @classmethod
    def build(
        cls, chunks: list[Chunk], batch_size: int = 16, reuse: "DenseIndex | None" = None
    ) -> "DenseIndex":
        """Embeds every chunk, except those whose id and text already have a
        vector in `reuse` (so adding documents only embeds the new chunks)."""
        if not chunks:
            return cls(chunks, np.zeros((0, 0), dtype=np.float32))
        cached = {}
        if reuse is not None:
            cached = {c.id: (c.index_text, v) for c, v in zip(reuse.chunks, reuse.vectors)}
        vectors: list = [None] * len(chunks)
        todo = []
        for i, c in enumerate(chunks):
            hit = cached.get(c.id)
            if hit is not None and hit[0] == c.index_text:
                vectors[i] = hit[1]
            else:
                todo.append(i)
        for start in range(0, len(todo), batch_size):
            idx = todo[start : start + batch_size]
            embedded = embed_texts([chunks[i].index_text for i in idx])
            for i, v in zip(idx, embedded):
                vectors[i] = v
        print(f"  embedded {len(todo)} chunks, reused {len(chunks) - len(todo)}")
        return cls(chunks, np.vstack(vectors))

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        qvec = embed_texts([QUERY_INSTRUCTION + query])[0]
        qvec = qvec / (np.linalg.norm(qvec) or 1.0)
        sims = self.vectors @ qvec
        top_idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def save(self, path) -> None:
        np.savez(
            path,
            vectors=self.vectors,
            ids=np.array([c.id for c in self.chunks], dtype=str),   # plain unicode: loads without pickle
            embed_model=EMBED_MODEL,
        )

    @classmethod
    def load(cls, path, chunks_by_id: dict[str, Chunk]) -> "DenseIndex":
        data = np.load(path, allow_pickle=False)
        try:
            ids = data["ids"].tolist()      # arrays load lazily: an object-array index only fails here
        except ValueError:
            if PUBLIC_MODE:
                raise   # an object-array index needs pickle; the public demo never unpickles
            data = np.load(path, allow_pickle=True)   # index written by an older version
            ids = data["ids"].tolist()
        stored_model = str(data["embed_model"]) if "embed_model" in data else None
        if stored_model is not None and stored_model != EMBED_MODEL:
            raise EmbeddingError(
                f"dense index was built with {stored_model!r} but EMBED_MODEL is now "
                f"{EMBED_MODEL!r} — re-run `python -m lexeuropa.cli ingest`"
            )
        chunks = [chunks_by_id[i] for i in ids]
        inst = cls.__new__(cls)
        inst.chunks = chunks
        inst.vectors = data["vectors"]
        return inst
