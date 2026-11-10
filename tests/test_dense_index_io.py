"""The dense index file format: plain unicode ids (no pickle) now; old object-array files still load locally."""
import numpy as np
import pytest

import lexeuropa.retrieval.dense_index as di
from lexeuropa.schema import Chunk


def _chunks(n=3):
    return [Chunk(id=f"X:en:art_{i}:001", celex="X", lang="en", instrument_title="t", citation=f"Art {i}",
                  unit_type="article_paragraph", article=str(i), para_index="1", text="t", source_url="") for i in range(n)]


def test_saved_index_has_plain_unicode_ids_and_loads_without_pickle(tmp_path, monkeypatch):
    chunks = _chunks(); idx = di.DenseIndex(chunks, np.eye(3, dtype=np.float32))
    path = tmp_path / "d.npz"; idx.save(path)
    with np.load(path, allow_pickle=False) as z:
        assert z["ids"].dtype.kind == "U"
    monkeypatch.setattr(di, "PUBLIC_MODE", True)          # the public demo's stricter rule
    loaded = di.DenseIndex.load(path, {c.id: c for c in chunks})
    assert [c.id for c in loaded.chunks] == [c.id for c in chunks]


def _write_old_format(path, chunks):
    np.savez(path, vectors=np.eye(len(chunks), dtype=np.float32), ids=np.array([c.id for c in chunks], dtype=object), embed_model=di.EMBED_MODEL)


def test_an_old_object_array_index_still_loads_locally(tmp_path, monkeypatch):
    chunks = _chunks(); path = tmp_path / "old.npz"; _write_old_format(path, chunks)
    monkeypatch.setattr(di, "PUBLIC_MODE", False)
    assert len(di.DenseIndex.load(path, {c.id: c for c in chunks}).chunks) == 3


def test_the_public_demo_refuses_an_index_that_needs_pickle(tmp_path, monkeypatch):
    chunks = _chunks(); path = tmp_path / "old.npz"; _write_old_format(path, chunks)
    monkeypatch.setattr(di, "PUBLIC_MODE", True)
    with pytest.raises(ValueError):
        di.DenseIndex.load(path, {c.id: c for c in chunks})
