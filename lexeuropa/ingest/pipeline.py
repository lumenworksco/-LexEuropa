"""End-to-end ingestion: fetch -> parse -> chunk -> persist -> index."""
from __future__ import annotations

import dataclasses
import json

from lexeuropa.config import BM25_INDEX_PATH, CHUNKS_PATH, DENSE_INDEX_PATH, INSTRUMENTS_PATH
from lexeuropa.ingest import cellar
from lexeuropa.ingest.chunker import chunk_instrument
from lexeuropa.ingest.seed_corpus import SEED_INSTRUMENTS
from lexeuropa.retrieval.bm25_index import BM25Index
from lexeuropa.retrieval.dense_index import DenseIndex
from lexeuropa.schema import Chunk, Instrument


def run_ingest(instruments: list[Instrument] = SEED_INSTRUMENTS, *, build_dense: bool = True) -> list[Chunk]:
    all_chunks: list[Chunk] = []

    for instrument in instruments:
        for lang in instrument.languages:
            print(f"fetching {instrument.celex} ({lang}) ...")
            xhtml = cellar.fetch(instrument.celex, lang)
            chunks = chunk_instrument(instrument, lang, xhtml)
            print(f"  -> {len(chunks)} chunks")
            all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("ingestion produced zero chunks — check the seed corpus and parser")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    with open(INSTRUMENTS_PATH, "w", encoding="utf-8") as f:
        for inst in instruments:
            f.write(json.dumps(dataclasses.asdict(inst), ensure_ascii=False) + "\n")

    print(f"building BM25 index over {len(all_chunks)} chunks ...")
    BM25Index(all_chunks).save(BM25_INDEX_PATH)

    if build_dense:
        print("building dense (bge-large-en-v1.5) index ... this calls HF Inference for every chunk")
        DenseIndex.build(all_chunks).save(DENSE_INDEX_PATH)

    print(f"done: {len(all_chunks)} chunks from {len(instruments)} instruments")
    return all_chunks


def load_chunks() -> list[Chunk]:
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


def load_instruments() -> list[Instrument]:
    instruments = []
    with open(INSTRUMENTS_PATH, encoding="utf-8") as f:
        for line in f:
            instruments.append(Instrument(**json.loads(line)))
    return instruments
