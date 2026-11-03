# LexEuropa

[![CI](https://github.com/lumenworksco/-LexEuropa/actions/workflows/ci.yml/badge.svg)](https://github.com/lumenworksco/-LexEuropa/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

**Citation-verified legal research over EU law, using open-weight models.**

LexEuropa fetches real EU legal texts from the Publications Office CELLAR service, chunks them at article/paragraph granularity, retrieves with a hybrid BM25 + dense + exact-reference pipeline, generates an answer as structured *claims with citations*, and then **checks every citation against what was actually retrieved**. Each claim in the answer carries a label: `supported`, `interpretative`, `analysis`, `unsupported` or `uncertain`.

> Not legal advice. This is a research prototype; its output does not replace a qualified legal professional.

## How it works

```
question
  → reference resolver ("Art. 17(1) GDPR", "C-131/12")  ┐
  → BM25 keyword search                                 ├→ RRF fusion → cross-encoder rerank
  → dense search (bge-large-en-v1.5)                    ┘
  → LLM generation → { claims: [{ text, citations }] }
  → verification: existence gate (hard check) + LLM judge, per cited chunk
  → answer with per-claim labels
```

| Component | Implementation |
|---|---|
| Corpus | GDPR + two CJEU judgments (C-131/12 *Google Spain*, C-136/17), English only, 569 chunks |
| Retrieval | `rank_bm25`, `BAAI/bge-large-en-v1.5` via HF Inference, regex/alias reference resolver, reciprocal rank fusion |
| Reranker | `BAAI/bge-reranker-v2-m3`, local on CPU |
| Generation | `Qwen2.5-3B-Instruct` + LoRA fine-tune (merged), on a scale-to-zero HF Inference Endpoint |
| Judge | stock `Qwen2.5-7B-Instruct` via HF Inference Providers, deliberately a different model from the generator |
| Verification | A cited chunk that was not retrieved is `unsupported` with no LLM call. Source-identity check catches claims naming one case but citing another. Each citation is judged independently, concurrently. |

## Quickstart

Requires Python 3.11+ and a [Hugging Face token](https://huggingface.co/settings/tokens) with Inference Providers access.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
echo "HF_TOKEN=hf_..." > .env

python -m lexeuropa.cli ingest
python -m lexeuropa.cli ask "What are the conditions for erasure under Article 17 GDPR?" --instrument gdpr
```

The default generator points at a private dedicated Endpoint. To run without it, point generation at any OpenAI-compatible server:

```bash
export LEXEUROPA_LLM_BASE_URL=https://router.huggingface.co/v1
export LEXEUROPA_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Tests (the offline subset needs no network or token):

```bash
python -m pytest tests/ -v -k "not full_hybrid"
```

## Fine-tuning result

A LoRA adapter (r=16) was trained on 506 examples built deterministically from real chunk text, with a provision-level held-out split. On 42 held-out questions:

| | citation precision | citation recall |
|---|---|---|
| Stock Qwen2.5-7B | 0.24 | 0.26 |
| Qwen2.5-3B + LoRA | 0.29 | 0.29 |

A modest gain from a model less than half the size, showing better citation discipline and output-format adherence, **not** better legal reasoning. Full method, caveats and known limitations are in [`CLAUDE.md`](CLAUDE.md).

## Known limitations

- Three-document, English-only corpus; the fine-tuned adapter is shaped by it.
- The adapter's training data contains some claims split mid-sentence at inline cross-references. The splitter is fixed (`lexeuropa/finetune/build_dataset.py`), but the dataset has not been rebuilt and the adapter not retrained.
- The judge rarely uses the `analysis` label; treat that label as unreliable.
- Judgment paragraphs get no ranking beyond exact paragraph references.

## Roadmap

See [`docs/VISION.md`](docs/VISION.md) for the long-term plan (multilingual corpus, knowledge graph, continued pretraining). Next concrete steps: rebuild the dataset and retrain, grow the corpus, add paragraph-aware retrieval for judgments.

## License

[MIT](LICENSE). Legal texts are © European Union, reused under the EU's [reuse policy](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html).
