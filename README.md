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

The default generator points at the author's scale-to-zero Inference Endpoint, which needs that account's token. The model itself is public, so you can serve [`iPwnds/lexeuropa-qwen3b-merged`](https://huggingface.co/iPwnds/lexeuropa-qwen3b-merged) yourself (vLLM, Ollama, ...) or use a stock model:

```bash
export LEXEUROPA_LLM_BASE_URL=https://router.huggingface.co/v1
export LEXEUROPA_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Tests (the offline subset needs no network or token):

```bash
python -m pytest tests/ -v -k "not full_hybrid"
```

## Fine-tuning result

A LoRA adapter (r=16) was trained on 506 examples built deterministically from real chunk text, with a provision-level held-out split. On 42 held-out questions, all three generators run on the **same, current retrieval** (gold source retrieved for 42/42), with the stock 7B as judge:

| | citation precision | citation recall |
|---|---|---|
| Stock Qwen2.5-7B | 0.82 | 0.88 |
| Qwen2.5-3B + LoRA (v1) | 0.95* | 0.95* |
| Qwen2.5-3B + LoRA (v2, rebuilt dataset) | 0.93 | 0.93 |

\*40 of 42 questions; the other two hit a cold-start `503` from the scale-to-zero endpoint, not a model failure.

The fine-tuned 3B beats the stock 7B by roughly 0.1 on citation precision, better on 9 questions and worse on 2-3, from a model less than half the size. v1 and v2 are indistinguishable at this sample size, so the live default stays v1. **Read this as better output-format and citation discipline, not better legal reasoning.** The questions are templated (the same templates appear in training, on different provisions), so the absolute numbers are optimistic. An earlier run reported 0.24 vs 0.29; that predates retrieval fixes and is not comparable.

Artifacts: [v1 merged](https://huggingface.co/iPwnds/lexeuropa-qwen3b-merged), [v1 adapter](https://huggingface.co/iPwnds/lexeuropa-qwen3b-lora), [v2 merged](https://huggingface.co/iPwnds/lexeuropa-qwen3b-merged-v2), [v2 adapter](https://huggingface.co/iPwnds/lexeuropa-qwen3b-lora-v2), [training dataset](https://huggingface.co/datasets/iPwnds/lexeuropa-sft). Full method and caveats are in [`CLAUDE.md`](CLAUDE.md).

## Known limitations

- Three-document, English-only corpus; the fine-tuned adapter is shaped by it.
- The live adapter (v1) was trained on data with some claims split mid-sentence at inline cross-references. The splitter is fixed and v2 was trained on the rebuilt data, but v2 did not score measurably better, so v1 stays the default.
- The judge rarely uses the `analysis` label; treat that label as unreliable.
- Judgment paragraphs get no ranking beyond exact paragraph references.

## Roadmap

See [`docs/VISION.md`](docs/VISION.md) for the long-term plan (multilingual corpus, knowledge graph, continued pretraining). Next concrete steps: grow the corpus, add paragraph-aware retrieval for judgments.

## License

[MIT](LICENSE). Legal texts are © European Union, reused under the EU's [reuse policy](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html).
