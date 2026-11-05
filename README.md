# LexEuropa

[![CI](https://github.com/lumenworksco/-LexEuropa/actions/workflows/ci.yml/badge.svg)](https://github.com/lumenworksco/-LexEuropa/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

**An open-model AI legal research assistant for EU data protection and digital law, with every citation verified.**

LexEuropa runs on models published on Hugging Face (a LoRA-tuned [Qwen2.5-3B generator](https://huggingface.co/iPwnds/lexeuropa-qwen3b-merged) served from an Inference Endpoint), so nothing heavy lives on your machine. It fetches real EU legal texts from the Publications Office CELLAR service, chunks them at article/paragraph granularity, retrieves with a hybrid BM25 + dense + exact-reference pipeline, generates an answer as structured *claims with citations*, and then **checks every citation against what was actually retrieved**. Each claim in the answer carries a label: `supported`, `interpretative`, `analysis`, `unsupported` or `uncertain`.

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
| Corpus | 24 instruments, 4,208 chunks, English only: [see below](#corpus) |
| Retrieval | `rank_bm25`, `BAAI/bge-large-en-v1.5` via HF Inference, regex/alias reference resolver, reciprocal rank fusion |
| Reranker | Optional. `BAAI/bge-reranker-v2-m3` locally on CPU if you install the `rerank` extra; otherwise the fused ranking is used |
| Generation | `Qwen2.5-3B-Instruct` + LoRA fine-tune (merged), on a scale-to-zero HF Inference Endpoint |
| Judge | stock `Qwen2.5-7B-Instruct` via HF Inference Providers, deliberately a different model from the generator |
| Verification | A cited chunk that was not retrieved is `unsupported` with no LLM call. Source-identity check catches claims naming one case but citing another. Each citation is judged independently, concurrently. |

## Quickstart

Requires Python 3.11+ and a [Hugging Face token](https://huggingface.co/settings/tokens) with Inference Providers access.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # light: no torch, no local model weights
# pip install -e ".[rerank]"       # optional: local cross-encoder reranker (~3 GB)
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

## Corpus

Everything is fetched from the Publications Office CELLAR service; every CELEX number was verified against a live fetch, and every parse was checked for gap-free article / paragraph numbering.

| Legislation | CELEX |
|---|---|
| GDPR, Regulation (EU) 2016/679 | 32016R0679 |
| Law Enforcement Directive, (EU) 2016/680 | 32016L0680 |
| EU institutions' data protection, (EU) 2018/1725 | 32018R1725 |
| Digital Services Act, (EU) 2022/2065 | 32022R2065 |
| Digital Markets Act, (EU) 2022/1925 | 32022R1925 |
| Data Governance Act, (EU) 2022/868 | 32022R0868 |
| Data Act, (EU) 2023/2854 | 32023R2854 |
| NIS 2 Directive, (EU) 2022/2555 | 32022L2555 |
| AI Act, (EU) 2024/1689 | 32024R1689 |

CJEU judgments: Google Spain (C-131/12), Schrems I (C-362/14), Schrems II (C-311/18), Digital Rights Ireland (C-293/12), Tele2 Sverige (C-203/15), La Quadrature du Net (C-511/18), Breyer (C-582/14), Wirtschaftsakademie (C-210/16), Fashion ID (C-40/17), Planet49 (C-673/17), Nowak (C-434/16), GC and Others v CNIL (C-136/17), Österreichische Post (C-300/21), Meta v Bundeskartellamt (C-252/21), SCHUFA (C-634/21).

Not yet covered: recitals and annexes (articles only), the Treaties and the Charter, older directives (95/46, ePrivacy) that CELLAR serves only as plain HTML, and any language other than English.

## Fine-tuning result

A LoRA adapter (r=16) was trained on 506 examples built deterministically from real chunk text of the original three documents (GDPR and two judgments), with a provision-level held-out split.

**Held-out provisions of those documents** (42 templated questions, same retrieval and judge for all):

| | citation precision | citation recall |
|---|---|---|
| Stock Qwen2.5-7B | 0.82 | 0.88 |
| Qwen2.5-3B + LoRA (v1, live) | 0.95* | 0.95* |
| Qwen2.5-3B + LoRA (v2, rebuilt dataset) | 0.93 | 0.93 |

\*40 of 42; two requests hit a cold-start `503`.

**Instruments the adapter never saw** (25 natural-language questions across the 21 added instruments, gold answers checked against chunk text, so this is the honest generalization test):

| | hit@k | citation precision | citation recall |
|---|---|---|---|
| Stock Qwen2.5-7B, with reranker | 0.96 | 0.71 | 0.80 |
| Qwen2.5-3B + LoRA v1, with reranker | 0.96 | 0.88 | 0.88 |
| Qwen2.5-3B + LoRA v1, no reranker | 0.92 | 0.83 | 0.84 |
| Stock Qwen2.5-7B, no reranker | 0.92 | 0.74 | 0.84 |

The adapter generalizes to law it was never trained on and still beats the stock 7B on citation precision (+0.17 with the reranker). The reranker adds about 0.04-0.05 and one extra retrieval hit, so it is optional. n=25, so treat differences of a few points as noise. v1 and v2 are indistinguishable, so v1 stays the default. **This shows better citation discipline, not better legal reasoning.** An earlier 0.24 vs 0.29 figure predates retrieval fixes and is not comparable.

Artifacts: [v1 merged](https://huggingface.co/iPwnds/lexeuropa-qwen3b-merged), [v1 adapter](https://huggingface.co/iPwnds/lexeuropa-qwen3b-lora), [v2 merged](https://huggingface.co/iPwnds/lexeuropa-qwen3b-merged-v2), [v2 adapter](https://huggingface.co/iPwnds/lexeuropa-qwen3b-lora-v2), [training dataset](https://huggingface.co/datasets/iPwnds/lexeuropa-sft). Full method and caveats are in [`CLAUDE.md`](CLAUDE.md).

## Known limitations

- English-only; the adapter was trained on only the GDPR and two judgments, then evaluated (not retrained) on the rest of the corpus.
- Recitals and annexes are not indexed, so questions about a regulation's stated purpose or an annex list can miss.
- The live adapter (v1) was trained on data with some claims split mid-sentence at inline cross-references. The splitter is fixed and v2 was trained on the rebuilt data, but v2 did not score measurably better, so v1 stays the default.
- The judge rarely uses the `analysis` label; treat that label as unreliable.
- Judgment paragraphs get no ranking beyond exact paragraph references.

## Roadmap

See [`docs/VISION.md`](docs/VISION.md) for the long-term plan (multilingual corpus, knowledge graph, continued pretraining). Next concrete steps: index recitals and annexes, retrain the adapter on the full corpus, grow the corpus, add paragraph-aware retrieval for judgments.

## License

[MIT](LICENSE). Legal texts are © European Union, reused under the EU's [reuse policy](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html).
