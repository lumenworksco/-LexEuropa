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
| Corpus | 31 instruments, 6,872 chunks, English only: [see below](#corpus) |
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

31 instruments, 6,872 chunks, English only. Everything is fetched from the Publications Office CELLAR service. Every CELEX number was verified against a live fetch, and every parse was checked against the real article count (e.g. TFEU 358, Charter 54, GDPR 99). Chunks are article paragraphs, **recitals**, **annex parts** and judgment paragraphs.

| Primary law | CELEX |
|---|---|
| Treaty on European Union (consolidated) | 12016M/TXT |
| Treaty on the Functioning of the EU (consolidated) | 12016E/TXT |
| Charter of Fundamental Rights | 12016P/TXT |

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
| Data Protection Directive 95/46/EC | 31995L0046 |
| ePrivacy Directive 2002/58/EC | 32002L0058 |
| e-Commerce Directive 2000/31/EC | 32000L0031 |
| InfoSoc Directive 2001/29/EC | 32001L0029 |

CJEU judgments: Google Spain (C-131/12), Schrems I (C-362/14), Schrems II (C-311/18), Digital Rights Ireland (C-293/12), Tele2 Sverige (C-203/15), La Quadrature du Net (C-511/18), Breyer (C-582/14), Wirtschaftsakademie (C-210/16), Fashion ID (C-40/17), Planet49 (C-673/17), Nowak (C-434/16), GC and Others v CNIL (C-136/17), Österreichische Post (C-300/21), Meta v Bundeskartellamt (C-252/21), SCHUFA (C-634/21).

Not covered: other languages, national law, Advocate General opinions, Treaty protocols and declarations, and consolidated (amended-as-of-today) versions of the secondary legislation.

## Evaluation

The generator is a LoRA-tuned Qwen2.5-3B (v1, trained on 506 examples from the GDPR and two judgments). Final run on the current 6,872-chunk index, no reranker, stock Qwen2.5-7B as the judge for every row. Precision is the share of cited sources that point at the gold article or case, and recall is whether at least one does.

| | 42 templated questions (P / R) | 25 natural-language questions (P / R) |
|---|---|---|
| Stock Qwen2.5-7B | 0.92 / 0.98 | 0.63 / 0.84 |
| **Qwen2.5-3B + LoRA v1 (default)** | **1.00 / 1.00** | **0.76 / 0.76** |
| Qwen2.5-3B + LoRA v3 (retrained on full corpus) | 0.95 / 0.95 | 0.56 / 0.56 |

The 25-question set, split (retrieval surfaced the gold source for 77% of the legislation questions and 100% of the judgment questions):

| | legislation (13) P / R | judgments (12) P / R |
|---|---|---|
| Stock Qwen2.5-7B | 0.43 / 0.77 | 0.85 / 0.92 |
| v1 | 0.69 / 0.69 | 0.82 / 0.83 |
| v3 | 0.46 / 0.46 | 0.67 / 0.67 |

What this does and does not show:

- v1 is **more precise** than the stock 7B on legislation (it cites fewer wrong sources) but **not more complete**: its recall on unseen law is below the stock model's, and on judgments it is no better. The fine-tune bought citation discipline, not coverage.
- **Retraining on the full corpus (v3) made it worse**, and a pre-declared promotion rule (v3 must be within 0.03 of v1 everywhere) failed, so v1 stays the default. A plausible cause is that the training questions all name their target provision ("What does Recital 47 GDPR state?"), unlike the natural questions in the 25-set, but that is untested. v3 is not published.
- The 42-question set is templated with phrasing v1 was trained on, so its near-perfect scores are optimistic. The 25-question set is small (n=25), so differences of a few points are noise.
- Held-out wording can still reach training through verbatim quotation (judgments quote recitals).
- Retrieval, not generation, caps the legislation questions. Improving it (e.g. indexing article titles) is the clearest next step.

Artifacts: [v1 merged](https://huggingface.co/iPwnds/lexeuropa-qwen3b-merged), [v1 adapter](https://huggingface.co/iPwnds/lexeuropa-qwen3b-lora), [v2 merged](https://huggingface.co/iPwnds/lexeuropa-qwen3b-merged-v2), [v2 adapter](https://huggingface.co/iPwnds/lexeuropa-qwen3b-lora-v2), [training dataset](https://huggingface.co/datasets/iPwnds/lexeuropa-sft) (`v2` tag = the 506 examples v1 and v2 were trained on). Method and caveats: [`CLAUDE.md`](CLAUDE.md).

## Known limitations

- English only; no national law.
- Retrieval is the weak link on natural-language questions about legislation (the right article surfaced for about 3 in 4). Article titles and headings are not indexed for most instruments.
- The live adapter (v1) was trained on only the GDPR and two judgments, and its output has some claims split mid-sentence at inline cross-references (fixed in the data builder; the retrain that used the fix did not score better).
- The judge rarely uses the `analysis` label; treat that label as unreliable.
- Output is research assistance, not legal advice.

## Roadmap

See [`docs/VISION.md`](docs/VISION.md) for the long-term plan. Next concrete steps: improve retrieval for legislation (index article titles, query rewriting), build natural-language training questions without model-written text, and add Advocate General opinions and treaty protocols.

## License

[MIT](LICENSE). Legal texts are © European Union, reused under the EU's [reuse policy](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html).
