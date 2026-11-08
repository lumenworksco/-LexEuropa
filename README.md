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

Requires Python 3.11+ and a [Hugging Face token](https://huggingface.co/settings/tokens) with Inference Providers access. Embeddings, the judge, the fallback and (by default) generation are HF-hosted calls that need account credits; set `LEXEUROPA_EMBED_BACKEND=local` to embed queries on your machine instead.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # light: no torch, no local model weights
# pip install -e ".[rerank]"       # optional: local reranker and local embeddings (~3 GB)
# export LEXEUROPA_EMBED_BACKEND=local   # embed queries on this machine instead of via HF (free)
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

All rows below ran on the same retrieval (6,872 chunks, article headings indexed, at most one recital in the result list, no reranker), the same parser and the same judge (stock Qwen2.5-7B). P = share of cited sources that point at the gold article or case; R = whether at least one does.

| generator | 42 templated (P / R) | 25 natural-language (P / R) | legislation (13) | judgments (12) |
|---|---|---|---|---|
| **LexEuropa v1 + fallback (default)** | 0.95 / 0.95 | **0.86 / 0.88** | 0.81 / 0.85 | 0.91 / 0.92 |
| LexEuropa v1 alone (3B, LoRA) | 0.95 / 0.95 | 0.82 / 0.84 | 0.81 / 0.85 | 0.82 / 0.83 |
| stock Qwen2.5-7B | 0.91 / 0.98 | 0.67 / 0.88 | 0.57 / 0.92 | 0.78 / 0.83 |
| stock Qwen2.5-72B | 0.93 / 1.00 | 0.72 / 0.88 | 0.59 / 0.83 | 0.85 / 0.92 |
| stock Qwen3-235B | 0.91 / 0.98 | 0.66 / 0.92 | 0.53 / 0.92 | 0.80 / 0.92 |
| stock DeepSeek-V3 | 0.89 / 0.98 | 0.68 / 0.88 | 0.58 / 0.92 | 0.79 / 0.83 |

(Llama-3.3-70B is omitted: 13 of its 25 natural-language answers broke the JSON format.)

**The fallback.** If the fine-tuned model refuses (no cited claim) or its output cannot be parsed, the question is retried once on Qwen2.5-72B through the Hugging Face router and the answer is labelled as coming from the fallback (`LEXEUROPA_FALLBACK_LLM_MODEL=""` disables it). It fired on 1 of 25 natural questions and 0 of 42 templated ones. It exists because a live test showed v1 refusing "Can a company send marketing emails without consent?" even though all four paragraphs of ePrivacy Article 13 had been retrieved.

What this does and does not show:

- The 3B fine-tune is **more precise** than every stock model tested, including 235B-parameter ones, because it cites the single most relevant article instead of hedging across several. Its recall is about the same as the 7B and 72B, a few points below the biggest models.
- **That is partly a property of the benchmark.** Each question has one gold article, so a model that cites exactly that article scores best. The metric measures *which sources are cited*, not whether the prose answer is correct or complete, and larger models that cite several relevant provisions are penalized on precision. Treat these numbers as evidence of citation discipline, not of legal reasoning.
- n=25 for the natural-language set, so one question moves a score by about 4 points. The 42 templated questions are easy (the provision is named) and v1 was trained on that phrasing.
- Retraining more adapters did not help. v3 (reference-style training questions) regressed; v4 (natural-language topic questions, whole-article targets) was clearly better than v3 but below v1 (0.68 / 0.75 on the 25 questions, measured with a recital cap of 3, a slightly different retrieval). v3 and v4 are not published.
- Retrieval, not generation, was the biggest lever: indexing article headings and capping recitals lifted hit@8 on 25 questions from 0.88 to 0.96 and, on 34 fresh validation questions, from 0.85 to 0.91.
- Held-out wording can still reach training through verbatim quotation (judgments quote recitals).
- A second weakness seen live: asked about risk management for high-risk AI systems, v1 quoted Recital 65 instead of the operative Article 9 even though Article 9 was retrieved.

Artifacts: [v1 merged](https://huggingface.co/iPwnds/lexeuropa-qwen3b-merged), [v1 adapter](https://huggingface.co/iPwnds/lexeuropa-qwen3b-lora), [v2 merged](https://huggingface.co/iPwnds/lexeuropa-qwen3b-merged-v2), [v2 adapter](https://huggingface.co/iPwnds/lexeuropa-qwen3b-lora-v2), [training dataset](https://huggingface.co/datasets/iPwnds/lexeuropa-sft) (revision tags `v2` = 506 examples used for v1/v2, `v3` = 1,680, `main` = v4's 1,402). Method and caveats: [`CLAUDE.md`](CLAUDE.md).

## Known limitations

- English only; no national law.
- Retrieval surfaces the right provision for about 24 of 25 natural questions on the tuned set and 31 of 34 on fresh ones; the remaining misses are mostly judgments quoting a statute outranking it.
- The fine-tuned model sometimes refuses when the sources do answer the question (mitigated by the fallback) and sometimes quotes a recital instead of the operative article.
- The live adapter (v1) was trained on only the GDPR and two judgments, and its output has some claims split mid-sentence at inline cross-references (fixed in the data builder; the retrain that used the fix did not score better).
- The judge rarely uses the `analysis` label; treat that label as unreliable.
- Output is research assistance, not legal advice.

## Roadmap

See [`docs/VISION.md`](docs/VISION.md) for the long-term plan. Next concrete steps: judge answer quality beyond which sources are cited (human or rubric review of answers), prefer operative articles over recitals when both are retrieved, and add Advocate General opinions and treaty protocols.

## License

[MIT](LICENSE). Legal texts are © European Union, reused under the EU's [reuse policy](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html).
