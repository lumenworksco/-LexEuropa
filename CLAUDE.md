# CLAUDE.md — LexEuropa

Guidance for working in this repo. See `docs/VISION.md` for the full long-term project vision; this file covers what's actually built (Stage 1: Legal RAG) plus setup and commands.

## What's implemented

A working vertical slice of Stage 1 ("Legal RAG"): fetch real EU legal texts, chunk them at article/paragraph granularity, index them for hybrid retrieval, generate a cited answer with an open-weight LLM, and verify every citation against what was actually retrieved. **English-only** — see "Design decisions".

```
User question → reference resolver + BM25 + dense (bge-large-en-v1.5) → RRF fusion
              → cross-encoder rerank → LLM generation (structured claims+citations)
              → citation verification (existence gate + LLM-judge label)
              → answer with per-claim labels
```

Stages 3–4 from `docs/VISION.md` (continued pretraining, a custom foundation
model, the knowledge graph) are **not** built. Stage 2 (fine-tuning) has
been run once end-to-end — a real LoRA adapter trained, merged, deployed
for eval, and compared against the baseline — see "Stage 2: Fine-tuning"
below for the actual numbers and what they do and don't show.

## What runs where

| Component | Where | Model |
|---|---|---|
| Generation (default) | Dedicated HF Inference Endpoint, scale-to-zero (`LLM_BASE_URL`) | `iPwnds/lexeuropa-qwen3b-merged` (Qwen2.5-3B + LoRA, merged — see "Stage 2: Fine-tuning") |
| Judge (always, regardless of `LLM_MODEL`) | HF Inference Providers (`router.huggingface.co/v1`, featherless-ai) | `Qwen/Qwen2.5-7B-Instruct` |
| Embeddings | HF Inference Providers (`router.huggingface.co/hf-inference`) | `BAAI/bge-large-en-v1.5` |
| Reranker | **Optional, local** (CPU; `pip install -e ".[rerank]"`, off if not installed or `LEXEUROPA_USE_RERANKER=0`) | `BAAI/bge-reranker-v2-m3` — confirmed not viable via `hf-inference`, see "What's implemented" below |
| BM25, reference resolver, verification logic | Local (pure Python, no model) | — |

**The fine-tuned model is the live default**, not just an eval artifact —
`config.py`'s `LLM_BASE_URL`/`LLM_MODEL` point at the dedicated Endpoint
(`lexeuropa-eval`, `iPwnds/lexeuropa-qwen3b-merged`) by default. The
Endpoint is **scale-to-zero** (`minReplica: 0`, 15-minute idle timeout) —
it costs nothing while idle, but a cold request after 15 idle minutes
pays a restart cost (observed: well under a minute, model is already on
attached storage). `generation/client.py`'s default timeout was raised
from 120s to 240s to cover this. The judge is a deliberately separate
model/endpoint (`JUDGE_MODEL`/`JUDGE_BASE_URL`, pinned to the stock
hosted 7B via Inference Providers) regardless of what `LLM_MODEL` points
at — see "Design decisions" and
`tests/test_verification.py::test_judge_always_calls_its_own_configured_model_and_endpoint`.
Override `LEXEUROPA_LLM_BASE_URL`/`LEXEUROPA_LLM_MODEL` to point generation
at something else (e.g. back at the stock router, or a local server).

**The reranker stays local — confirmed broken via `hf-inference`
serverless, not just unavailable.** Four request shapes were tried
against `BAAI/bge-reranker-v2-m3`:
1. `text-classification` with `{"text": ..., "text_pair": ...}` — `400`.
2. `text-classification` with a list of `[text, text_pair]` pairs — `400`.
3. A dedicated `/pipeline/rerank` path — `400` ("needs an `inputs` key").
4. `sentence-similarity` with `{"source_sentence": ..., "sentences": [...]}`
   — returns `200`, but the scores are **backwards**: tested with one
   clearly-relevant GDPR passage, one generic GDPR preamble, and one
   totally unrelated sentence about a fox — the unrelated sentence scored
   *highest* (0.874 vs 0.658/0.713). HF's generic pipeline wrapper doesn't
   correctly invoke this model's actual cross-encoder joint-scoring
   mechanism; it silently returns semantically meaningless scores instead
   of erroring. **Using this shape would make retrieval worse, not just
   fail to move it off-machine — don't resurrect it.**

Remaining options, neither attempted:
1. Keep it local (current state — the correct default, not a compromise).
2. Deploy it on a dedicated HF Inference Endpoint with a custom-image
   engine that actually supports cross-encoder reranking (e.g. TEI) —
   needs billing, untested whether TEI's `/rerank` route is correctly
   wired for an Endpoint deployment of this specific model.

## Setup

Requires Python 3.11+ and an [HF account + token](https://huggingface.co/settings/tokens) with Inference Providers access.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

echo "HF_TOKEN=hf_..." > .env    # gitignored; loaded automatically by config.py
```

Then:

```bash
python -m lexeuropa.cli ingest
python -m lexeuropa.cli ask "What are the conditions for erasure under Article 17 GDPR?" --instrument gdpr
```

Use a fine-grained, Inference-only token, not a `write`-scoped one.

**Billing:** this was built against a free/prepaid HF account. Mid-session,
chat-completion calls started failing with a real `402 Payment Required`
("You have depleted your monthly included credits") — confirmed via direct
`curl`, not a bug here. Embeddings (`hf-inference`, first-party) kept
working; only third-party-provider chat completions (featherless-ai) were
credit-limited. Purchasing pre-paid credits or subscribing to HF PRO is
required for sustained use — that's the user's call, not something to
work around in code.

**Cloudflare / bot-fingerprint block, already fixed in code, documented so
it isn't rediscovered:** HF's router proxies some providers (featherless-ai)
to third-party infrastructure sitting behind Cloudflare, which returned a
`403` ("Access denied ... based on your browser's signature", error 1010)
for Python's default `urllib` request fingerprint specifically — the exact
same request succeeded via `curl`. Fixed by sending a browser `User-Agent`
header (`config.HTTP_USER_AGENT`, used by both `generation/client.py` and
`retrieval/dense_index.py`). If you see an unexplained 403 from an HF
Inference Providers call, check this first.

## Commands

```bash
# Fetch, parse, chunk and index the corpus (24 instruments, ~4,200 chunks, English only; see seed_corpus.py).
# Caches raw fetches to data/raw/, so re-running only re-fetches missing docs.
python -m lexeuropa.cli ingest

# Ask a question. --instrument gives the reference resolver a default
# instrument to resolve bare "Article N" mentions against (see below).
python -m lexeuropa.cli ask "What are the conditions for erasure under Article 17 GDPR?" --instrument gdpr
python -m lexeuropa.cli ask "..." --json     # machine-readable output

# Run the test suite. One test exercises the full live pipeline (HF chat +
# HF embeddings + the local reranker) and skips cleanly if HF_TOKEN/network
# aren't available. Everything else (chunking, the reference resolver,
# BM25, the verifier's existence gate, prompt parsing) runs fully offline
# against tests/fixtures/, no network needed:
python -m pytest tests/ -v
python -m pytest tests/ -v -k "not full_hybrid"   # offline subset only

# Run the small retrieval + citation eval harness against eval/questions.jsonl.
python eval/run_eval.py
```

If generation is requested without a working `HF_TOKEN`/credits, `ask`
fails with a clear error rather than a stack trace or a silent fallback to
a different backend.

## Architecture

```
lexeuropa/
  schema.py                 Instrument, Chunk dataclasses — the whole data model
  config.py                 paths, HF model/endpoint settings, instrument/case-number alias tables, .env loading
  ingest/
    cellar.py                fetches XHTML from the Publications Office CELLAR service
    parser.py                structure-aware parsing (ELI legislation divs / judgment count-tables)
    chunker.py                parsed units -> Chunk objects with stable ids + citations
    seed_corpus.py            the small, real corpus definition (verified CELEX numbers)
    pipeline.py                orchestrates fetch->parse->chunk->persist->index
  retrieval/
    bm25_index.py              language-neutral keyword search (rank_bm25)
    dense_index.py             bge-large-en-v1.5 embeddings via HF Inference Providers, brute-force cosine search
    reference_resolver.py      regex+alias exact-citation resolution ("Art. 17(1) GDPR", "C-131/12")
    fusion.py                  reciprocal rank fusion of resolver+BM25+dense rankings
    reranker.py                 BAAI/bge-reranker-v2-m3 cross-encoder (local, CPU), graceful fallback
    pipeline.py                 ties the above into one retrieve() call
  generation/
    client.py                   OpenAI-compatible chat client (points at HF Inference Providers)
    prompts.py                   system prompt + structured-claims JSON parsing + bracket-repair
  verification/
    verifier.py                  existence gate + LLM-judge -> supported/interpretative/analysis/unsupported/uncertain
  answer.py                      wires retrieval+generation+verification into answer_question()
  cli.py                          `ingest` and `ask` subcommands

data/raw/          cached CELLAR fetches (gitignored, rebuild with `ingest`)
data/processed/    chunks.jsonl, instruments.jsonl (gitignored)
data/index/         BM25 pickle + dense .npz (gitignored)
eval/                small hand-built gold set + hit@k / citation precision-recall harness
tests/               offline unit tests + tests/fixtures/ (real cached CELLAR XHTML, not synthetic)
.env                 HF_TOKEN (gitignored, not committed)
```

## Design decisions (and why)

- **English-only, by explicit request.** The seed corpus, `LANG_TO_CELLAR`,
  `INSTRUMENT_ALIASES` and the system prompt are all single-language. The
  vision doc's long-term plan still describes a multilingual system (§6) —
  that's the target architecture, not what's built.
- **EUR-Lex's own site is not fetched.** A live check showed
  `eur-lex.europa.eu` sits behind a CloudFront WAF that returns HTTP 202
  with `x-amzn-waf-action: challenge` for scripted requests — no document
  body. The Publications Office's CELLAR service
  (`publications.europa.eu/resource/celex/{CELEX}` with content
  negotiation via `Accept` / `Accept-Language`) resolves the same
  documents cleanly and is what `ingest/cellar.py` uses.
- **No stemming in BM25.** Plain lowercase Unicode word/number splitting,
  no English stemmer — keeps the index language-neutral rather than baking
  in an English-only assumption a future non-English corpus would degrade under.
- **No FAISS.** The corpus is ~4,200 chunks (a 17 MB dense matrix); a brute-force NumPy
  cosine search is simpler to read and fast enough.
- **The reference resolver only resolves an "Article N" mention when an
  instrument alias (e.g. "GDPR") is textually nearby, or a default
  instrument is passed explicitly.** Silently assuming an instrument for a
  bare article number would be exactly the kind of unsupported guess the
  verification layer exists to catch elsewhere.
- **The verifier's existence gate is a hard, non-LLM check.** If a claim
  cites a chunk id that wasn't in the retrieved set, it's labeled
  `unsupported` immediately — no judge call.
- **The generator is an open-weight model (Qwen2.5), served hosted rather
  than run on this machine.** `generation/client.py` speaks the
  OpenAI-compatible protocol, so it also works unmodified against a local
  Ollama/vLLM/llama.cpp server via `LEXEUROPA_LLM_BASE_URL` if needed.
- **Citations use short handles ("S1") in the prompt, mapped back to real
  chunk ids in `answer.py` before verification.** Tested as a fix for a
  JSON-reliability problem on a smaller local model (see "History" below);
  kept because it also just makes prompts shorter.
- **`parse_response` repairs one specific JSON failure shape**: every
  inner list/object closed correctly, but the model stopped one token
  before the final outer `}`. It reconstructs the missing closers by
  tracking bracket depth outside of string literals, and deliberately does
  *not* attempt this when `finish_reason == "length"` (a real cutoff, which
  can truncate mid-string) or when the text ends inside an open string.
  Live-validated: it recovered 2 real answers that would otherwise have
  been `ParseError`s. See `tests/test_prompts.py`.
- **Test fixtures are real, cached CELLAR responses**, not synthetic text.

## History: why some of the above exists

This was first built against a local Ollama + `qwen2.5:3b-instruct` stack
on an 8GB Mac, then switched to HF Inference Providers mid-session at the
user's request ("run it from HF, not locally") plus a request to drop
Dutch support. Findings from the local-3B stack that motivated code still
in this repo:
- The `S1`-style citation handles and the bracket-repair in
  `parse_response` were both built to fix a specific, repeatable JSON
  failure on `qwen2.5:3b-instruct` (never a full generic "3B can't do
  JSON" problem — one exact missing-brace pattern, confirmed via the raw
  output of 4 live failures). Neither has been *needed* on the hosted 7B
  yet, but both are cheap and harmless to keep, and the repair already
  proved itself once on a local eval re-run (fired twice, recovered two
  real answers).
- The reranker runs on CPU, explicitly, because letting
  `sentence-transformers` auto-select the Metal/MPS GPU crashed with
  `kIOGPUCommandBufferCallbackErrorOutOfMemory` when Ollama's own
  GPU-resident models were also loaded on that 8GB machine. Since
  generation and embedding no longer run locally at all, this specific
  contention no longer applies — but CPU is still fine for a model this size.
- Findings that were specific to the 3B/Ollama/Dutch stack and don't carry
  over (superseded, not currently true): the 4096-token Ollama context
  ceiling, the `lexeuropa-qwen` custom Modelfile (deleted), and the
  "verwerping"/"verwerking" Dutch fidelity slip. The **general point**
  those illustrated — the LLM-judge checks whether cited text supports a
  claim, not whether it's word-for-word faithful or complete — is still
  architecturally true and worth remembering; the specific numbers aren't.

## Known limitations (current stack)

- **~~The reference resolver doesn't handle plural article references~~
  — fixed.** "Articles 9 and 10", "Article 6, 7 and 9" and
  "Articles 17(1) and 6(2)" now all resolve each article independently
  (`_PLURAL_ARTICLES_RE`/`_ARTICLE_ITEM_RE` in `reference_resolver.py`;
  requires a mandatory "and" to avoid overlapping with the existing
  single-`Article N` pattern, which still handles the non-plural case).
  See `tests/test_reference_resolver.py`'s plural-article tests.
- **Judgment paragraphs get no general ranking treatment** beyond the
  single-paragraph pin already in place (`resolve_precise`, for an
  unambiguous case+paragraph mention). A bare case number with no
  paragraph named still resolves to every paragraph of that judgment,
  not just the operative holding.
- **~~The LLM-judge doesn't check source-identity~~ — fixed.** A
  deterministic check (`_source_identity_mismatch` in `verifier.py`) now
  catches a claim that names one case but cites only a different
  judgment's paragraphs before the judge is even called — see "Design
  decisions" and `tests/test_verification.py`'s
  `test_claim_naming_one_case_but_citing_a_different_one_is_unsupported`,
  which directly reproduces the original live bug.
- **~~The judge evaluated all of a claim's cited chunks together~~ —
  fixed.** Each cited chunk is now judged independently
  (`verify_claim` calls `judge_fn` once per chunk) and the claim-level
  label is the best of the per-citation results — but each citation's
  own outcome is preserved in `ClaimVerification.citation_labels` /
  the CLI's "per-citation breakdown", so a co-citation that didn't
  actually hold up is visible instead of hidden behind the one that
  did. See `_aggregate_citation_labels` and
  `tests/test_verification.py::test_irrelevant_co_citation_does_not_hide_behind_a_relevant_one`.
  Trade-off: a claim with N citations now costs N judge calls instead
  of 1 — observed live, a two-source question's judge calls alone added
  roughly a minute to `ask`'s wall time. Mitigated: the per-citation
  judge calls for a claim now run concurrently (`ThreadPoolExecutor` in
  `verify_claim`), so a claim's latency is about one judge call, not N.
- **Resolver pinning was gated on the wrong count — fixed, found while
  live-testing the above.** `retrieval/pipeline.py`'s pinning previously
  gated on `len(resolver.resolve(...))`, which includes a bare
  case-number's full-paragraph dump (can be 100+ chunks). A query naming
  both an exact article *and* a bare case number (e.g. "Article 17 GDPR
  ... C-131/12") pushed that combined count past
  `MAX_PINNED_RESOLVER_HITS`, silently disabling pinning for the article
  hit too — live-observed: Article 17 GDPR chunks were completely absent
  from "Authorities retrieved" for exactly this kind of question. Fixed
  by gating on a new `resolver.resolve_precise()` (the narrow hits only,
  excluding the bare-case dump) instead of `resolve()`'s combined total.
  See `tests/test_retrieval_pinning.py::test_article_pin_survives_a_bare_case_number_in_the_same_query`,
  which reproduces the live bug directly; confirmed fixed against the
  real corpus afterward (Article 17(1)/(2)/(3) now appear in
  "Authorities retrieved" for that exact question).
- **The judge's `analysis` label was never observed, in 3 genuine live
  attempts on the hosted 7B.** Built from real corpus text: (1) a claim
  combining Article 6(1)(a) and Article 17(1)(b) → `supported`, not
  `analysis` — Article 17(1)(b) already cross-references Article 6(1)(a)
  explicitly, so this one was arguably too easy; (2) a claim combining
  Article 5(1)(c) data minimisation with Article 9(1)'s special-category
  prohibition to conclude a controller should avoid unnecessary health
  questions on a survey → `supported` again, even though neither excerpt
  states that conclusion; (3) a claim chaining Article 21 (objection) into
  Article 17 (erasure) → `interpretative`, still not `analysis`. By
  contrast, `interpretative` triggers correctly and reliably: a claim
  that C-131/12 ¶88 establishes a de-referencing obligation despite
  Directive 95/46 never using that term got `interpretative` with a
  sensible reason, on the first attempt. Working theory, not confirmed:
  this model's judge tends to call anything "supported" once the
  necessary premises are directly quoted, even when the claim's
  *conclusion* is a synthesis — it may rarely or never reach for
  `analysis` at all. Don't trust an `analysis` label if one ever appears;
  it hasn't been observed to fire even when it should.
- **The per-article/per-case gold standard in `eval/questions.jsonl` can
  under-score a citation that's substantively correct but from a secondary
  source** (e.g. a judgment paragraph that quotes an Article verbatim) or
  a genuinely correct but differently-numbered provision (e.g. Article
  4(11)'s definition of consent, cited for a question whose gold is
  Article 7). Read eval numbers with that in mind, not as pure retrieval error.
- **`eval/questions.jsonl` questions 3-5 are resolved via the reference
  resolver** (they name "Article 6", "C-131/12", "C-136/17" explicitly),
  so their hit@k is close to guaranteed by design — only questions 1-2
  exercise genuine semantic retrieval.

## Stage 2: Fine-tuning (trained, deployed once for eval, results below)

Requested explicitly: a real fine-tuned model, not just calling a stock
one via API. What exists so far, and what's still blocked:

**What this can realistically teach**: schema adherence (always valid
JSON), correct-handle citation among distractor sources, and refusing
when the given sources don't cover the question — the exact failure
modes hit during Stage-1 testing on a smaller local model (see
"History"). **What it can't teach**: legal reasoning generally — the
corpus is 3 documents (GDPR + 2 CJEU judgments), so an adapter trained
only on it will be GDPR/search-engine-de-referencing-shaped, not a
general EU legal reasoner. Treat it as a reliability fix for the RAG
pipeline's output format, not a step toward general legal expertise —
that needs a much larger, more varied training corpus first.

**No training data was model-written.** Generation and the LLM-judge are
blocked by depleted HF credits, and writing targets from memory would
violate this project's own no-fabrication rule. Instead:

- `lexeuropa/finetune/split.py` — a **deterministic split by provision**,
  not by example: every 5th GDPR article plus articles already used in
  `eval/questions.jsonl` or manual testing (6, 7, 9, 17, 21), and every
  5th judgment paragraph, held out from training entirely (25.8% — 147 of
  569 chunks). Run `python -m lexeuropa.finetune.split` to regenerate
  `data/finetune/held_out_provisions.json`.
- `lexeuropa/finetune/build_eval_questions.py` — expands the eval set to
  42 templated questions (`eval/questions.jsonl` only had 5) covering
  *only* held-out provisions, with gold `celex`/`article` taken directly
  from chunk metadata. Written to `eval/questions_heldout.jsonl`, kept
  separate from the original hand-built file.
- `lexeuropa/finetune/build_dataset.py` — builds SFT training examples
  using only train-split chunks: real hybrid retrieval (BM25 + HF
  embeddings + the local reranker — none of that is credit-blocked) picks
  realistic distractor sources, and the target answer is built by
  splitting the real chunk text into short claims (one per lettered
  point, e.g. GDPR's "(a)...(f)" lists; the whole paragraph otherwise),
  each citing exactly the chunk it came from. The gold chunk's position
  among sources is shuffled per example, so the adapter can't learn
  "always cite S1". ~20% refusal examples are mixed in: a question about
  one chunk, paired with sources retrieved for an unrelated one, target
  labeled with an empty-citation refusal claim. Every example is
  validated through the real `parse_response` before being kept.
  No-leakage is structural, not just checked after the fact: `bm25`,
  `dense` and the resolver used while building examples are constructed
  only from train-split chunks, so a held-out chunk has no code path to
  appear even as a distractor — see
  `tests/test_finetune_dataset.py::test_bm25_and_resolver_built_from_train_split_cannot_surface_held_out_chunks`.
  **Fixed flaw:** the original `split_into_claims` also split on inline
  cross-references ("point (a) of Article 6(1)"), producing mid-sentence
  claims that the first adapter learned. `_list_markers` now only accepts
  real, alphabetical list markers (test:
  `test_split_into_claims_ignores_inline_point_cross_references`). The
  dataset was rebuilt with `--no-rerank` (distractors from the fused
  BM25+dense+resolver ranking; the CPU reranker costs ~50s/example), and
  has 0 dangling claims. **The deployed adapter predates the fix** and must
  be retrained (billed) to benefit.
  Built: **506 examples** (422 positive + 84 refusal, 16.6% refusal rate),
  gold citation spread across all 8 source slots (42–64 examples each —
  not concentrated on S1). Output: `data/finetune/sft_dataset.jsonl`
  (TRL-style `{"messages": [...]}`), stats in
  `data/finetune/sft_dataset_stats.json`.
- `lexeuropa/finetune/train.py` — a LoRA SFT script (TRL + PEFT) for
  `hf jobs uv run`, targeting `Qwen/Qwen2.5-3B-Instruct` — smaller than the
  7B used for production generation, chosen because a small model is both
  where this kind of fine-tuning helps most and the cheaper one to
  train/serve. **Never run — see the blocker below.**

**Blocked on billing, not code.** This account has **$0 pre-paid balance
for HF Jobs** — verified live: even the cheapest possible job
(`hf jobs run --flavor cpu-basic python:3.12 ...`, nominally $0.01/hr)
returned `402 Payment Required: Pre-paid credit balance is insufficient`.
This is a separate balance from the Inference Providers credits also
depleted earlier. Nothing computational can run on HF — training or
otherwise — until pre-paid credits are added. A first LoRA attempt on a
small model would likely cost single-digit dollars (an hour or so on an
L4/A10G), but that's an estimate, not a bill.

**What actually happened, in order:**
1. Dataset uploaded: `iPwnds/lexeuropa-sft` (public HF dataset repo, 506 examples).
2. Smoke-tested `train.py` with `--max-steps 5` on `t4-small`, then again on
   `l4x1` after fixing dtype/max_length/label-masking (see below) — each
   run included the `_check_label_masking` assertion, which failed loudly
   rather than silently training on the wrong tokens.
3. Full fine-tune: `Qwen/Qwen2.5-3B-Instruct` + LoRA (r=16, alpha=32) on
   `l4x1`, 2 epochs, 64 steps, **28 minutes, train_loss 0.0113, 99.98%
   token accuracy**. Adapter pushed to `iPwnds/lexeuropa-qwen3b-lora`
   (public) — confirmed real (119.8MB `adapter_model.safetensors`, not a
   placeholder).
4. Merged into a standalone model (`lexeuropa/finetune/merge_adapter.py`,
   a cheap `cpu-upgrade` job) and pushed to
   `iPwnds/lexeuropa-qwen3b-merged` — deploying a bare PEFT adapter to an
   Endpoint and hoping the serving engine auto-loads it was judged too
   risky to get right blind on billed infrastructure; a merged standalone
   model deploys like any other.
5. Deployed to a temporary HF Inference Endpoint (`l4x1`, vLLM engine,
   `aws/us-east-1`) to run the eval, then **paused immediately after** —
   114 minutes total, **$1.52**.
6. Ran `eval/questions_heldout.jsonl` through both the stock hosted 7B
   (baseline) and the fine-tuned 3B+LoRA (merged), with the judge pinned
   to the stock 7B in both cases via `JUDGE_MODEL`/`JUDGE_BASE_URL` (see
   "Design decisions" — this is exactly the scenario that separation was
   built for, and here it mattered for real).

**Results** (n=42, held-out provisions only):

| | hit@k | citation precision | citation recall | errors |
|---|---|---|---|---|
| Baseline (stock Qwen2.5-7B) | 0.62 | 0.24 | 0.26 | 0 |
| Fine-tuned (Qwen2.5-3B + LoRA) | 0.62 | **0.29** | **0.29** | 1 (network blip, not the adapter — see below) |

hit@k is identical by construction — retrieval doesn't depend on the
generator, so this just confirms both runs saw the same sources. The real
comparison is precision/recall: a **modest improvement from a model
less than half the size**. Per-question: 4 questions improved, 1
regressed (Article 9(1), 1.00→0.00), 37 unchanged — not a dramatic win,
and don't oversell it as one. The one fine-tuned-run error was a DNS
resolution failure on the local machine mid-run (`nodename nor servname
provided`), unrelated to the adapter.

**Concrete behavioral evidence the fine-tune took**: the adapter's
refusal phrasing is now verbatim "The provided sources do not contain
information addressing this question" — exactly the refusal template
from `build_dataset.py`'s synthetic examples, not the baseline's more
varied phrasing ("The Court did not provide a statement...", "...is not
explicitly stated..."). The adapter learned the house style, which is
what it was trained to do.

**What this does and doesn't show:** a 3B model matching or slightly
beating a 7B model on citation precision, after 28 minutes of LoRA
training on 506 synthetic examples, is a real result for schema
adherence and citation discipline — the problems this adapter actually
targets. It is not evidence of improved legal reasoning on either
model's part; both still miss the majority of paragraph-specific
judgment citations for the same structural reason (see "Known
limitations" — judgment paragraphs have no resolver support, and a
templated question like "what did the Court say in paragraph 75?" gives
BM25/dense nothing to match on).

**One more live data point, found afterward while testing the retrieval
fixes above (not part of the n=42 eval run):** asked "According to the
GDPR, how does Article 17 relate to C-131/12?" — a two-source synthesis
question — the fine-tuned model refused ("The provided sources do not
contain information addressing this question") even once retrieval was
fixed to correctly surface both Article 17 GDPR and the relevant C-131/12
paragraph. Consistent with the adapter's training data, which only
modeled refusal for "sources don't cover this question at all," not
"sources are present but need synthesizing across two different document
types" — the ~16.6% refusal examples in `build_dataset.py` don't cover
this shape. One live anecdote, not a measured regression — it's the kind
of question that would need its own eval cases to quantify properly.

**Billed-infrastructure notes for next time:**
- `hf endpoints deploy` requires `--custom-image` with `--engine vllm`
  (e.g. `vllm/vllm-openai:latest`); it is not inferred from `--engine` alone.
- A DNS failure on the local machine (here: a Wi-Fi toggle left `scutil
  --dns` reporting "No DNS configuration available") blocks the `hf` CLI
  exactly like it blocks everything else — there is no automatic retry
  path, and attempting to route around it via a hardcoded IP (`curl
  --resolve`) is correctly treated as a containment-escape pattern and
  blocked. If this happens again with a billed resource running, pause it
  from a *different device* via the HF dashboard — don't wait on the
  affected machine's network to recover.

## Stage 2 follow-up: re-evaluation on current retrieval (2026-11)

The n=42 table above (0.24 vs 0.29) was measured **before** the resolver
fixes (pinning, plural articles, `resolve_precise`); retrieval then
surfaced the gold source for only 26/42 questions. Re-running everything
on today's retrieval (42/42) with the same stock-7B judge, via
`eval/run_eval.py --file eval/questions_heldout.jsonl`:

| | precision | recall | notes |
|---|---|---|---|
| Stock Qwen2.5-7B | 0.82 | 0.88 | `baseline_v2pipeline_eval_results.json` |
| 3B + LoRA v1 | 0.95 | 0.95 | n=40; 2 cold-start 503s from the endpoint, not model errors. `finetuned_v1_currentpipeline_eval_results.json` |
| 3B + LoRA v2 | 0.93 | 0.93 | trained on the rebuilt dataset (claim-splitter fix, `--no-rerank` distractors). `finetuned_v2_eval_results.json` |

Takeaways: most of the old gap was retrieval, not generation; the fine-tune
still adds ~0.1 precision over the stock 7B (better on 9, worse on 2-3
questions); v2 (fixed training data) is not measurably better than v1, so
**v1 remains the default** and the endpoint points at it. Questions share
templates with training, so absolute numbers are optimistic. v2 artifacts
exist on the Hub (`...-lora-v2`, `...-merged-v2`) if wanted.

Operational notes: `hf endpoints`/`HfApi.update_inference_endpoint` with a
new `repository` must also pass the new repo's `revision` (a stale commit
sha from the old repo fails the deploy). `run_eval.py` needs `PYTHONPATH=.`
when run as a script. Total cost of this round: roughly a few dollars of
Jobs plus about two hours of endpoint time, now scaled to zero.

## Corpus growth (2026-11): 3 documents -> 24 instruments, ~4,200 chunks

Added 8 regulations/directives (LED 2016/680, EUDPR 2018/1725, DSA, DMA,
Data Governance Act, Data Act, NIS 2, AI Act) and 13 CJEU judgments
(Schrems I/II, Digital Rights Ireland, Tele2, La Quadrature du Net, Breyer,
Wirtschaftsakademie, Fashion ID, Planet49, Nowak, Osterreichische Post,
Meta v Bundeskartellamt, SCHUFA). Every CELEX number was verified against
CELLAR and every parse checked for gap-free article/paragraph numbering
before indexing. Things that changed or bit:

- **Newer judgments use different markup.** From ~2020 on, paragraph
  numbers are `<p class="coj-count">N</p>` cells (no `id="pointN"`; bullets
  are a bare dash). `parse_judgment` now falls back to this form; real
  excerpt fixture `tests/fixtures/cjeu_311_18_excerpt.xhtml`. Six of the 13
  judgments parsed to zero units before this.
- **Alias matching is whole-word now** (longest alias first) because short
  aliases like "DSA" would otherwise substring-match inside other words.
  Joined cases carry extra case numbers in `config.EXTRA_CASE_NUMBER_ALIASES`.
- **Old directives are not in the corpus:** 95/46 and 2002/58 (ePrivacy)
  return 404 for `application/xhtml+xml` (only plain `text/html`, which the
  ELI parser cannot read). Recitals and annexes are not parsed either
  (articles only) - a real gap for an "AI lawyer".
- **The frozen train/held-out split still covers only the original three
  documents.** `split.compute_split` assumes GDPR-or-judgment and would
  mis-handle new legislation; do NOT regenerate `held_out_provisions.json`
  (it would silently move provisions v1/v2 were trained on into the eval
  set). New instruments are simply "never seen by the adapters" and are
  evaluated with `eval/questions_unseen.jsonl` (25 natural-language
  questions, gold checked against chunk text). A retrain on the full
  corpus needs a new split computed per document type first.
- **Reranker is optional.** On the unseen set it adds ~0.04-0.05 precision
  and one retrieval hit (n=25, noisy), so `sentence-transformers` moved
  to the `rerank` extra; base install has no torch and CI is light.
- **Generation `max_tokens` raised 1500 -> 3000:** the adapter copies long
  judgment paragraphs verbatim and one answer was cut off mid-JSON.

Unseen-instrument results (n=25): stock 7B 0.71/0.80 (precision/recall,
reranked), v1 adapter 0.88/0.88 reranked, 0.83/0.84 without reranker
(one of those 25 was the max_tokens cut-off above). Result files:
`data/finetune/unseen_*.json`. Operational gotcha: a scale-to-zero
Endpoint answers `503` until warm even after `resume()` reports running;
send a throwaway request and wait for success before an eval, or every
question errors.

## Corpus v3 and the v3 retrain (2026-11): 31 instruments, 6,872 chunks

Added recitals (1,387) and annexes (57) as chunk types; the Treaties (TEU,
TFEU) and the Charter (`12016M/TXT`, `12016E/TXT`, `12016P/TXT` - the slash in
the CELEX is percent-encoded for the URL and flattened to `_` on disk), plus
Directives 95/46, 2002/58 (ePrivacy), 2000/31 (e-Commerce) and 2001/29
(InfoSoc). CELLAR serves these in two older layouts, each with its own parser
and a real-excerpt test: the legacy OJ layout (`class="ti-art"`, treaties and
Charter; `parse_legacy_oj`) and unclassed plain HTML (pre-2010 directives,
fetched via the `text/html` fallback in `cellar.fetch`; `parse_plain_html`).
`parse()` picks the parser from the markup. Article counts were checked
against the real instruments (TEU 55, TFEU 358, Charter 54, 95/46 34, ...).
Treaty parsing stops at the Protocols (their "Article 1..." numbering repeats).
`DenseIndex.build(reuse=...)` re-embeds only new/changed chunks.

**New split (`finetune/split.py`).** The original three documents' split is
frozen (verified: all 147 held-out and 422 train ids unchanged; the previous
file is kept as `held_out_provisions_v1.json`). New legislation holds out
every 5th article/recital/annex part; every provision named by
`eval/questions_unseen.jsonl` is held out, and the 11 judgments it names are
held out *entirely*. Known limit: held-out wording can still reach training
through quotation (judgments quote GDPR/ePrivacy recitals verbatim; EUDPR/LED
repeat GDPR wording) - a held-out chunk is never a source or target, but its
text is not guaranteed unseen. Verified by matching held-out text against
the training prompts.

**v3 training set:** 1,680 examples (1,400 positive stratified ~75 per
instrument with training chunks + 280 refusals), questions use each chunk's own
citation ("What does Recital 47 GDPR state?"), `--no-rerank` distractors, real
Qwen token lengths measured (max 3,712, none over the 5,120 limit), 0 dangling
claims. Only 3 of the 15 judgments contribute training paragraphs; the rest are
the unseen-eval cases.

**Decision rule for promoting v3 (written before seeing results):** v3
becomes the default only if its precision and recall are each within 0.03 of
v1's on every set/part, re-run on the current index with the same reranker
setting.

**Outcome: v3 FAILED the rule; v1 remains the default.** Final run
(`data/finetune/final_{base,v1,v3}_{heldout,unseen}.json`, 6,872-chunk index,
no reranker, stock-7B judge; P = citation precision, R = recall):

| | 42 templated (P/R) | 25 NL all | 25 NL legislation (13) | 25 NL judgments (12) |
|---|---|---|---|---|
| stock 7B | .92/.98 | .63/.84 | .43/.77 | .85/.92 |
| v1 | 1.00/1.00 | .76/.76 | .69/.69 | .82/.83 |
| v3 | .95/.95 | .56/.56 | .46/.46 | .67/.67 |

Honest reading: v1 is more *precise* than the stock 7B on legislation but its
*recall* is below the stock model's (cites fewer, sometimes none, of the right
sources) and on judgments it is no better - the adapter bought citation
discipline, not coverage. v3, trained on 1,680 examples over 19 instruments,
regressed on everything and is below the stock model on the NL set. Untested
hypothesis: every training question names its target provision ("What does
Recital 47 GDPR state?"), so the model learned to match a named reference
instead of choosing among sources for a natural question; v1 had the same
weakness but less data to overfit. Caveats: the 42-set uses "paragraph N of
C-xxx" phrasing v1 was trained on; n=25 is noisy; retrieval surfaced the gold
source for only 10/13 legislation questions (0.77), so retrieval caps those
results. v3 repos (`iPwnds/lexeuropa-qwen3b-{lora,merged}-v3`) exist and are
deliberately private. Training cost of the round: ~71 min on l4x1 plus
endpoint time for the evals.

**Next steps, in order of expected payoff:** (1) retrieval for natural
legislation questions - the ELI parser drops article titles (`eli-title`),
indexing them would give BM25/dense something to match; (2) natural-language
training questions that do not name the provision (no model-written text:
e.g. derive from article titles/recital openings); (3) retrain only after (1)
and (2), judged by the same pre-declared rule.

## Round 2 (2026-11): retrieval fixes, v4 retrain, and the credit wall

**Retrieval (kept, measured with `eval/retrieval_eval.py`, no generation/billing).**
Diagnosis of the 25-set misses: the top 8 were all *recitals* of the right
instrument (they discuss topics in prose); the operative article ranked 8th-89th.
Fixes (cap since lowered 3 -> 1, see the sweep below): (1) article headings are now captured (`Chunk.title`, ELI `eli-title`
and the two legacy layouts) and indexed with the text via `Chunk.index_text`
(display text stays verbatim); (2) `retrieve()` searches deeper (40) and caps
recitals in the final list (`LEXEUROPA_MAX_RECITALS`, default 3, `_cap_recitals`).
hit@8 on the 25-set: 0.88 -> 0.96 (titles alone: 0.88, MRR .755; cap: .96,
MRR .771); gold is always inside the top-200 pool, so the remaining miss
(LED Art. 4) is ranking. Caveat: the cap value was chosen while looking at
this same small set. Stock 7B on the same set improved 0.63/0.84 -> 0.69/0.88.

**Cap validation (offline, `LEXEUROPA_EMBED_BACKEND=local`, free).** The cap
of 3 was first chosen on the 25-set, so 34 fresh natural-language questions
(`eval/questions_retrieval.jsonl`, gold checked against chunk text) were
written to validate it. hit@8 / MRR on the fresh 34 by recital cap: none .85 /
.565, 1 **.91 / .586**, 2 .88 / .572, 3 .88 / .570, 5 .85 / .565; on the
original 25 every cap 1-5 gives .96 (MRR best at 1: .795). Default is now 1.
A further per-judgment cap (max 3 or 2 paragraphs of one case) changed
nothing and was removed. Remaining misses on the fresh set: EUDPR Art. 57
(neighbouring EDPS articles retrieved) and two old-directive questions
(95/46 Art. 7, ePrivacy Art. 5) where judgment paragraphs quoting the
statute outrank it. `LEXEUROPA_EMBED_BACKEND=local` runs the same bge model
on CPU (needs the `rerank` extra, ~1.3 GB of weights) and reproduced the
HF-API numbers exactly (hit@8 .96, MRR .771), so retrieval can be evaluated
with no HF credits.

**v4 training data (`build_dataset.py`).** Hypothesis for v3's failure: every
question named its target provision and the target cited one source. v4 adds
*topic* examples: question templated from the article's own heading ("What
does the AI Act say about human oversight?", no provision named), sources = all
paragraphs of that article + real retrieved distractors, target = one claim
group per paragraph citing that paragraph's own handle (avg 2.9 paragraphs).
1,402 examples: 539 topic + 630 reference + 233 refusals; max 3,484 Qwen
tokens; 0 gold provisions from the held-out set. Trained 176 steps / 60 min on
l4x1, loss 0.0055 (`iPwnds/lexeuropa-qwen3b-{lora,merged}-v4`, private).
Dataset revisions on the Hub: `v2` (506), `v3` (1,680), `main` (v4).

**Results (current retrieval, no reranker, stock-7B judge; P/R; 1 JSON-parse
error each in the 25-set, excluded):**

| | 42 templated | 25 NL (24 valid) | legislation (12) | judgments (12) |
|---|---|---|---|---|
| stock 7B | .90/.98 | .69/.88 | .55/.83 | .83/.92 |
| v4 | .98/.98 | .68/.75 | .60/.75 | .75/.75 |

v4 is far better than v3 (.56/.56, older retrieval) but still has lower recall
than the stock model on natural questions. **v1 was NOT re-run on the new
retrieval, so the pre-declared v4-vs-v1 rule could not be applied; v1 stays
the default by default.** Round-1 v1 numbers (older retrieval, same sets) were
1.00/1.00 and .76/.76 and are not comparable to the table above.

**Why v1 was not re-run: the Hugging Face account ran out of credits.**
Endpoint resume now returns `Forbidden: Payment method required for namespace:
iPwnds` and Inference Providers (query embeddings, judge, router generation)
return `HTTP 402 ... Purchase pre-paid credits`. The endpoint was left scaled
to zero (serving v1, no cost). **Until credits are added the live system cannot
run** (retrieval needs embedding calls; generation needs the endpoint). Adding
credits is the user's call; to finish the comparison afterwards: warm the
endpoint on v1 (send a real request and wait for success - `resume()` reporting
"running" is not enough), then `eval/run_eval.py` on both sets with
`LEXEUROPA_USE_RERANKER=0`, into `data/finetune/final2_v1_*.json`.

**Honest bottom line:** across three adapters the fine-tune consistently
improves citation *precision* and format discipline, but never clearly beats the
stock Qwen2.5-7B on natural questions about unseen law, and its recall is
lower. The most valuable remaining work is not more training but retrieval
quality and, if the goal is "own models", a stronger base model than 3B.

## Round 3 (2026-11): credits restored, like-for-like comparison, fallback

HF credits were topped up ($10) and the payment method accepted `resume()`.
Everything below ran on the same retrieval (cap 1, titles, no reranker, local
query embeddings), the fixed parser and the stock-7B judge; files
`data/finetune/final3_*.json`.

| generator | 42 templated | 25 NL | legislation(13) | judgments(12) |
|---|---|---|---|---|
| v1 + fallback (default) | .95/.95 | .86/.88 | .81/.85 | .91/.92 |
| v1 alone | .95/.95 | .82/.84 | .81/.85 | .82/.83 |
| stock Qwen2.5-7B | .91/.98 | .67/.88 | .57/.92 | .78/.83 |
| stock Qwen2.5-72B | .93/1.00 | .72/.88 | .59/.83 | .85/.92 |
| stock Qwen3-235B | .91/.98 | .66/.92 | .53/.92 | .80/.92 |
| stock DeepSeek-V3 | .89/.98 | .68/.88 | .58/.92 | .79/.83 |
| Llama-3.3-70B | unusable: 13/25 JSON-format failures | | | |

This **supersedes the earlier "the fine-tune never clearly beats stock" reading**
(which compared v1's round-1 numbers on older retrieval with round-2 numbers
for other models). Like-for-like, v1 has the best precision of everything
tried (.82 vs .66-.72), recall within a few points of the 72B/7B. Caveat that
matters: the benchmark has ONE gold article per question, so a model that
cites exactly one article wins on precision and larger models that cite
several relevant provisions lose; it scores which sources are cited, not
whether the answer is right. n=25 (one question = 4 points).

**Parser bug found and fixed:** `parse_response` rejected a valid JSON object
followed by trailing text ("Extra data"), which sank the 72B model (32/42
errors) until fixed - it now reads the first complete object
(`raw_decode`). Test: `test_parse_response_ignores_text_after_the_json_object`.

**Fallback generator** (`answer.py`, `config.FALLBACK_LLM_*`): if the primary
model refuses (no cited claim) or its output cannot be parsed, retry once on
Qwen2.5-72B via the router; output carries `generator`/`generator_model` and the
CLI prints a note. Motivated by a live smoke test where v1 refused "Can a
company send marketing emails without consent?" with ePrivacy Art. 13(1)-(4)
retrieved. It fired on 1/25 NL questions (fixing a zero) and 0/42 templated.
Live smoke test also showed v1 quoting Recital 65 instead of AI Act Art. 9.

Cost notes: l4x1 $0.80/h, l40sx1 $1.80/h, a10g-large $1.50/h, a100-large
$2.50/h (`hf jobs hardware`). A bigger-base fine-tune was NOT run: the big
stock models do not beat v1 on this benchmark, so there is no evidence the
extra spend (~$5-10) would help. Endpoint is scaled to zero, serving v1.
A run that hangs on a provider (DeepSeek-V3 did, for over an hour) needs a
hard limit: `perl -e 'alarm N; exec @ARGV' ...` (macOS has no `timeout`).

## Web interface (2026-11)

`python3 -m lexeuropa.cli serve [--port 8765] [--no-browser]` (`lexeuropa/web.py`
+ `lexeuropa/web_static/index.html`; stdlib only, no framework). Routes: `GET /`,
`GET /api/info` (31 instruments, chunk counts, the aliases behind the "focus on"
selector), `POST /api/ask` (`{"question", "instrument"}` -> `answer_question`
plus `elapsed_s`). Safety, all covered by `tests/test_web.py`: binds to
127.0.0.1 only (there is no login); rejects any Host header that is not
localhost/127.0.0.1/[::1] (DNS rebinding); POST must be `application/json` (so a
cross-origin page needs a preflight this server never grants); 16 KB body and
2,000-character question limits; strict Content-Security-Policy; questions are
never logged; one generation at a time (lock). The page inserts model/source
text only via `textContent` (never HTML), keeps history in `localStorage` (15
entries, shortened source text), and says plainly that questions are processed
by Hugging Face-hosted models. `answer.py::_load_indices` is now `lru_cache`d
so the server loads the indexes once. Verified: HTML parses, `node --check` on
the script, API behaviour via curl, 24 server tests. NOT verified: how it looks
in a real browser (no browser tool was available).

**Finding from the first real web request (v1, reranker on):** for "Can a
company send marketing emails without consent?" v1 returned ONE claim, ePrivacy
Art. 13(4) (disguising the sender), labelled SUPPORTED - a correct citation of
the wrong paragraph; the same question in the CLI earlier got a refusal, then the
fallback gave a correct but incomplete answer (it dropped the "own similar
products" and "context of a sale" conditions of Art. 13(2) and put the exception
before the rule). The verifier only checks claim-vs-cited-text, never
claim-vs-question and never completeness. Proposed next step: an answer-relevance
check (does any claim address the question?) that triggers the fallback, plus
prompting for rule-before-exception and full conditions; evaluate on the 25-set
before enabling. Cold start measured at 77 s.

## Interface redesign (2026-11)

The first page (form on top, answer below, legend dominating the left column,
big all-caps pills, card-in-card) was organised around the form. Redesigned
around reading and checking: three regions - a left rail (wordmark, new
question, history with per-item delete, "what it covers" and "how to read"
dialogs), a single reading column (the question as a heading, a one-line verdict
with a status dot, numbered statements set in a serif face, small label with a
distinct glyph per status, reasons shown only for non-supported statements) and a
sticky **Sources panel** (cited passages numbered to match the citation markers,
click a marker to scroll/highlight its passage, "also retrieved" collapsed, link
to EUR-Lex). Under 1100 px the panel becomes a bottom sheet; under 860 px the rail
becomes a drawer. System font stacks only (CSP forbids web fonts): Iowan
Old Style/Charter/Palatino for text, SF/Segoe for UI. Light and dark follow the
OS; `prefers-reduced-motion` and print are handled; `#example` shows a built-in,
hand-written example answer (real Directive text, clearly labelled) with no model
call. Real progress: `answer_question(progress=cb)` reports retrieving /
generating / reviewing / second_opinion / checking and `POST /api/ask` with
`"stream": true` returns them as newline-delimited JSON (a closed tab raises
BrokenPipe, which stops further spend). New pipeline stage: an answer-relevance
check (`verification/relevance.py`, Qwen2.5-72B) after generation; if the
statements do not address the question the fallback is tried (with extra
instructions: rule before exception, all conditions), and `answers_question` /
`relevance_reason` are returned and shown. Tests: `tests/test_web.py`,
`tests/test_relevance.py`, and `tests/ui/test_ui.js` (jsdom, 31 checks, also in
CI). LIMITATION: the agent cannot render it (headless Firefox is blocked in
the sandbox: "Operation not permitted"), so the visuals are designed blind and
iterated from the user's screenshots.

**Second design pass (from the user's screenshot of v2):** the native `<select>`
and default-styled links looked foreign. Replaced with one consistent system:
a custom searchable, grouped, keyboard-accessible **law picker** (combobox +
listbox), one button family (`.btn`, `.primary`, `.ghost`, `.block`), quiet
rail links, a masthead (gold rule, faint ring of 12 EU stars as a watermark),
and **Latin legal maxims with English translations in both margins** of the empty
state (two different ones per visit, vertical type, shown from 1240 px, hidden
in results/print). Inline SVG favicon. `tests/ui/test_ui.js` now 46 checks.

## Relevance check: measured (2026-11)

`verification/relevance.py` asks Qwen2.5-72B (~1.8 s, a fraction of a cent)
whether the answer's statements address the question; "no" triggers the
fallback (with rule-before-exception guidance) and the UI shows a warning.
**On the 25-question set** (v1 + check + fallback): precision .86 / recall .88,
identical to v1 + fallback without it, question for question (25/25 same
scores); the check said "addresses the question" for all 25, i.e. no false
alarms, and also for the 3 answers that scored zero against the single gold
article (those cited other, plausibly relevant provisions, or were retrieval
misses). The 42-question run hit its 30-minute limit (slow HF provider for the
judge model) and produced no result; the fallback never fired on that set
earlier. **Direct test on the motivating failure** (question: "Can a company
send marketing emails without consent?"): Art. 13(4) (disguising the sender) ->
flagged FALSE; an answer about erasure -> FALSE; exception-only answer -> TRUE
(it does not detect incompleteness); rule + exception -> TRUE. Verdict: it
catches wrong-topic answers (not incomplete ones), shows no benefit or harm on
the benchmark because the set contains no such case, and is kept ON. Not
caught by anything yet: an answer that is on topic but omits the main rule or
the exception's conditions - that needs a completeness check against reference
answers (next step). Endpoint left scaled to zero.

## Answer-quality evaluation (2026-11)

`eval/answer_quality.jsonl` (29 questions; 25 in scope with key-point checklists,
4 out-of-corpus traps), `eval/run_quality.py` (grader = Qwen2.5-72B; `--calibrate`
grades synthetic answers made from real provision text: verbatim 0.98, first 40%
0.66, unrelated 0.00), `tests/test_quality_set.py` (structure; every key phrase must
occur in the gold provision when the corpus is present). Baseline (8 sources, no
expansion; `data/finetune/quality_v1.json`, `quality_q72.json`): system mean
completeness .56 / 16% fully complete; stock 72B .62 / 16%; gold provision
retrieved 20/25 for both; traps 4/4 both; contradictions 1 vs 0.

Findings, in order of importance:
- **Retrieval-limited first.** Grading the retrieved text itself (ceiling) gives .72;
  so retrieval caps completeness and generation loses ~.1-.15 more.
- **Whole-article context** (top 3 articles, all paragraphs, up to 14 sources):
  ceiling .83 / 80% fully complete. End to end with the expanded articles FIRST
  (`quality_v1_expand.json`): mean .547 (flat), fully complete 32% (2x), completeness
  when gold retrieved .667 (from .588), citation set P/R .90/.92 (from .86/.88)
  (`final5_v1exp_unseen.json`) - but gold retrieval fell 20 -> 17/25: the expanded
  articles displaced the right article for gdpr_fines, gdpr_access, ai_fines
  (ai_fines then answered from Art. 101 with a confidently wrong 3% / EUR 15M instead
  of Art. 99(3)'s 7% / EUR 35M, labelled supported) and v1 copied only the first
  paragraph it saw for dataact_switching (Art. 25(1) listed first, 1.0 -> 0.0).
  Hence the SHIPPED design appends siblings after the untouched ranked results
  (`expand_articles`; ceiling .83 / 68% fully complete; NOT yet re-measured end to end
  - one run, ~$0.7 of endpoint time, would confirm). `LEXEUROPA_EXPAND_ARTICLES`,
  `LEXEUROPA_MAX_SOURCES`; the dataset builder passes `expand=0`.
- **Dropped:** a "general regime" prior excluding EUDPR/LED unless the question
  names institutions/law enforcement (ceiling .69); a completeness reviewer prompt
  (flagged 23/25 incl. 4/4 complete answers: no discrimination); model size (72B
  only .62).
- **Parallel regimes** (GDPR / EUDPR / LED have look-alike DPO, breach and principles
  articles) crowd each other out and blend in answers; the "Focus on" picker is the
  user-side remedy. 5 of 25 questions miss the gold provision for this or similar
  reasons (gdpr_lawful_bases, gdpr_breach, gdpr_adm, charter_art8, led_principles).
- **Corpus limitation found:** the corpus is the original act text, not consolidated
  (ePrivacy Art. 5(3) is the 2002 "right to refuse"; the current law requires consent
  after Directive 2009/136/EC). The cookie question was therefore left out of the set.
- Judge noise ~2% (one verbatim answer graded .5). n=25 and one run per system: per-
  question scores swing a lot (e.g. -1.0 to +0.67); only the aggregate is meaningful.

Suggested next steps: (1) confirm the append design end to end; (2) retrieval that
resolves parallel regimes (e.g. a regime hint in the answer prompt or a per-instrument
section in the UI); (3) consolidated texts; (4) a v1-style retrain whose targets cite
ALL relevant paragraphs of an article (the v4 topic examples did this) now that the
completeness set exists to judge it.

## Pre-declared rule for promoting v4 on completeness (written before measuring)

v4 (`iPwnds/lexeuropa-qwen3b-*-v4`, private) was trained on whole-article targets
that cite every paragraph, and has never been scored on completeness. It replaces v1
as the default only if, on the SAME retrieval (append-expansion, no reranker, local
embeddings) and the same grader: (a) mean completeness on the 25 in-scope questions is
at least v1's + 0.05, AND (b) citation precision and recall on
`eval/questions_unseen.jsonl` are each within 0.03 of v1's (v1 with the same retrieval),
AND (c) all 4 trap questions are still declined, AND (d) contradictions are no
more than v1's + 1. Otherwise v1 stays. n=25 and one run each, so a smaller gap than
0.05 is treated as noise.

## Confirmation run, grader failure, and the v4 decision (2026-11)

`data/finetune/quality_{v1_append,v4_append}.json`, `final6_{v1_append,v4_append}_unseen.json`.
Same retrieval (append expansion, no reranker, local embeddings), same grader.

| | mean completeness | fully complete | gold retrieved | contradictions | traps | citation P/R (25-set) |
|---|---|---|---|---|---|---|
| v1, 8 sources (before) | .559 | 16% | 20/25 | 1 | 4/4 | .86/.88 |
| v1 + append expansion (shipped) | .528 | 16% | 20/25 | 0 | 4/4 | .84/.84 |
| **v4 + append expansion** | **.751** | **48%** | 20/25 | 3 | 4/4 | .82/.88 |
| stock 72B, 8 sources | .621 | 16% | 20/25 | 0 | 4/4 | - |

**Harness bug found and fixed:** the first v1_append numbers showed .188 because 17 of 25
grading calls hit HTTP 504 from the router and were silently scored as 0. The grader now
retries (4 attempts), `summarize` EXCLUDES failed gradings (reports
`grader_failures_excluded`) instead of counting zeros, and `run_quality.py --regrade FILE`
re-scores saved answers without touching the generator. After re-grading, v1_append is
.528. Lesson: a failed measurement must never look like a bad result. (Tests:
`test_grade_retries_a_flaky_gateway...`, `test_a_failed_grading_is_excluded...`.)

**v4 vs the pre-declared rule** (completeness >= v1 + .05; citation P/R within .03;
traps 4/4; contradictions <= v1 + 1): (a) PASS (+.22), (b) PASS (.82/.88 vs .84/.84),
(c) PASS, (d) **FAIL (3 vs <= 1)** -> **v1 stays the default.** The 3 flags: `ai_fines`
(REAL: correct 7%/EUR 35M plus "also up to EUR 1.5M", which is the fine for EU institutions,
regime blending), `gdpr_fines` (REAL: opens "the sources do not contain information", then
lists AI Act fines before the right GDPR tiers; irrelevant appended articles copied), and
`schrems2` (grader misflag: the note describes an omission). v4 copies every paragraph it is
given, including irrelevant ones, so append expansion of up to 3 articles hurts it.
Whether v4's gain comes from training or from expansion is NOT known (v4 was never run
without expansion on this set).

Next experiment (~$1.5): restrict expansion to articles of the same instrument as the
top-ranked result (regime-consistent), run v4 on the quality set with and without
expansion, apply the same rule. v4 is switched in by pointing the endpoint at
`iPwnds/lexeuropa-qwen3b-merged-v4` and setting `LEXEUROPA_LLM_MODEL` (the endpoint was
restored to v1 and scaled to zero).

## Extending the corpus

Add entries to `lexeuropa/ingest/seed_corpus.py`. Every CELEX number there
was verified against a live CELLAR fetch before being added — do the same
for new entries (`curl -IL -H "Accept: application/xhtml+xml" -H
"Accept-Language: eng" http://publications.europa.eu/resource/celex/<CELEX>`
should return a 303 redirect, not a 404). Add any new instrument's short
name to `INSTRUMENT_ALIASES` in `config.py` if bare "Article N" references
to it should resolve.
