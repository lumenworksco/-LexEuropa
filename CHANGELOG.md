# Changelog

All notable changes. Dates are 2026; the project follows [Keep a Changelog](https://keepachangelog.com/) loosely.

## [0.9.0] - 2026-11-10

Feature-complete for EU law as a research assistant. A retrain on the full library (v5) is prepared but not run.

### Added
- 50 more instruments (81 in all, 14,880 chunks): consumer, contract, private international law, competition, employment, equality, whistleblowing, trade secrets, free movement, anti-money-laundering, crypto-assets, copyright, cyber resilience and asylum legislation, plus 23 landmark Court of Justice judgments.
- Consolidated texts: acts amended since publication are read from their latest consolidated version.
- Operative parts (the rulings) of every judgment, shown alongside the judgment paragraphs.
- Repealed-law handling, an institution-only filter, and withholding of statements that do not answer the question.
- Static demo of the interface and a project page on GitHub Pages; issue forms, security policy, citation file, changelog.
- v5 retrain preparation: split, dataset and promotion rule (not trained).

### Changed
- Whole-article expansion chooses the articles with the most ranked paragraphs.
- Default generator is the v4 adapter (`iPwnds/lexeuropa-qwen3b-merged-v4`).

### Measured, tried and rejected
- Query rewriting (unstable) and instrument routing (net loss) are implemented but off by default; see `CLAUDE.md` for the numbers.

## Unreleased

### Added
- `serve-public`: a public demo server where visitors sign in with Hugging Face and their own token pays (built and tested; not deployed, because hosting a Docker Space now needs HF PRO).
- A recorded-answers mode for the static demo, and `scripts/build_gallery.py` to record real answers.

### Fixed
- The hosted fallback, relevance check, grader and citation judge had stopped working when the router dropped the Qwen2.5 models; defaults now point at available Qwen3 Instruct models.

## Earlier work (not tagged)

### 2026-11-09
- v4 adapter promoted; named-law restriction; same-instrument expansion; answer-relevance check with fallback model.
- Redesigned interface: sources panel, law picker, progress streaming, Latin maxims.
- Answer-quality evaluation with checklists and a grader; fixed a harness bug that scored failed gradings as zero.

### 2026-11, first vertical slice
- First vertical slice: CELLAR ingestion, hybrid retrieval, citation verification, a LoRA fine-tune of Qwen2.5-3B, 31 instruments.
