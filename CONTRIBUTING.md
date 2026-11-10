# Contributing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v -k "not full_hybrid"   # offline, no network or token needed
```

- Tests run against real cached CELLAR fixtures in `tests/fixtures/`, not synthetic text. Keep it that way.
- The live test (`full_hybrid`) needs `HF_TOKEN` in `.env` and skips cleanly without it.
- No model-written training data and no fabricated citations: every claim in a target answer must come from real chunk text.
- Adding a document to the corpus? Verify its CELEX number against a live CELLAR fetch first (see `CLAUDE.md`, "Extending the corpus").
- Architecture notes, design decisions and known limitations live in `CLAUDE.md` (mirrored in `AGENTS.md`).
- Pull requests use a checklist (tests, verified CELEX numbers, no model-written training data, before/after numbers for measured changes). Be kind: the organisation's code of conduct applies.
- The project page and static demo live in `site/` and are assembled by `python scripts/build_site.py` (deployed by the Pages workflow). After the corpus changes, run `python scripts/build_site.py --refresh-info` and commit `site/demo/info.json`.
