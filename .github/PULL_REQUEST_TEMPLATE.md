## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist

- [ ] `python -m pytest tests/ -k "not full_hybrid"` passes (and `cd tests/ui && npm test` if the interface changed)
- [ ] New corpus entries: the CELEX number was verified against a live CELLAR fetch and the parse has no numbering gaps
- [ ] No model-written training data and no fabricated citations or test fixtures (real CELLAR text only)
- [ ] A measured change (retrieval, prompts, models) states the before/after numbers and the question set used
- [ ] `README.md` and `CLAUDE.md`/`AGENTS.md` are updated when behaviour or numbers change
