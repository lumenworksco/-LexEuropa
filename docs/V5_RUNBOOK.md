# v5 retrain runbook (needs HF credits; about $4-5)

Everything before this point is already done and verified without credits: the v5 split
(`data/finetune/held_out_provisions_v5.json`), the dataset (`data/finetune/sft_dataset_v5.jsonl`, 2,400
examples: 1,100 topic + 900 reference + 400 refusals, max 4,827 Qwen tokens, no held-out text, no eval question),
and the v4 files preserved as `sft_dataset_v4.jsonl`.

0. **Recalibrate first (about $0.15).** On 2026-11-10 the hosted Qwen2.5 models vanished from the router and the defaults moved to
   Qwen3 Instruct models (see CLAUDE.md, "MODEL DRIFT"). Run `python eval/run_quality.py --calibrate` (verbatim should be about
   .98, first-40% about .66, unrelated about 0) before trusting any grade, and run v4 AND v5 under the new models: the earlier
   v4 numbers used the old judge and grader and cannot be compared with.
0b. **Record the demo gallery** in the same endpoint session: `python scripts/build_gallery.py --resume-endpoint` (about $1).
1. Upload the dataset as revision `v5` of `iPwnds/lexeuropa-sft` (public; the `main` revision stays v4's data).
2. Smoke run: `hf jobs uv run --flavor l4x1 --secrets HF_TOKEN lexeuropa/finetune/train.py --dataset iPwnds/lexeuropa-sft --dataset-revision v5 --output-repo iPwnds/lexeuropa-qwen3b-lora-v5 --max-steps 5`
3. Full run (about 100 min on l4x1, about $1.40): same command without `--max-steps`.
4. Merge on a cheap CPU job: `lexeuropa/finetune/merge_adapter.py --adapter-repo iPwnds/lexeuropa-qwen3b-lora-v5 --output-repo iPwnds/lexeuropa-qwen3b-merged-v5`.
5. Point the endpoint `lexeuropa-eval` at the merged-v5 repo (pass the new repo's `revision`), warm it with a real request, evaluate, scale to zero.
6. Apply the pre-declared promotion rule (docs in CLAUDE.md, "Promotion rule for v5") and promote only if it passes.
