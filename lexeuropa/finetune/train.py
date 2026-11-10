# /// script
# dependencies = [
#   "transformers>=4.46",
#   "trl>=0.12",
#   "peft>=0.13",
#   "accelerate>=1.0",
#   "datasets>=3.0",
#   "huggingface_hub>=0.26",
# ]
# ///
"""LoRA SFT fine-tune of a Qwen2.5-Instruct model on LexEuropa's dataset.

Meant to run via `hf jobs uv run`, e.g.:

    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN train.py \\
        --dataset your-username/lexeuropa-sft \\
        --base-model Qwen/Qwen2.5-3B-Instruct \\
        --output-repo your-username/lexeuropa-qwen3b-lora \\
        --max-steps 20   # smoke run first — see CLAUDE.md

l4x1, not t4-small: a first smoke run on T4 measured 0.115
samples/second — T4 (Turing) has no bf16 tensor cores, so the model ran in
fp32. L4 (Ada) supports bf16 and has more VRAM headroom for the longer
sequences this dataset needs (see --max-length below).

3B, not 7B: this adapter's job is schema adherence, correct-handle
citation among distractors, and refusing when sources don't cover the
question — not deep legal reasoning (the corpus is 3 documents). That's
exactly the failure mode a smaller model showed most on the local Ollama
stack (see CLAUDE.md's "History" section), so it's also where fine-tuning
should help most, and it's the cheaper model to train and serve.

Dataset format is prompt/completion (conversational), not plain messages:
measuring actual tokenized lengths (see CLAUDE.md) showed 98% of examples
exceed TRL's default 1024-token max_length, and Qwen2.5's chat template
has no {% generation %} tags, so assistant_only_loss can't mask the loss
to just the answer. prompt/completion is TRL's template-agnostic way to
get the same effect: build_dataset.py emits {"prompt": [...],
"completion": [...]} instead of a single "messages" list.

The dataset must already exist as an HF dataset repo (upload
data/finetune/sft_dataset.jsonl with `hf upload --type dataset`) — that
upload needs an explicit go-ahead, since it puts data in the user's HF namespace.
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _check_label_masking(trainer: SFTTrainer, tokenizer, raw_completion_text: str) -> None:
    """Fails fast, before spending GPU time, if loss isn't actually
    masked to the completion. Decodes only the tokens where labels !=
    -100 for one example and checks the real completion JSON appears in
    them — not just that *something* is unmasked."""
    example = trainer.train_dataset[0]
    input_ids = example["input_ids"]
    labels = example["labels"]
    kept = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    if not kept:
        raise RuntimeError("label-masking check failed: every token was masked out (labels all -100)")
    decoded = tokenizer.decode(kept)
    print(f"label-masking check — decoded unmasked tokens:\n{decoded}\n")
    if raw_completion_text.strip() not in decoded:
        raise RuntimeError(
            "label-masking check failed: the real completion JSON does not appear in the "
            f"decoded unmasked tokens. Expected to find:\n{raw_completion_text!r}\nGot:\n{decoded!r}"
        )
    print("label-masking check passed: loss is computed on the completion JSON, not the prompt.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HF dataset repo id, e.g. user/lexeuropa-sft")
    parser.add_argument("--dataset-revision", default=None, help="branch/tag of the dataset repo (v5 is uploaded as revision v5)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-repo", required=True, help="HF model repo id to push the adapter to")
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps for a smoke run")
    parser.add_argument("--max-length", type=int, default=5120, help="covers the measured max (4999) with margin")
    parser.add_argument("--num-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split="train", revision=args.dataset_revision)
    print(f"loaded {len(dataset)} training examples, columns: {dataset.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir="./lexeuropa-lora-out",
        max_length=args.max_length,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        push_to_hub=True,
        hub_model_id=args.output_repo,
        hub_private_repo=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"tokenized train dataset size: {len(trainer.train_dataset)} (effective max_length={args.max_length})")
    raw_completion = dataset[0]["completion"][0]["content"]
    _check_label_masking(trainer, tokenizer, raw_completion)

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
