# /// script
# dependencies = [
#   "transformers>=4.46",
#   "peft>=0.13",
#   "accelerate>=1.0",
#   "huggingface_hub>=0.26",
# ]
# ///
"""Merges a LoRA adapter into its base model and pushes the merged, standalone
model — simpler and lower-risk to deploy to an Inference Endpoint than
hoping the serving engine auto-loads a bare PEFT adapter repo correctly.

    hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN merge_adapter.py \\
        --adapter-repo your-username/lexeuropa-qwen3b-lora \\
        --output-repo your-username/lexeuropa-qwen3b-merged
"""
from __future__ import annotations

import argparse

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--output-repo", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_repo)
    peft_config = PeftConfig.from_pretrained(args.adapter_repo)
    base_model_name = peft_config.base_model_name_or_path
    print(f"base model: {base_model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, args.adapter_repo)
    merged = model.merge_and_unload()

    merged.push_to_hub(args.output_repo, private=True)
    tokenizer.push_to_hub(args.output_repo, private=True)
    print(f"pushed merged model to {args.output_repo}")


if __name__ == "__main__":
    main()
