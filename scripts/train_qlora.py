"""QLoRA fine-tune of Qwen2.5-7B-Instruct on step-grading data.

Assumes `train.jsonl` and `val.jsonl` in --data directory with chat-format
{"messages": [...]} records produced by build_training_data.py.

Saves the LoRA adapter + tokenizer to --out on successful completion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Import Unsloth first — it monkey-patches transformers for speed.
    from unsloth import FastLanguageModel  # noqa: E402
    from unsloth.chat_templates import get_chat_template  # noqa: E402
    from datasets import load_dataset  # noqa: E402
    from transformers import TrainingArguments  # noqa: E402
    from trl import SFTTrainer  # noqa: E402

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading {args.model} (4-bit) at max_seq_len={args.max_seq_len}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,       # auto (bf16 on Blackwell)
        load_in_4bit=True,
    )
    if tokenizer.chat_template is None:
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    def fmt(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = load_dataset("json", data_files=str(args.data / "train.jsonl"), split="train").map(fmt)
    val_ds = load_dataset("json", data_files=str(args.data / "val.jsonl"), split="train").map(fmt)
    print(f"[train] train={len(train_ds)} val={len(val_ds)}", flush=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            bf16=True,
            optim="adamw_8bit",
            weight_decay=0.0,
            seed=args.seed,
            report_to="none",
        ),
    )

    stats = trainer.train()
    print(f"[train] done: {stats.metrics}", flush=True)

    adapter_dir = args.out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    with (args.out / "train_metrics.json").open("w") as f:
        json.dump(stats.metrics, f, indent=2)
    print(f"[train] adapter saved to {adapter_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
