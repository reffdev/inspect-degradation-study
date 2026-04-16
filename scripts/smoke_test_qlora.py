"""QLoRA VRAM smoke test: confirm the overnight config fits in 16GB.

Loads the production model in 4-bit, attaches the production LoRA config,
and runs a handful of full-length training steps at the production seq
length and effective batch. Reports peak CUDA memory so we know whether
overnight will OOM before we kick it off.

Mirrors train_qlora.py defaults. If those change, update here too.

Usage:
    python scripts/smoke_test_qlora.py
    python scripts/smoke_test_qlora.py --max-seq-len 4096 --steps 8
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8,
                    help="Number of optimizer steps to run (each is batch*grad_accum forwards).")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        print("[smoke] no CUDA — aborting", file=sys.stderr)
        return 1

    print(f"[smoke] gpu: {torch.cuda.get_device_name(0)}  "
          f"capability: {torch.cuda.get_device_capability(0)}  "
          f"total_vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"[smoke] model: {args.model}  seq={args.max_seq_len}  "
          f"bs={args.batch_size}  grad_accum={args.grad_accum}  lora_r={args.lora_r}")

    torch.cuda.reset_peak_memory_stats()

    t_load = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    post_load_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke] loaded in {time.perf_counter()-t_load:.1f}s  peak_vram_after_load={post_load_gb:.2f} GB")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Build enough filler examples to feed `steps * batch_size * grad_accum` items.
    n_samples = args.steps * args.batch_size * args.grad_accum
    # Pad each example out to near max_seq_len so we measure worst-case VRAM.
    # Roughly 1 filler token per word; aim for ~seq_len*0.9 words to leave chat-template headroom.
    target_words = int(args.max_seq_len * 0.85)
    filler = " ".join(f"step-word-{i}" for i in range(target_words))
    samples = [{
        "messages": [
            {"role": "system", "content": "You are a step grader. Emit one JSON object."},
            {"role": "user",   "content": f"CONTEXT: {filler}\nGrade the current step."},
            {"role": "assistant", "content": '{"validity":"pass","complexity":"low","dependency":"n/a","severity":null,"is_looping":false}'},
        ]
    } for _ in range(n_samples)]

    def fmt(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
        )}

    ds = Dataset.from_list(samples).map(fmt)
    print(f"[smoke] built {len(ds)} filler samples at ~{target_words} words each")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        args=TrainingArguments(
            output_dir="./smoke-qlora-out",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.steps,
            learning_rate=2e-4,
            logging_steps=1,
            bf16=True,
            optim="adamw_8bit",
            weight_decay=0.0,
            report_to="none",
            save_strategy="no",
            warmup_steps=0,
        ),
    )

    t0 = time.perf_counter()
    stats = trainer.train()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print("")
    print("=" * 60)
    print(f"[smoke] RESULTS")
    print("=" * 60)
    print(f"[smoke] wall: {elapsed:.1f}s for {args.steps} optim steps "
          f"({elapsed/args.steps:.1f}s/step)")
    print(f"[smoke] peak VRAM: {peak_gb:.2f} GB / 16.0 GB "
          f"({peak_gb/16*100:.0f}% of 5080)")
    print(f"[smoke] final loss: {stats.metrics.get('train_loss')}")
    if peak_gb > 15.5:
        print("[smoke] WARNING: peak VRAM dangerously close to 16GB — real run may OOM "
              "on unusually long examples. Consider --max-seq-len 3072 or smaller LoRA rank.")
    elif peak_gb > 14:
        print("[smoke] peak VRAM is tight but safe. Overnight should fit.")
    else:
        print("[smoke] peak VRAM has comfortable headroom. Overnight will fit.")

    # Extrapolate wall time for the real run.
    # Real Nebius data: ~2500 train examples, effective batch 16, 2 epochs
    # -> ~316 optim steps per N.
    per_step = elapsed / args.steps
    est_overnight_hours = (per_step * 316 * 3) / 3600
    print(f"[smoke] {per_step:.1f}s/step")
    print(f"[smoke] projected 3-run overnight (N=3,5,8 × ~316 steps each): {est_overnight_hours:.1f} hours")
    if per_step > 30:
        print(f"[smoke] ABORT: step time {per_step:.0f}s is too slow. Do not kick off overnight.")
        print(f"[smoke]         Investigate before committing: expected <30s/step on 5080.")
        return 2
    print(f"[smoke] step time is acceptable (<30s). Safe to kick off overnight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
