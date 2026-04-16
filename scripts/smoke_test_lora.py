"""Smoke test: can this machine train a LoRA adapter, and how fast?

Run before investing in the full fine-tuned-grader pipeline. Proves the
backend (CUDA / CPU) can load a model, attach a LoRA adapter, complete
training steps with decreasing loss, and save the adapter. Uses a 0.5B
model so it finishes in minutes.

Sweep sequence length to calibrate throughput at realistic inputs.
Real grader prompts are ~4-8k tokens (rubric + task + prior steps +
current step), so the seq=256 default is only a sanity check. Run
again at --max-length 4096 to estimate real-world wall time.

Target environment: WSL2 Ubuntu + CUDA + RTX 5080 (Blackwell sm_120).
Install with a Blackwell-capable PyTorch wheel:
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
    pip install transformers peft datasets accelerate

Usage:
    python smoke_test_lora.py                       # seq 256, quick sanity
    python smoke_test_lora.py --max-length 4096     # realistic seq
    python smoke_test_lora.py --max-length 4096 --batch-size 1

Note: 0.5B in BF16 is ~1GB — fits trivially on a 16GB 5080 without QLoRA.
The real 8B training run will need 4-bit quantization (bitsandbytes +
Unsloth) to fit; this smoke test is only validating the base stack.
"""
from __future__ import annotations

import argparse
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./smoke-lora-out"
ADAPTER_DIR = "./smoke-lora-adapter"


def pick_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def build_sample(i: int, target_tokens: int) -> dict[str, str]:
    """Build a single training example padded with filler to hit target length.

    The filler lets us probe throughput at realistic seq lengths without
    needing the actual grader prompt. Content is irrelevant for timing.
    """
    filler = " ".join([f"prior-step-{j}" for j in range(target_tokens)])
    return {
        "text": (
            "<|im_start|>system\nYou are a step grader.<|im_end|>\n"
            f"<|im_start|>user\nStep {i}: {filler}<|im_end|>\n"
            '<|im_start|>assistant\n{"validity":"PASS","severity":"NONE"}<|im_end|>'
        )
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=32)
    args_cli = ap.parse_args()

    device, backend = pick_device()
    print(f"[smoke] backend: {backend}  device: {device}")
    print(f"[smoke] seq_len: {args_cli.max_length}  batch: {args_cli.batch_size}")
    if backend == "cuda":
        print(f"[smoke] gpu: {torch.cuda.get_device_name(0)}  capability: {torch.cuda.get_device_capability(0)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if backend != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device)

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    samples = [build_sample(i, args_cli.max_length) for i in range(args_cli.num_samples)]
    ds = Dataset.from_list(samples).map(
        lambda x: tok(
            x["text"],
            truncation=True,
            max_length=args_cli.max_length,
            padding="max_length",
        ),
        batched=True,
    )
    ds = ds.map(lambda x: {"labels": x["input_ids"]})

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args_cli.batch_size,
        num_train_epochs=1,
        logging_steps=1,
        bf16=(dtype == torch.bfloat16),
        report_to="none",
        save_strategy="no",
    )

    t0 = time.perf_counter()
    Trainer(model=model, args=args, train_dataset=ds).train()
    elapsed = time.perf_counter() - t0

    model.save_pretrained(ADAPTER_DIR)

    steps = len(ds) // args_cli.batch_size
    it_per_s = steps / elapsed
    # 8B vs 0.5B: ~16x more params. Attention cost also scales, so this
    # estimate is a lower bound on wall-time (upper bound on throughput).
    print(f"[smoke] completed {steps} steps in {elapsed:.1f}s ({it_per_s:.2f} it/s)")
    print(f"[smoke] 8B estimate at seq={args_cli.max_length}: ~{it_per_s / 16:.3f} it/s")
    print(f"[smoke] adapter saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
