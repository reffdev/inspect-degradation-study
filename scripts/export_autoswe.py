"""Export sanitized Auto-SWE traces for public release.

Creates a JSONL file with one trace per line containing:
- trace_id, task_id, model, source, success
- task_goal (issue title)
- steps: [{index, action, observation, thought, metadata}]
- metadata: stage, issue_type, run_status

Strips: internal IDs, IP addresses, file paths, git tokens,
machine references. Keeps only what's needed to understand
and re-grade the traces.

Usage:
    python scripts/export_autoswe.py auto-swe.db exported_traces.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from inspect_degradation.datasets.autoswe import load_autoswe

def _sanitize_text(text: str) -> str:
    """Remove potentially sensitive patterns from text."""
    # Internal IPs
    text = re.sub(r"http://192\.168\.\d+\.\d+[:\d]*\S*", "http://[REDACTED]", text)
    text = re.sub(r"http://10\.\d+\.\d+\.\d+[:\d]*\S*", "http://[REDACTED]", text)
    # Absolute paths that reveal machine structure
    text = re.sub(r"/opt/swe/[^\s\"']+", "/[WORKDIR]/...", text)
    text = re.sub(r"/home/\w+/[^\s\"']+", "/[HOME]/...", text)
    # Git tokens/credentials
    text = re.sub(r"github_pat_\S+", "[REDACTED_TOKEN]", text)
    text = re.sub(r"ghp_\S+", "[REDACTED_TOKEN]", text)
    text = re.sub(r"sk-or-v1-\S+", "[REDACTED_KEY]", text)
    return text


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/export_autoswe.py <db_path> <output.jsonl>")
        return

    db_path = sys.argv[1]
    output_path = Path(sys.argv[2])

    print(f"Loading traces from {db_path}...")
    traces = load_autoswe(
        db_path,
        granularity="run",
        min_steps=3,
        random_sample=False,  # export all, not a sample
    )
    print(f"Loaded {len(traces)} traces, {sum(len(t.steps) for t in traces)} total steps")

    exported = 0
    with output_path.open("w", encoding="utf-8") as f:
        for trace in traces:
            record = {
                "trace_id": trace.trace_id,
                "task_id": trace.task_id,
                "task_goal": _sanitize_text(trace.task_goal),
                "model": trace.model,
                "source": trace.source,
                "success": trace.success,
                "n_steps": len(trace.steps),
                "metadata": {
                    "stage": trace.metadata.get("stage"),
                    "issue_type": trace.metadata.get("issue_type"),
                    "run_status": trace.metadata.get("run_status"),
                    "framework": "autoswe",
                },
                "steps": [
                    {
                        "index": step.index,
                        "action": _sanitize_text(step.action),
                        "observation": _sanitize_text(step.observation) if step.observation else None,
                        "thought": _sanitize_text(step.thought) if step.thought else None,
                        "model_id": step.metadata.get("model_id"),
                        "duration_ms": step.metadata.get("duration_ms"),
                        "prompt_tokens": step.metadata.get("prompt_tokens"),
                        "completion_tokens": step.metadata.get("completion_tokens"),
                    }
                    for step in trace.steps
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1

    print(f"Exported {exported} traces to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify sanitization
    print("\nVerifying sanitization...")
    text = output_path.read_text(encoding="utf-8")
    checks = {
        "github_pat_": "GitHub PAT tokens",
        "ghp_": "GitHub tokens",
        "sk-or-v1-": "OpenRouter keys",
        "192.168.": "Internal IPs",
        "/opt/swe/": "Server paths",
    }
    clean = True
    for pattern, label in checks.items():
        if pattern in text:
            print(f"  WARNING: {label} found in export!")
            clean = False
    if clean:
        print("  All checks passed — no sensitive data detected")


if __name__ == "__main__":
    main()
