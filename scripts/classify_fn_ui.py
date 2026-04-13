"""Web UI for classifying grader false negatives by hindsight-dependence.

Shows each false negative with ONLY prior context and the current step.
No future steps are visible. For each step, classify:
  - "decision_time": the error is detectable without future context
  - "hindsight": the error is only identifiable by seeing what happened next

This decomposes the 73% FNR into construct mismatch vs actual grader failure.

Usage:
    python scripts/classify_fn_ui.py
    python scripts/classify_fn_ui.py --trail-root /path/to/trail-benchmark/benchmarking
    python scripts/classify_fn_ui.py --n 100 --seed 123

Opens http://localhost:8766 in your browser. Classifications are saved
to results/fn_classifications.jsonl as you go.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from inspect_degradation.datasets.trail import load_trail
from inspect_degradation.store import GradedTraceStore
from inspect_degradation.validation.agreement import pair_grades

STUDY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIL_ROOT = Path(r"E:\Projects\zerg\trail-benchmark\benchmarking")
PORT = 8766


def _load_false_negatives(trail_root: Path, n: int, seed: int) -> tuple[list[dict], dict]:
    """Load MEDIUM+ false negatives with context."""
    import random

    corpus = load_trail(trail_root)
    predicted = GradedTraceStore(
        STUDY_ROOT / "results/phase1/minimax.cache.jsonl"
    ).load_all()
    pairs = pair_grades(predicted, corpus.reference)

    # Filter to MEDIUM+ false negatives
    fn_pairs = []
    for p in pairs:
        ref = p.reference
        if ref.validity.value != "fail":
            continue
        if ref.severity is None or ref.severity.value not in ("medium", "high"):
            continue
        if p.predicted.validity.value not in ("pass", "neutral"):
            continue
        fn_pairs.append(p)

    random.seed(seed)
    sample = random.sample(fn_pairs, min(n, len(fn_pairs)))

    # Build trace lookup for context
    trace_lookup = {t.trace_id: t for t in corpus.traces}

    # Exclude final steps: no future context means the grader and
    # annotator had the same information, so a miss is a genuine
    # grader failure, not a construct mismatch.
    sample = [
        p for p in sample
        if p.trace_id in trace_lookup
        and p.step_index < len(trace_lookup[p.trace_id].steps) - 1
    ]

    items = []
    for pair in sample:
        trace = trace_lookup.get(pair.trace_id)
        if trace is None:
            continue

        # Skip if this is the last step (belt-and-suspenders)
        if pair.step_index >= len(trace.steps) - 1:
            continue

        prior = trace.prior(pair.step_index)
        current = trace.steps[pair.step_index] if pair.step_index < len(trace.steps) else None

        items.append({
            "trace_id": pair.trace_id,
            "step_index": pair.step_index,
            "trail_severity": pair.reference.severity.value,
            "grader_label": pair.predicted.validity.value,
            "task_goal": trace.task_goal or "",
            "prior_steps": [
                {
                    "index": s.index,
                    "thought": s.thought or "",
                    "action": s.action or "",
                    "observation": s.observation or "",
                }
                for s in prior
            ],
            "current_step": {
                "index": current.index if current else pair.step_index,
                "thought": current.thought or "" if current else "",
                "action": current.action or "" if current else "",
                "observation": current.observation or "" if current else "",
            },
            "future_steps": [
                {
                    "index": s.index,
                    "thought": s.thought or "",
                    "action": s.action or "",
                    "observation": s.observation or "",
                }
                for s in trace.steps[pair.step_index + 1:]
            ],
            "total_steps": len(trace.steps),
            "trail_categories": (
                pair.reference.raw.get("trail_categories", [])
                if pair.reference.raw else []
            ),
            "trail_dependency": (
                pair.reference.dependency.value
                if pair.reference.dependency else None
            ),
        })

    return items, {"total_fn": len(fn_pairs), "sampled": len(items)}


def _load_existing(path: Path) -> dict:
    labels = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = f"{obj['trace_id']}:{obj['step_index']}"
                    labels[key] = obj
                except (json.JSONDecodeError, KeyError):
                    continue
    return labels


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>False Negative Classifier</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
.header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }
.header h1 { font-size: 16px; color: #58a6ff; }
.header .sub { font-size: 12px; color: #8b949e; margin-top: 2px; }
.progress { color: #8b949e; font-size: 14px; }
.nav { display: flex; gap: 8px; align-items: center; }
.nav button { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.nav button:hover { background: #30363d; border-color: #8b949e; }
.container { max-width: 1200px; margin: 0 auto; padding: 16px 24px; }
.info-bar { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.info-bar h2 { font-size: 14px; color: #58a6ff; margin-bottom: 8px; }
.meta { display: flex; gap: 20px; font-size: 13px; color: #8b949e; flex-wrap: wrap; }
.meta .fail { color: #f85149; }
.meta .neutral { color: #d29922; }
.task-goal { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-top: 8px; font-size: 13px; max-height: 150px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; color: #8b949e; }
.instruction { background: #1c1e26; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.instruction h3 { font-size: 13px; color: #f0883e; margin-bottom: 8px; }
.instruction p { font-size: 13px; color: #8b949e; line-height: 1.6; }
.instruction strong { color: #c9d1d9; }
.step-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 8px; overflow: hidden; }
.step-card.current { border-color: #f85149; border-width: 2px; }
.step-card.prior { opacity: 0.7; }
.step-header { display: flex; justify-content: space-between; padding: 6px 12px; background: #21262d; font-size: 12px; }
.step-tag { font-weight: 600; color: #58a6ff; }
.step-tag.current-tag { color: #f85149; }
.step-body { padding: 10px 12px; font-size: 13px; }
.step-body .block { margin-bottom: 8px; }
.step-body .label { font-size: 11px; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px; margin-bottom: 2px; }
.step-body .text { white-space: pre-wrap; word-break: break-word; max-height: 400px; overflow-y: auto; background: #0d1117; border: 1px solid #21262d; border-radius: 4px; padding: 6px 8px; line-height: 1.5; }
.step-card.current .step-body .text { max-height: 600px; }
.step-body .text.thought { color: #d2a8ff; }
.step-body .text.action { color: #c9d1d9; }
.step-body .text.obs { color: #7d8590; }
.classify-panel { background: #161b22; border: 2px solid #30363d; border-radius: 8px; padding: 20px; margin-top: 16px; text-align: center; }
.classify-panel h3 { font-size: 14px; color: #58a6ff; margin-bottom: 4px; }
.classify-panel .question { font-size: 13px; color: #8b949e; margin-bottom: 16px; }
.classify-btns { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
.classify-btn { padding: 12px 28px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; border: 2px solid; transition: all 0.15s; }
.classify-btn.decision { background: #1a1e2e; border-color: #f85149; color: #f85149; }
.classify-btn.decision:hover { background: #f85149; color: #fff; }
.classify-btn.hindsight { background: #1a1e2e; border-color: #3fb950; color: #3fb950; }
.classify-btn.hindsight:hover { background: #3fb950; color: #fff; }
.classify-btn.skip { background: #1a1e2e; border-color: #484f58; color: #484f58; }
.classify-btn.skip:hover { background: #484f58; color: #fff; }
.classify-btn.selected { opacity: 1; }
.classify-btn.selected.decision { background: #f85149; color: #fff; }
.classify-btn.selected.hindsight { background: #3fb950; color: #fff; }
.result-bar { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; margin-top: 16px; }
.result-bar .bar { height: 24px; border-radius: 4px; display: flex; overflow: hidden; margin: 8px 0; }
.bar-dt { background: #f85149; }
.bar-hs { background: #3fb950; }
.result-text { font-size: 13px; color: #8b949e; }
.kbd { background: #21262d; border: 1px solid #30363d; border-radius: 3px; padding: 1px 6px; font-family: monospace; font-size: 12px; color: #c9d1d9; }
.fmt-thought { color: #d2a8ff; font-style: italic; }
.fmt-code { background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 6px 8px; font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; white-space: pre-wrap; color: #e6edf3; display: block; margin: 4px 0; overflow-x: auto; }
.fmt-obs { color: #7d8590; border-left: 3px solid #30363d; padding-left: 8px; margin: 4px 0; }
.fmt-json { background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 6px 8px; font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; white-space: pre-wrap; color: #7ee787; display: block; margin: 4px 0; max-height: 200px; overflow-y: auto; }
.fmt-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px; margin-bottom: 2px; }
.fmt-label.thought-label { color: #d2a8ff; }
.fmt-label.code-label { color: #79c0ff; }
.fmt-label.obs-label { color: #7d8590; }
.fmt-section { margin-bottom: 6px; }
.fmt-diff { background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 6px 0; margin: 6px 0; font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; overflow-x: auto; }
.fmt-diff-line { white-space: pre; padding: 0 10px; line-height: 1.4; }
.fmt-diff-add { background: rgba(46, 160, 67, 0.15); color: #7ee787; border-left: 3px solid #2ea043; }
.fmt-diff-del { background: rgba(248, 81, 73, 0.15); color: #ffa198; border-left: 3px solid #da3633; }
.fmt-diff-hunk { color: #a5a5ff; background: rgba(80, 80, 160, 0.1); padding: 2px 10px; font-weight: 600; }
.fmt-diff-file { color: #d2a8ff; font-weight: 700; padding: 2px 10px; }
.fmt-diff-ctx { color: #8b949e; padding-left: 10px; }
.fmt-collapse { background: #161b22; border: 1px dashed #30363d; border-radius: 4px; padding: 4px 10px; margin: 4px 0; font-size: 11px; color: #8b949e; cursor: pointer; font-family: 'Consolas', monospace; }
.fmt-collapse:hover { color: #c9d1d9; background: #1c2028; }
.fmt-collapse.open { background: #0d1117; color: #c9d1d9; }
.fmt-task-block { background: #0d1117; border-left: 3px solid #58a6ff; padding: 8px 12px; margin: 6px 0; white-space: pre-wrap; color: #c9d1d9; }
.fmt-task-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #58a6ff; margin-bottom: 4px; }
.fmt-facts-block { background: #0d1117; border-left: 3px solid #d29922; padding: 8px 12px; margin: 6px 0; white-space: pre-wrap; color: #8b949e; font-size: 12px; }
.fmt-facts-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #d29922; margin-bottom: 4px; }
.fmt-reply-block { background: #0d1117; border-left: 3px solid #3fb950; padding: 8px 12px; margin: 6px 0; white-space: pre-wrap; color: #c9d1d9; }
.fmt-reply-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #3fb950; margin-bottom: 4px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>False Negative Classifier</h1>
    <div class="sub">Classify grader misses by hindsight-dependence</div>
  </div>
  <div class="nav">
    <button onclick="goPrev()">Prev</button>
    <span class="progress" id="progressText"></span>
    <button onclick="goNext()">Next</button>
  </div>
</div>

<div class="container" id="main"></div>

<script>
let ITEMS = [];
let LABELS = {};
let current = 0;

async function init() {
  const resp = await fetch('/api/data');
  const data = await resp.json();
  ITEMS = data.items;
  LABELS = data.labels;
  render(0);
}

function esc(s) {
  if (!s) return '';
  var d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function formatTrace(raw) {
  if (!raw) return '';
  var s = raw;

  // Try to detect and parse JSON blobs
  try {
    var parsed = JSON.parse(s);
    if (typeof parsed === 'object') {
      return formatStructured(parsed);
    }
  } catch(e) {}

  // Try to detect memory_step JSON wrapper
  if (s.indexOf('{"memory_step"') === 0 || s.indexOf('{"role"') === 0) {
    try {
      var obj = JSON.parse(s);
      return formatStructured(obj);
    } catch(e) {}
  }

  // If this looks like agent boilerplate (system prompt + tools + task),
  // use the smart agent formatter
  if (s.indexOf('You are a world expert') !== -1 ||
      s.indexOf('You are an expert assistant') !== -1 ||
      (s.indexOf('Here is your task:') !== -1 && s.indexOf('tools:') !== -1)) {
    return formatAgentContent(s);
  }

  // Parse Thought/Code/Observation blocks
  s = esc(s);

  // Detect and format diff blocks (unified diff format)
  // A diff starts with "--- a/" or "--- " and has +/- lines
  var NL = String.fromCharCode(10);
  var lines = s.split(NL);
  var out = [];
  var inDiff = false;
  var diffBuf = [];

  function flushDiff() {
    if (diffBuf.length === 0) return;
    var html = '<div class="fmt-diff">';
    for (var i = 0; i < diffBuf.length; i++) {
      var line = diffBuf[i];
      var cls = 'fmt-diff-ctx';
      if (line.indexOf('+++') === 0 || line.indexOf('---') === 0) {
        cls = 'fmt-diff-file';
      } else if (line.indexOf('@@') === 0) {
        cls = 'fmt-diff-hunk';
      } else if (line.charAt(0) === '+') {
        cls = 'fmt-diff-add';
      } else if (line.charAt(0) === '-') {
        cls = 'fmt-diff-del';
      }
      out.push('<div class="fmt-diff-line ' + cls + '">' + (line || ' ') + '</div>');
    }
    out.push('</div>');
    diffBuf = [];
  }

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var isDiffStart = line.indexOf('--- a/') === 0 || line.indexOf('--- /') === 0 ||
                      (line.indexOf('diff --git') === 0);
    var isDiffLine = line.indexOf('+++') === 0 || line.indexOf('---') === 0 ||
                     line.indexOf('@@') === 0 ||
                     line.charAt(0) === '+' || line.charAt(0) === '-';

    if (!inDiff && isDiffStart) {
      inDiff = true;
      diffBuf.push(line);
    } else if (inDiff && (isDiffLine || line.charAt(0) === ' ' || line === '')) {
      // In diff: accept +/- lines, context lines (starting with space), and blank lines
      diffBuf.push(line);
      // End diff if next line looks non-diff
      if (i + 1 < lines.length) {
        var next = lines[i + 1];
        var nextIsDiff = next.indexOf('+++') === 0 || next.indexOf('---') === 0 ||
                         next.indexOf('@@') === 0 || next.charAt(0) === '+' ||
                         next.charAt(0) === '-' || next.charAt(0) === ' ' || next === '';
        if (!nextIsDiff) {
          // Flush diff buffer since next line breaks pattern
          out.push('<div class="fmt-diff">');
          for (var j = 0; j < diffBuf.length; j++) {
            var dline = diffBuf[j];
            var dcls = 'fmt-diff-ctx';
            if (dline.indexOf('+++') === 0 || dline.indexOf('---') === 0) {
              dcls = 'fmt-diff-file';
            } else if (dline.indexOf('@@') === 0) {
              dcls = 'fmt-diff-hunk';
            } else if (dline.charAt(0) === '+') {
              dcls = 'fmt-diff-add';
            } else if (dline.charAt(0) === '-') {
              dcls = 'fmt-diff-del';
            }
            out.push('<div class="fmt-diff-line ' + dcls + '">' + (dline || ' ') + '</div>');
          }
          out.push('</div>');
          diffBuf = [];
          inDiff = false;
        }
      }
    } else {
      if (inDiff) {
        // Flush and exit diff mode
        out.push('<div class="fmt-diff">');
        for (var j = 0; j < diffBuf.length; j++) {
          var dline = diffBuf[j];
          var dcls = 'fmt-diff-ctx';
          if (dline.indexOf('+++') === 0 || dline.indexOf('---') === 0) {
            dcls = 'fmt-diff-file';
          } else if (dline.indexOf('@@') === 0) {
            dcls = 'fmt-diff-hunk';
          } else if (dline.charAt(0) === '+') {
            dcls = 'fmt-diff-add';
          } else if (dline.charAt(0) === '-') {
            dcls = 'fmt-diff-del';
          }
          out.push('<div class="fmt-diff-line ' + dcls + '">' + (dline || ' ') + '</div>');
        }
        out.push('</div>');
        diffBuf = [];
        inDiff = false;
      }
      out.push(line);
    }
  }
  // Flush any remaining diff
  if (diffBuf.length > 0) {
    out.push('<div class="fmt-diff">');
    for (var j = 0; j < diffBuf.length; j++) {
      var dline = diffBuf[j];
      var dcls = 'fmt-diff-ctx';
      if (dline.indexOf('+++') === 0 || dline.indexOf('---') === 0) {
        dcls = 'fmt-diff-file';
      } else if (dline.indexOf('@@') === 0) {
        dcls = 'fmt-diff-hunk';
      } else if (dline.charAt(0) === '+') {
        dcls = 'fmt-diff-add';
      } else if (dline.charAt(0) === '-') {
        dcls = 'fmt-diff-del';
      }
      out.push('<div class="fmt-diff-line ' + dcls + '">' + (dline || ' ') + '</div>');
    }
    out.push('</div>');
  }
  s = out.join(NL);

  // Code blocks: ```py ... ``` (use RegExp constructor to avoid Python escape issues)
  var codeRegex = new RegExp('```(?:py|python)?' + String.fromCharCode(10) + '([\\s\\S]*?)```', 'g');
  s = s.replace(codeRegex, function(m, code) {
    return '<div class="fmt-label code-label">Code</div><code class="fmt-code">' + code.trim() + '</code>';
  });

  // Thought: / Code: / Observation: labels (line-start, no backslashes needed)
  s = s.replace(new RegExp('(^|' + String.fromCharCode(10) + ')Thought:', 'g'),
    '$1<div class="fmt-label thought-label">Thought</div>');
  s = s.replace(new RegExp('(^|' + String.fromCharCode(10) + ')Code:', 'g'),
    '$1<div class="fmt-label code-label">Code</div>');
  s = s.replace(new RegExp('(^|' + String.fromCharCode(10) + ')Observation:?', 'g'),
    '$1<div class="fmt-label obs-label">Observation</div>');
  s = s.replace(new RegExp('(^|' + String.fromCharCode(10) + ')Execution logs:', 'g'),
    '$1<div class="fmt-label obs-label">Execution logs</div>');
  s = s.replace(new RegExp('(^|' + String.fromCharCode(10) + ')Last output from code snippet:', 'g'),
    '$1<div class="fmt-label obs-label">Last output</div>');

  // &lt;end_code&gt; and LLM markers
  s = s.split('&lt;end_code&gt;').join('<span style="color:#484f58;font-size:11px">&lt;end_code&gt;</span>');
  s = s.replace(new RegExp('&lt;LLM (.+?)&gt;', 'g'),
    '<div style="color:#484f58;font-size:11px;margin-top:6px">LLM: $1</div>');

  // input:/output: markers
  s = s.replace(new RegExp('(^|' + String.fromCharCode(10) + ')(input|output): ', 'g'),
    '$1<div class="fmt-label" style="color:#484f58">$2</div>');

  return s;
}

function formatStructured(obj) {
  // Handle memory_step wrapper
  if (obj.memory_step && typeof obj.memory_step === 'string') {
    return '<div class="fmt-obs" style="font-size:11px;color:#484f58">' + esc(obj.memory_step) + '</div>';
  }
  // Handle message objects with role/content
  if (obj.role && obj.content) {
    var role = obj.role;
    var content = obj.content;
    var html = '<div class="fmt-label" style="color:#58a6ff">' + esc(role) + '</div>';
    if (typeof content === 'string') {
      html += formatAgentContent(content);
    } else if (Array.isArray(content)) {
      for (var i = 0; i < content.length; i++) {
        if (content[i].type === 'text' && content[i].text) {
          html += formatAgentContent(content[i].text);
        }
      }
    }
    return html;
  }
  // Handle messages array
  if (obj.messages && Array.isArray(obj.messages)) {
    var html = '';
    var msgs = obj.messages;
    var start = Math.max(0, msgs.length - 4);
    if (start > 0) {
      html += '<div style="color:#484f58;font-size:11px">(' + start + ' earlier messages hidden)</div>';
    }
    for (var i = start; i < msgs.length; i++) {
      html += formatStructured(msgs[i]);
    }
    return html;
  }
  // Fallback: pretty-print JSON
  return '<pre class="fmt-json">' + esc(JSON.stringify(obj, null, 2)).substring(0, 2000) + '</pre>';
}

// Extract and highlight the meaningful parts of SmolAgents-style content,
// collapsing boilerplate (system instructions, tool definitions).
function formatAgentContent(text) {
  if (!text) return '';
  if (typeof text !== 'string') text = String(text);

  // Detect SmolAgents-style inputs: they contain marker phrases
  var hasTaskBlock = text.indexOf('Here is your task:') !== -1 ||
                     text.indexOf('Here is the task:') !== -1;
  var hasToolsList = text.indexOf('You can leverage these tools:') !== -1 ||
                     text.indexOf('You can use these tools:') !== -1 ||
                     text.indexOf('list of tools:') !== -1;
  var hasAgentPreamble = text.indexOf('You are an expert assistant') === 0 ||
                         text.indexOf('You are a world expert') === 0 ||
                         text.indexOf('You are an agent') === 0;

  if (!hasTaskBlock && !hasToolsList && !hasAgentPreamble) {
    // Not agent boilerplate; just format as plain text (no recursion)
    return formatPlain(text);
  }

  var html = '';
  var NL = String.fromCharCode(10);

  // Extract the actual task. Try fenced form first, then unfenced.
  var taskRegex = new RegExp('Here is (?:your|the) task:[\\s]*(?:Task:)?[\\s]*```([^`]*?)```');
  var taskMatch = text.match(taskRegex);
  if (!taskMatch) {
    // Unfenced: task runs until blank line or next section marker
    var idx = text.indexOf('Here is your task:');
    if (idx === -1) idx = text.indexOf('Here is the task:');
    if (idx !== -1) {
      var after = text.substring(idx);
      // Skip past the label line
      var nlIdx = after.indexOf(NL);
      if (nlIdx !== -1) after = after.substring(nlIdx + 1);
      // End at next major marker
      var stopMarkers = [NL + NL, 'List of facts', 'You can leverage', 'Now begin'];
      var endPos = after.length;
      for (var m = 0; m < stopMarkers.length; m++) {
        var p = after.indexOf(stopMarkers[m]);
        if (p !== -1 && p < endPos) endPos = p;
      }
      taskMatch = [null, after.substring(0, endPos).trim()];
    }
  }

  if (taskMatch) {
    html += '<div class="fmt-task-block">';
    html += '<div class="fmt-task-label">Task</div>';
    html += esc(taskMatch[1].trim());
    html += '</div>';
  }

  // Extract facts if present (fenced form)
  var factsRegex = new RegExp('List of facts[^' + NL + ']*:[\\s]*```([\\s\\S]*?)```');
  var factsMatch = text.match(factsRegex);
  if (!factsMatch) {
    // Unfenced: facts run until "Now begin"
    var fIdx = text.indexOf('List of facts');
    if (fIdx !== -1) {
      var afterF = text.substring(fIdx);
      var nIdx = afterF.indexOf(NL);
      if (nIdx !== -1) afterF = afterF.substring(nIdx + 1);
      var endF = afterF.indexOf('Now begin');
      if (endF === -1) endF = afterF.length;
      factsMatch = [null, afterF.substring(0, endF).trim()];
    }
  }
  if (factsMatch && factsMatch[1].length > 10) {
    html += '<div class="fmt-facts-block">';
    html += '<div class="fmt-facts-label">Facts</div>';
    html += esc(factsMatch[1].trim());
    html += '</div>';
  }

  // Extract what follows "Now begin"
  var beginIdx = text.indexOf('Now begin');
  if (beginIdx !== -1) {
    var afterBegin = text.substring(beginIdx);
    var exclIdx = afterBegin.indexOf('!');
    if (exclIdx !== -1) {
      var instruction = afterBegin.substring(exclIdx + 1).trim();
      if (instruction.length > 0 && instruction.length < 500) {
        html += '<div class="fmt-reply-block">';
        html += '<div class="fmt-reply-label">Instruction</div>';
        html += esc(instruction);
        html += '</div>';
      }
    }
  }

  // Count and collapse tool definitions (lines starting with "- wordname:")
  var toolLines = text.split(NL);
  var toolNames = [];
  for (var tl = 0; tl < toolLines.length; tl++) {
    var ln = toolLines[tl];
    // Match lines like "- toolname:" at the start
    if (ln.length > 3 && ln.charAt(0) === '-' && ln.charAt(1) === ' ') {
      var rest = ln.substring(2);
      var colonIdx = rest.indexOf(':');
      if (colonIdx > 0 && colonIdx < 40) {
        var name = rest.substring(0, colonIdx);
        // Simple word check: only letters, digits, underscores
        var valid = name.length > 0;
        for (var ci = 0; ci < name.length; ci++) {
          var cc = name.charCodeAt(ci);
          if (!((cc >= 48 && cc <= 57) || (cc >= 65 && cc <= 90) ||
                (cc >= 97 && cc <= 122) || cc === 95)) {
            valid = false; break;
          }
        }
        if (valid) toolNames.push(name);
      }
    }
  }
  if (toolNames.length > 0) {
    html += '<div class="fmt-collapse">';
    html += '(' + toolNames.length + ' tool definitions hidden: ';
    html += esc(toolNames.join(', '));
    html += ')</div>';
  }

  // Note the system instructions were collapsed
  if (hasAgentPreamble) {
    html += '<div class="fmt-collapse">(system instructions hidden)</div>';
  }

  // If we extracted nothing meaningful, fall back to plain text (no recursion)
  if (html.length < 50) {
    return formatPlain(text);
  }

  return html;
}

// Plain-text formatter: applies thought/code/observation/diff highlighting
// but does NOT recurse into agent-content detection. Used as the terminal
// formatter to avoid infinite recursion between formatTrace and formatAgentContent.
function formatPlain(text) {
  if (!text) return '';
  var s = esc(text);
  var NL = String.fromCharCode(10);

  // Diff detection (same logic as in formatTrace, but terminal)
  var lines = s.split(NL);
  var out = [];
  var diffBuf = [];
  var inDiff = false;

  function flushDiffBuffer() {
    if (diffBuf.length === 0) return;
    out.push('<div class="fmt-diff">');
    for (var j = 0; j < diffBuf.length; j++) {
      var dl = diffBuf[j];
      var dc = 'fmt-diff-ctx';
      if (dl.indexOf('+++') === 0 || dl.indexOf('---') === 0) dc = 'fmt-diff-file';
      else if (dl.indexOf('@@') === 0) dc = 'fmt-diff-hunk';
      else if (dl.charAt(0) === '+') dc = 'fmt-diff-add';
      else if (dl.charAt(0) === '-') dc = 'fmt-diff-del';
      out.push('<div class="fmt-diff-line ' + dc + '">' + (dl || ' ') + '</div>');
    }
    out.push('</div>');
    diffBuf = [];
  }

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var isDiffStart = line.indexOf('--- a/') === 0 || line.indexOf('--- /') === 0 ||
                      line.indexOf('diff --git') === 0;
    var isDiffLine = line.indexOf('+++') === 0 || line.indexOf('---') === 0 ||
                     line.indexOf('@@') === 0 ||
                     line.charAt(0) === '+' || line.charAt(0) === '-';
    if (!inDiff && isDiffStart) {
      inDiff = true;
      diffBuf.push(line);
    } else if (inDiff && (isDiffLine || line.charAt(0) === ' ' || line === '')) {
      diffBuf.push(line);
    } else {
      if (inDiff) { flushDiffBuffer(); inDiff = false; }
      out.push(line);
    }
  }
  if (inDiff) flushDiffBuffer();
  s = out.join(NL);

  // Apply Thought/Code/Observation markers
  s = s.split('&lt;end_code&gt;').join('<span style="color:#484f58;font-size:11px">&lt;end_code&gt;</span>');
  return s;
}

function updateProgress() {
  var n = Object.keys(LABELS).length;
  var dt = 0, hs = 0;
  for (var k in LABELS) {
    if (LABELS[k].hindsight_class === 'decision_time') dt++;
    if (LABELS[k].hindsight_class === 'hindsight') hs++;
  }
  document.getElementById('progressText').textContent =
    n + '/' + ITEMS.length + ' classified (' + dt + ' decision, ' + hs + ' hindsight)';
}

function goPrev() { if (current > 0) render(current - 1); }
function goNext() { if (current < ITEMS.length - 1) render(current + 1); }

function render(idx) {
  current = idx;
  var item = ITEMS[idx];
  var key = item.trace_id + ':' + item.step_index;
  var existing = LABELS[key];

  var html = '';

  // Instruction
  html += '<div class="instruction">';
  html += '<h3>Could a grader detect this error WITHOUT future steps?</h3>';
  html += '<p>TRAIL says this step is a <strong>' + item.trail_severity + '-severity failure</strong>';
  if (item.trail_categories && item.trail_categories.length > 0) {
    html += ' (' + esc(item.trail_categories.join(', ')) + ')';
  }
  html += ', but the grader said <strong>' + item.grader_label + '</strong>. ';
  if (item.trail_dependency) {
    html += 'Dependency: <strong>' + item.trail_dependency + '</strong>. ';
  }
  html += 'You can see the full trace including future steps. ';
  html += 'The question is: <strong>could the error have been identified at decision time</strong> ';
  html += '(using only prior steps and the current step), or does detecting it ';
  html += '<strong>require knowledge of what happened next</strong>?</p>';
  html += '</div>';

  // Info bar
  html += '<div class="info-bar">';
  html += '<h2>' + esc(item.trace_id.substring(0, 60)) + '</h2>';
  html += '<div class="meta">';
  html += '<span>Step ' + item.step_index + ' of ' + item.total_steps + '</span>';
  html += '<span class="fail">TRAIL: fail (' + item.trail_severity + ')</span>';
  html += '<span class="neutral">Grader: ' + item.grader_label + '</span>';
  html += '</div>';
  html += '<div class="task-goal">' + esc(item.task_goal) + '</div>';
  html += '</div>';

  // Prior steps
  var priors = item.prior_steps;
  if (priors.length > 0) {
    for (var i = 0; i < priors.length; i++) {
      var s = priors[i];
      html += '<div class="step-card prior">';
      html += '<div class="step-header"><span class="step-tag">Prior Step ' + s.index + '</span></div>';
      html += '<div class="step-body">';
      if (s.thought) {
        html += '<div class="block"><div class="label">Thought</div>';
        html += '<div class="text thought">' + formatTrace(s.thought.substring(0, 800)) + '</div></div>';
      }
      if (s.action) {
        html += '<div class="block"><div class="label">Action</div>';
        html += '<div class="text action">' + formatTrace(s.action.substring(0, 800)) + '</div></div>';
      }
      if (s.observation) {
        html += '<div class="block"><div class="label">Observation</div>';
        html += '<div class="text obs">' + formatTrace(s.observation.substring(0, 600)) + '</div></div>';
      }
      html += '</div></div>';
    }
  }

  // Current step (highlighted)
  var cs = item.current_step;
  html += '<div class="step-card current">';
  html += '<div class="step-header"><span class="step-tag current-tag">CURRENT STEP ' + cs.index + ' (the one TRAIL says is wrong)</span></div>';
  html += '<div class="step-body">';
  if (cs.thought) {
    html += '<div class="block"><div class="label">Thought</div>';
    html += '<div class="text thought">' + formatTrace(cs.thought) + '</div></div>';
  }
  if (cs.action) {
    html += '<div class="block"><div class="label">Action</div>';
    html += '<div class="text action">' + formatTrace(cs.action) + '</div></div>';
  }
  if (cs.observation) {
    html += '<div class="block"><div class="label">Observation</div>';
    html += '<div class="text obs">' + formatTrace(cs.observation) + '</div></div>';
  }
  html += '</div></div>';

  // Future steps
  var futures = item.future_steps;
  if (futures.length > 0) {
    html += '<div style="margin:12px 0 4px;font-size:12px;color:#484f58;text-transform:uppercase;letter-spacing:0.5px">Future steps (what happened after)</div>';
    for (var i = 0; i < futures.length; i++) {
      var fs = futures[i];
      html += '<div class="step-card prior" style="opacity:0.5;border-left:3px solid #30363d">';
      html += '<div class="step-header"><span class="step-tag" style="color:#484f58">Future Step ' + fs.index + '</span></div>';
      html += '<div class="step-body">';
      if (fs.thought) {
        html += '<div class="block"><div class="label">Thought</div>';
        html += '<div class="text thought">' + formatTrace(fs.thought.substring(0, 600)) + '</div></div>';
      }
      if (fs.action) {
        html += '<div class="block"><div class="label">Action</div>';
        html += '<div class="text action">' + formatTrace(fs.action.substring(0, 600)) + '</div></div>';
      }
      if (fs.observation) {
        html += '<div class="block"><div class="label">Observation</div>';
        html += '<div class="text obs">' + formatTrace(fs.observation.substring(0, 500)) + '</div></div>';
      }
      html += '</div></div>';
    }
  }

  // Classification buttons
  html += '<div class="classify-panel">';
  html += '<h3>Your Classification</h3>';
  html += '<div class="question">Could a grader with only the task goal, prior steps, and current step have detected this error?</div>';
  html += '<div class="classify-btns">';

  var dtSel = existing && existing.hindsight_class === 'decision_time' ? ' selected' : '';
  var hsSel = existing && existing.hindsight_class === 'hindsight' ? ' selected' : '';

  html += '<button class="classify-btn decision' + dtSel + '" onclick="classify(&quot;decision_time&quot;)">';
  html += 'Detectable at decision time<br><span style="font-size:11px;font-weight:normal">The error is visible from prior context + current step alone</span></button>';

  html += '<button class="classify-btn hindsight' + hsSel + '" onclick="classify(&quot;hindsight&quot;)">';
  html += 'Requires hindsight<br><span style="font-size:11px;font-weight:normal">The error only becomes apparent from future steps</span></button>';

  html += '<button class="classify-btn skip" onclick="goNext()">';
  html += 'Skip<br><span style="font-size:11px;font-weight:normal">Unsure</span></button>';

  html += '</div>';
  html += '<div style="margin-top:10px;font-size:12px;color:#484f58">Keyboard: <span class="kbd">D</span> = detectable &nbsp; <span class="kbd">H</span> = hindsight &nbsp; <span class="kbd">S</span> / <span class="kbd">→</span> = skip/next &nbsp; <span class="kbd">←</span> = prev</div>';
  html += '</div>';

  // Running results
  var dt = 0, hs = 0;
  for (var k in LABELS) {
    if (LABELS[k].hindsight_class === 'decision_time') dt++;
    if (LABELS[k].hindsight_class === 'hindsight') hs++;
  }
  var total = dt + hs;
  if (total > 0) {
    html += '<div class="result-bar">';
    html += '<div class="result-text">Running tally: ' + dt + ' decision-time, ' + hs + ' hindsight (' + total + ' classified)</div>';
    var dtPct = (dt / total * 100).toFixed(0);
    var hsPct = (hs / total * 100).toFixed(0);
    html += '<div class="bar">';
    html += '<div class="bar-dt" style="width:' + dtPct + '%"></div>';
    html += '<div class="bar-hs" style="width:' + hsPct + '%"></div>';
    html += '</div>';
    html += '<div class="result-text" style="display:flex;justify-content:space-between">';
    html += '<span>Detectable: ' + dtPct + '%</span>';
    html += '<span>Hindsight: ' + hsPct + '%</span>';
    html += '</div></div>';
  }

  document.getElementById('main').innerHTML = html;
  updateProgress();
  window.scrollTo(0, 0);
}

async function classify(cls) {
  var item = ITEMS[current];
  var label = {
    trace_id: item.trace_id,
    step_index: item.step_index,
    trail_severity: item.trail_severity,
    grader_label: item.grader_label,
    hindsight_class: cls,
  };

  var resp = await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(label),
  });

  if (resp.ok) {
    var key = item.trace_id + ':' + item.step_index;
    LABELS[key] = label;
    // Auto-advance
    if (current < ITEMS.length - 1) {
      render(current + 1);
    } else {
      render(current);
    }
  }
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowLeft') goPrev();
  if (e.key === 'ArrowRight') goNext();
  if (e.key === 'd' || e.key === 'D') classify('decision_time');
  if (e.key === 'h' || e.key === 'H') classify('hindsight');
  if (e.key === 's' || e.key === 'S') goNext();
});

init();
</script>
</body>
</html>"""


class Handler(SimpleHTTPRequestHandler):
    items: list[dict] = []
    labels: dict = {}
    labels_path: Path = Path()

    def do_GET(self):
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "items": self.items,
                "labels": self.labels,
            }).encode())
        elif self.path == "/" or self.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            label = json.loads(body)

            key = f"{label['trace_id']}:{label['step_index']}"
            self.labels[key] = label

            with self.labels_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(label) + "\n")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trail-root", type=Path, default=DEFAULT_TRAIL_ROOT)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    labels_path = STUDY_ROOT / "results" / "fn_classifications.jsonl"

    print("Loading TRAIL and grader predictions...")
    items, stats = _load_false_negatives(args.trail_root, args.n, args.seed)
    labels = _load_existing(labels_path)

    print(f"Total MEDIUM+ false negatives: {stats['total_fn']}")
    print(f"Sampled: {stats['sampled']}")
    print(f"Existing classifications: {len(labels)}")
    print(f"Labels file: {labels_path}")
    print(f"\nOpen http://localhost:{args.port}")

    Handler.items = items
    Handler.labels = labels
    Handler.labels_path = labels_path

    server = HTTPServer(("localhost", args.port), Handler)
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        n_dt = sum(1 for v in labels.values() if v.get("hindsight_class") == "decision_time")
        n_hs = sum(1 for v in labels.values() if v.get("hindsight_class") == "hindsight")
        total = n_dt + n_hs
        if total:
            print(f"\n{total} classified: {n_dt} decision-time ({n_dt/total:.0%}), {n_hs} hindsight ({n_hs/total:.0%})")
        print(f"Saved to {labels_path}")


if __name__ == "__main__":
    main()
