"""Local web UI for human labeling of graded traces.

Serves a single-page app that shows each step alongside the grader's
labels, lets you enter your own labels, and saves to a JSONL file.

Usage:
    python scripts/human_labeler.py results/phase3/minimax.cache.jsonl

Opens http://localhost:8765 in your browser. Labels are saved to
results/phase3/minimax.human_labels.jsonl as you go. Progress is
preserved — refresh or restart anytime.
"""

from __future__ import annotations

import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from inspect_degradation.store import GradedTraceStore

PORT = 8765


def _load_traces(cache_path: str, source_traces: list | None = None) -> list[dict]:
    store = GradedTraceStore(cache_path)
    graded = store.load_all()

    # Build a lookup of source traces by trace_id for the original
    # action/observation content.
    source_by_id: dict[str, dict[int, dict]] = {}
    if source_traces:
        for t in source_traces:
            steps_map = {}
            for s in t.steps:
                steps_map[s.index] = {
                    "action": s.action,
                    "observation": s.observation,
                    "thought": s.thought,
                }
            source_by_id[t.trace_id] = steps_map

    result = []
    for t in graded:
        source_steps = source_by_id.get(t.trace_id, {})
        task_goal = ""
        if source_traces:
            for st in source_traces:
                if st.trace_id == t.trace_id:
                    task_goal = st.task_goal
                    break

        steps = []
        for s in t.steps:
            src = source_steps.get(s.step_index, {})
            steps.append({
                "step_index": s.step_index,
                "action": src.get("action", ""),
                "observation": src.get("observation", ""),
                "thought": src.get("thought", ""),
                "grader_validity": s.validity.value,
                "grader_complexity": s.complexity.value if s.complexity else None,
                "grader_dependency": s.dependency.value if s.dependency else None,
                "grader_severity": s.severity.value if s.severity else None,
                "grader_is_looping": s.is_looping,
            })
        result.append({
            "trace_id": t.trace_id,
            "task_id": t.task_id,
            "task_goal": task_goal,
            "model": t.model,
            "source": t.source,
            "success": t.success,
            "n_steps": len(t.steps),
            "steps": steps,
        })
    return result


def _load_existing_labels(labels_path: Path) -> dict:
    """Load existing human labels keyed by trace_id:step_index."""
    labels = {}
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
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
<title>Human Labeler</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
.header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }
.header h1 { font-size: 16px; color: #58a6ff; }
.progress { color: #8b949e; font-size: 14px; }
.nav { display: flex; gap: 8px; align-items: center; }
.nav button { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.nav button:hover { background: #30363d; border-color: #8b949e; }
.nav select { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 12px; border-radius: 6px; font-size: 13px; }
.container { max-width: 1400px; margin: 0 auto; padding: 16px 24px; }
.trace-info { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.trace-info h2 { font-size: 14px; color: #58a6ff; margin-bottom: 8px; }
.trace-meta { display: flex; gap: 24px; font-size: 13px; color: #8b949e; }
.trace-meta span { }
.trace-meta .success-true { color: #3fb950; }
.trace-meta .success-false { color: #f85149; }
.task-goal { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-top: 8px; font-size: 13px; max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; color: #8b949e; }
.step { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }
.step-header { display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: #21262d; border-bottom: 1px solid #30363d; }
.step-number { font-weight: 600; font-size: 13px; color: #58a6ff; }
.step-status { font-size: 12px; padding: 2px 8px; border-radius: 10px; }
.step-status.labeled { background: #238636; color: #fff; }
.step-status.unlabeled { background: #30363d; color: #8b949e; }
.step-body { display: flex; gap: 0; }
.step-content { flex: 1; padding: 12px 16px; border-right: 1px solid #30363d; min-width: 0; }
.step-labels { width: 340px; min-width: 340px; padding: 12px 16px; background: #0d1117; }
.content-block { margin-bottom: 12px; }
.content-label { font-size: 11px; font-weight: 600; text-transform: uppercase; color: #8b949e; margin-bottom: 4px; letter-spacing: 0.5px; }
.content-text { font-size: 13px; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 8px; line-height: 1.5; }
.grader-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }
.grader-label { color: #8b949e; min-width: 80px; }
.grader-value { color: #c9d1d9; font-weight: 500; }
.grader-value.fail { color: #f85149; }
.grader-value.neutral { color: #d29922; }
.grader-value.pass { color: #3fb950; }
.human-section { margin-top: 12px; padding-top: 12px; border-top: 1px solid #30363d; }
.human-section h4 { font-size: 12px; color: #58a6ff; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
.field-group { margin-bottom: 8px; }
.field-group label { display: block; font-size: 12px; color: #8b949e; margin-bottom: 3px; }
.btn-group { display: flex; gap: 4px; flex-wrap: wrap; }
.opt-btn { background: #21262d; border: 1px solid #30363d; color: #8b949e; padding: 3px 10px; border-radius: 4px; font-size: 12px; cursor: pointer; transition: all 0.1s; }
.opt-btn:hover { background: #30363d; color: #c9d1d9; }
.opt-btn.selected { background: #1f6feb; border-color: #388bfd; color: #fff; }
.opt-btn.selected.fail { background: #da3633; border-color: #f85149; }
.opt-btn.selected.neutral { background: #9e6a03; border-color: #d29922; }
.opt-btn.selected.pass { background: #238636; border-color: #2ea043; }
.grader-details { margin-bottom: 8px; }
.grader-toggle { font-size: 11px; color: #484f58; cursor: pointer; user-select: none; }
.grader-toggle:hover { color: #8b949e; }
.save-btn { width: 100%; background: #238636; border: 1px solid #2ea043; color: #fff; padding: 6px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; margin-top: 8px; }
.save-btn:hover { background: #2ea043; }
.save-btn.saved { background: #21262d; border-color: #30363d; color: #8b949e; }
.keyboard-hint { font-size: 11px; color: #484f58; text-align: center; margin-top: 6px; }
</style>
</head>
<body>

<div class="header">
  <h1>Human Labeler</h1>
  <div class="nav">
    <button onclick="prevTrace()">Prev Trace</button>
    <select id="traceSelect" onchange="goToTrace(this.value)"></select>
    <button onclick="nextTrace()">Next Trace</button>
    <span class="progress" id="progressText"></span>
  </div>
</div>

<div class="container" id="main"></div>

<script>
let TRACES = [];
let LABELS = {};
let currentTrace = 0;

async function init() {
  const resp = await fetch('/api/data');
  const data = await resp.json();
  TRACES = data.traces;
  LABELS = data.labels;

  const sel = document.getElementById('traceSelect');
  TRACES.forEach((t, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = t.trace_id.substring(0, 50) + ' (' + t.n_steps + ' steps)';
    sel.appendChild(opt);
  });

  renderTrace(0);
}

function makeBtnGroup(groupId, options, current, si) {
  var h = '<div class="btn-group">';
  for (var j = 0; j < options.length; j++) {
    var opt = options[j];
    var isSel = (current === opt || (typeof current === 'boolean' && String(current) === opt));
    var cls = isSel ? 'opt-btn selected ' + opt : 'opt-btn';
    h += '<button class="' + cls + '" data-group="' + groupId + '" data-value="' + opt + '" data-si="' + si + '">' + opt + '</button>';
  }
  h += '</div>';
  return h;
}

function updateProgress() {
  const total = TRACES.reduce((sum, t) => sum + t.n_steps, 0);
  const labeled = Object.keys(LABELS).length;
  document.getElementById('progressText').textContent = labeled + '/' + total + ' steps labeled';
}

function prevTrace() { if (currentTrace > 0) renderTrace(currentTrace - 1); }
function nextTrace() { if (currentTrace < TRACES.length - 1) renderTrace(currentTrace + 1); }
function goToTrace(i) { renderTrace(parseInt(i)); }

function renderTrace(idx) {
  currentTrace = idx;
  document.getElementById('traceSelect').value = idx;
  const t = TRACES[idx];

  let html = '<div class="trace-info">';
  html += '<h2>' + esc(t.trace_id) + '</h2>';
  html += '<div class="trace-meta">';
  html += '<span>Model: ' + esc(t.model || 'unknown') + '</span>';
  html += '<span>Steps: ' + t.n_steps + '</span>';
  html += '<span class="success-' + t.success + '">Success: ' + t.success + '</span>';
  html += '<span>Task: ' + esc((t.task_id || '').substring(0, 60)) + '</span>';
  html += '<button onclick="copyTrace()" style="background:#21262d;border:1px solid #30363d;color:#8b949e;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:12px">Copy trace</button>';
  html += '</div>';
  if (t.task_goal) {
    html += '<div class="task-goal">' + esc(t.task_goal) + '</div>';
  }
  html += '</div>';

  t.steps.forEach((s, si) => {
    const key = t.trace_id + ':' + s.step_index;
    const existing = LABELS[key];
    const isLabeled = !!existing;

    html += '<div class="step" id="step-' + si + '">';
    html += '<div class="step-header">';
    html += '<span class="step-number">Step ' + s.step_index + '</span>';
    html += '<span class="step-status ' + (isLabeled ? 'labeled' : 'unlabeled') + '">' + (isLabeled ? 'labeled' : 'unlabeled') + '</span>';
    html += '</div>';
    html += '<div class="step-body">';

    // Content
    html += '<div class="step-content">';
    if (s.thought) {
      html += '<div class="content-block"><div class="content-label">Thought</div>';
      html += '<div class="content-text" style="color:#d2a8ff">' + esc(s.thought) + '</div></div>';
    }
    if (s.action) {
      html += '<div class="content-block"><div class="content-label">Action</div>';
      html += '<div class="content-text">' + esc(s.action) + '</div></div>';
    }
    if (s.observation) {
      html += '<div class="content-block"><div class="content-label">Observation</div>';
      html += '<div class="content-text" style="color:#8b949e">' + esc(s.observation) + '</div></div>';
    }
    if (!s.action && !s.observation && !s.thought) {
      html += '<div class="content-block"><div class="content-label">No content available</div>';
      html += '<div class="content-text" style="color:#f85149">Source traces not loaded. Pass --source flag.</div></div>';
    }
    html += '</div>';

    // Labels panel
    html += '<div class="step-labels">';

    // Grader labels — hidden by default to avoid biasing human judgment
    html += '<details class="grader-details"><summary class="grader-toggle">Show grader labels</summary>';
    html += '<div class="grader-row"><span class="grader-label">validity</span><span class="grader-value ' + s.grader_validity + '">' + s.grader_validity + '</span></div>';
    html += '<div class="grader-row"><span class="grader-label">complexity</span><span class="grader-value">' + (s.grader_complexity || 'n/a') + '</span></div>';
    html += '<div class="grader-row"><span class="grader-label">dependency</span><span class="grader-value">' + (s.grader_dependency || 'n/a') + '</span></div>';
    html += '<div class="grader-row"><span class="grader-label">severity</span><span class="grader-value">' + (s.grader_severity || 'n/a') + '</span></div>';
    html += '<div class="grader-row"><span class="grader-label">looping</span><span class="grader-value">' + s.grader_is_looping + '</span></div>';
    html += '</details>';

    // Human labels
    html += '<div class="human-section">';
    html += '<h4>Your Label</h4>';

    const v = existing ? existing.validity : '';
    const c = existing ? existing.complexity : '';
    const d = existing ? existing.dependency : '';
    const sev = existing ? existing.severity : '';
    const loop = existing ? existing.is_looping : '';

    html += '<div class="field-group"><label>Validity</label>';
    html += makeBtnGroup('v-' + si, ['fail', 'neutral', 'pass'], v, si);
    html += '</div>';

    html += '<div class="field-group"><label>Complexity</label>';
    html += makeBtnGroup('c-' + si, ['low', 'medium', 'high'], c, si);
    html += '</div>';

    html += '<div class="field-group"><label>Dependency</label>';
    html += makeBtnGroup('d-' + si, ['independent', 'dependent', 'n/a'], d, si);
    html += '</div>';

    html += '<div class="field-group"><label>Severity</label>';
    html += makeBtnGroup('sev-' + si, ['low', 'medium', 'high'], sev, si);
    html += '</div>';

    html += '<div class="field-group"><label>Looping</label>';
    html += makeBtnGroup('loop-' + si, ['true', 'false'], String(loop), si);
    html += '</div>';

    html += '<button class="save-btn' + (isLabeled ? ' saved' : '') + '" id="btn-' + si + '" onclick="saveStep(' + si + ')">Save</button>';
    html += '<div class="keyboard-hint">Enter = save &amp; next</div>';

    html += '</div>'; // human-section
    html += '</div>'; // step-labels
    html += '</div>'; // step-body
    html += '</div>'; // step
  });

  document.getElementById('main').innerHTML = html;
  updateProgress();
  window.scrollTo(0, 0);
}

function selectOpt(btn, groupId, value, si) {
  // Deselect all in group, select this one.
  document.querySelectorAll('[data-group="' + groupId + '"]').forEach(b => {
    b.className = 'opt-btn';
  });
  btn.className = 'opt-btn selected ' + value;

  // Auto-set dependency and severity when validity changes.
  if (groupId.startsWith('v-') && value !== 'fail') {
    // Auto-select n/a for dependency, clear severity.
    const dBtns = document.querySelectorAll('[data-group="d-' + si + '"]');
    dBtns.forEach(b => {
      b.className = b.dataset.value === 'n/a' ? 'opt-btn selected n/a' : 'opt-btn';
    });
    const sevBtns = document.querySelectorAll('[data-group="sev-' + si + '"]');
    sevBtns.forEach(b => { b.className = 'opt-btn'; });
  }
}

function getSelected(groupId) {
  const btn = document.querySelector('[data-group="' + groupId + '"].selected');
  return btn ? btn.dataset.value : null;
}

async function saveStep(si) {
  const t = TRACES[currentTrace];
  const s = t.steps[si];
  const label = {
    trace_id: t.trace_id,
    step_index: s.step_index,
    validity: getSelected('v-' + si),
    complexity: getSelected('c-' + si),
    dependency: getSelected('d-' + si),
    severity: getSelected('sev-' + si),
    is_looping: getSelected('loop-' + si),
  };

  if (!label.validity) { alert('Validity is required'); return; }
  if (label.is_looping === 'true') label.is_looping = true;
  if (label.is_looping === 'false') label.is_looping = false;

  const resp = await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(label),
  });

  if (resp.ok) {
    const key = t.trace_id + ':' + s.step_index;
    LABELS[key] = label;
    const btn = document.getElementById('btn-' + si);
    btn.textContent = 'Saved';
    btn.className = 'save-btn saved';
    const status = document.querySelector('#step-' + si + ' .step-status');
    status.textContent = 'labeled';
    status.className = 'step-status labeled';
    updateProgress();
  }
}

function esc(s) {
  if (!s) return '';
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function copyTrace() {
  var t = TRACES[currentTrace];
  var text = 'Trace: ' + t.trace_id + '\\n';
  text += 'Model: ' + (t.model || 'unknown') + '\\n';
  text += 'Success: ' + t.success + '\\n';
  text += 'Task: ' + (t.task_id || '') + '\\n';
  if (t.task_goal) text += '\\nTask Goal:\\n' + t.task_goal + '\\n';
  t.steps.forEach(function(s) {
    text += '\\n--- Step ' + s.step_index + ' ---\\n';
    if (s.thought) text += 'THOUGHT: ' + s.thought.substring(0, 500) + '\\n';
    if (s.action) text += 'ACTION: ' + s.action.substring(0, 500) + '\\n';
    if (s.observation) text += 'OBSERVATION: ' + s.observation.substring(0, 500) + '\\n';
  });
  navigator.clipboard.writeText(text).then(function() {
    alert('Copied to clipboard');
  });
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') prevTrace();
  if (e.key === 'ArrowRight') nextTrace();
});

document.addEventListener('click', (e) => {
  var btn = e.target.closest('.opt-btn');
  if (!btn) return;
  var groupId = btn.dataset.group;
  var value = btn.dataset.value;
  var si = parseInt(btn.dataset.si);
  selectOpt(btn, groupId, value, si);
});

init();
</script>
</body>
</html>"""


class LabelHandler(SimpleHTTPRequestHandler):
    traces: list[dict] = []
    labels: dict = {}
    labels_path: Path = Path()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "traces": self.traces,
                "labels": self.labels,
            }).encode())
        elif parsed.path == "/" or parsed.path == "":
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
        pass  # suppress per-request logging


def _detect_and_load_sources(cache_path: str) -> list | None:
    """Try to reload source traces matching the cached grades.

    Looks at the first cached trace's source field to decide which
    loader to use, then reloads the traces so we have the original
    action/observation content.
    """
    store = GradedTraceStore(cache_path)
    first = next(iter(store), None)
    if first is None:
        return None

    source = first.source
    trace_ids = store.completed_trace_ids()
    print(f"Detected source: {source}, {len(trace_ids)} traces to match")

    if source == "nebius":
        from inspect_degradation.datasets.nebius import load_nebius
        # Load enough to cover all cached trace IDs.
        all_traces = load_nebius(limit=len(trace_ids) * 3, one_per_instance=True)
        matched = [t for t in all_traces if t.trace_id in trace_ids]
        print(f"Matched {len(matched)}/{len(trace_ids)} Nebius traces")
        return matched

    if source == "swe-smith":
        from inspect_degradation.datasets.swe_smith import load_swe_smith
        all_traces = load_swe_smith(limit=len(trace_ids) * 3, one_per_instance=True)
        matched = [t for t in all_traces if t.trace_id in trace_ids]
        print(f"Matched {len(matched)}/{len(trace_ids)} SWE-smith traces")
        return matched

    print(f"Unknown source '{source}', no content will be shown")
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/human_labeler.py <cache.jsonl>")
        return

    cache_path = sys.argv[1]
    labels_path = Path(cache_path).with_suffix(".human_labels.jsonl")

    print("Loading source traces for step content...")
    source_traces = _detect_and_load_sources(cache_path)
    traces = _load_traces(cache_path, source_traces=source_traces)
    labels = _load_existing_labels(labels_path)

    print(f"Loaded {len(traces)} traces, {sum(t['n_steps'] for t in traces)} steps")
    print(f"Existing labels: {len(labels)}")
    print(f"Labels file: {labels_path}")
    print(f"\nOpen http://localhost:{PORT}")

    LabelHandler.traces = traces
    LabelHandler.labels = labels
    LabelHandler.labels_path = labels_path

    server = HTTPServer(("localhost", PORT), LabelHandler)
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{len(labels)} labels saved to {labels_path}")


if __name__ == "__main__":
    main()
