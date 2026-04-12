"""Web UI to review the step-phase classifier (explore vs act).

Shows each step's action text alongside the classifier's decision
so you can spot misclassifications.

Usage:
    python scripts/review_step_phase.py results/phase3-autoswe/minimax.cache.jsonl

Opens http://localhost:8766. Use arrow keys to navigate traces.
"""

from __future__ import annotations

import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from inspect_degradation.step_phase import classify_step_phase
from inspect_degradation.store import GradedTraceStore

PORT = 8766


def _load_data(cache_path: str, source_loader=None) -> list[dict]:
    store = GradedTraceStore(cache_path)
    graded = store.load_all()

    # Try to get source traces for the action text.
    source_by_id: dict[str, dict[int, dict]] = {}
    if source_loader:
        for t in source_loader:
            source_by_id[t.trace_id] = {
                s.index: {
                    "action": s.action,
                    "observation": s.observation,
                    "thought": s.thought,
                }
                for s in t.steps
            }

    result = []
    for t in graded:
        source_steps = source_by_id.get(t.trace_id, {})
        steps = []
        for s in t.steps:
            src = source_steps.get(s.step_index, {})
            action = src.get("action", "") if isinstance(src, dict) else (src or "")
            observation = src.get("observation", "") if isinstance(src, dict) else ""
            thought = src.get("thought", "") if isinstance(src, dict) else ""

            # action = model output (reasoning + tool calls)
            # observation = tool results from environment
            display = ""
            if thought:
                display += f"THOUGHT:\n{thought}\n\n"
            if action:
                display += f"ACTION (model output):\n{action}\n\n"
            if observation:
                display += f"OBSERVATION (tool results):\n{observation}"

            if not display and s.raw:
                display = s.raw.get("completion", str(s.raw))

            # Classify on the action (model output) since that contains
            # the tool calls the model made.
            classify_text = action
            cached_phase = s.raw.get("step_phase") if s.raw else None
            live_phase = classify_step_phase(classify_text)

            steps.append({
                "step_index": s.step_index,
                "action": display[:5000],
                "observation": None,  # skip for speed
                "cached_phase": cached_phase,
                "live_phase": live_phase,
                "match": cached_phase == live_phase if cached_phase else None,
                "validity": s.validity.value,
            })
        result.append({
            "trace_id": t.trace_id,
            "model": t.model,
            "source": t.source,
            "n_steps": len(steps),
            "steps": steps,
        })
    return result


HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Step Phase Review</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
.header { background: #161b22; border-bottom: 1px solid #30363d; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }
.header h1 { font-size: 15px; color: #58a6ff; }
.nav { display: flex; gap: 8px; align-items: center; }
.nav button { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 5px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; }
.nav button:hover { background: #30363d; }
.nav select { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 5px 10px; border-radius: 6px; font-size: 12px; max-width: 400px; }
.stats { color: #8b949e; font-size: 13px; padding: 8px 20px; background: #161b22; border-bottom: 1px solid #30363d; }
.container { max-width: 1200px; margin: 0 auto; padding: 12px 20px; }
.step { display: flex; gap: 0; margin-bottom: 8px; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }
.step.mismatch { border-color: #f85149; }
.phase-badge { min-width: 90px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.phase-badge.explore { background: #1f6feb22; color: #58a6ff; }
.phase-badge.act { background: #da363322; color: #f85149; }
.step-num { min-width: 50px; display: flex; align-items: center; justify-content: center; font-size: 12px; color: #8b949e; background: #161b22; border-right: 1px solid #30363d; }
.step-content { flex: 1; padding: 8px 12px; background: #0d1117; font-size: 12px; white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow-y: auto; line-height: 1.4; }
.validity { min-width: 60px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 500; }
.validity.fail { color: #f85149; }
.validity.neutral { color: #d29922; }
.validity.pass { color: #3fb950; }
.filter-bar { padding: 8px 20px; background: #161b22; border-bottom: 1px solid #30363d; display: flex; gap: 12px; align-items: center; }
.filter-bar label { font-size: 12px; color: #8b949e; }
.filter-bar input[type=checkbox] { margin-right: 4px; }
</style>
</head>
<body>

<div class="header">
  <h1>Step Phase Classifier Review</h1>
  <div class="nav">
    <button onclick="prev()">Prev</button>
    <select id="traceSel" onchange="go(this.value)"></select>
    <button onclick="next()">Next</button>
  </div>
</div>
<div class="filter-bar">
  <label><input type="checkbox" id="showAll" checked onchange="render()"> Show all</label>
  <label><input type="checkbox" id="showExplore" checked onchange="render()"> Explore</label>
  <label><input type="checkbox" id="showAct" checked onchange="render()"> Act</label>
  <label><input type="checkbox" id="onlyMismatch" onchange="render()"> Mismatches only</label>
</div>
<div class="stats" id="stats"></div>
<div class="container" id="main"></div>

<script>
var DATA = [];
var cur = 0;

async function init() {
  var resp = await fetch('/api/data');
  DATA = await resp.json();
  var sel = document.getElementById('traceSel');
  DATA.forEach(function(t, i) {
    var opt = document.createElement('option');
    opt.value = i;
    opt.textContent = t.trace_id.substring(0, 55) + ' (' + t.n_steps + ')';
    sel.appendChild(opt);
  });
  render();
}

function prev() { if (cur > 0) { cur--; document.getElementById('traceSel').value = cur; render(); } }
function next() { if (cur < DATA.length - 1) { cur++; document.getElementById('traceSel').value = cur; render(); } }
function go(i) { cur = parseInt(i); render(); }

function render() {
  var t = DATA[cur];
  var showExplore = document.getElementById('showExplore').checked;
  var showAct = document.getElementById('showAct').checked;
  var onlyMismatch = document.getElementById('onlyMismatch').checked;
  var showAll = document.getElementById('showAll').checked;

  var nExplore = 0, nAct = 0, nMismatch = 0;
  t.steps.forEach(function(s) {
    if (s.live_phase === 'explore') nExplore++;
    else nAct++;
    if (s.cached_phase && s.cached_phase !== s.live_phase) nMismatch++;
  });

  document.getElementById('stats').textContent =
    t.trace_id + ' | model=' + t.model + ' | ' +
    t.n_steps + ' steps | ' +
    nExplore + ' explore, ' + nAct + ' act' +
    (nMismatch > 0 ? ', ' + nMismatch + ' mismatches' : '');

  var html = '';
  t.steps.forEach(function(s) {
    var phase = s.live_phase;
    if (!showAll) {
      if (phase === 'explore' && !showExplore) return;
      if (phase === 'act' && !showAct) return;
    }
    var isMismatch = s.cached_phase && s.cached_phase !== s.live_phase;
    if (onlyMismatch && !isMismatch) return;

    html += '<div class="step' + (isMismatch ? ' mismatch' : '') + '">';
    html += '<div class="step-num">' + s.step_index + '</div>';
    html += '<div class="phase-badge ' + phase + '">' + phase + '</div>';
    html += '<div class="step-content">' + esc(s.action) + '</div>';
    html += '<div class="validity ' + s.validity + '">' + s.validity + '</div>';
    html += '</div>';
  });

  document.getElementById('main').innerHTML = html;
  window.scrollTo(0, 0);
}

function esc(s) {
  if (!s) return '';
  var d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowLeft') prev();
  if (e.key === 'ArrowRight') next();
});

init();
</script>
</body>
</html>"""


class Handler(SimpleHTTPRequestHandler):
    data = []

    def do_GET(self):
        if urlparse(self.path).path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.data).encode())
        elif urlparse(self.path).path in ("/", ""):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/review_step_phase.py <cache.jsonl>")
        return

    cache_path = sys.argv[1]

    # Try to load source traces for action text.
    source_traces = None
    store = GradedTraceStore(cache_path)
    first = next(iter(store), None)
    if first:
        source = first.source
        trace_ids = store.completed_trace_ids()
        print(f"Detected source: {source}, {len(trace_ids)} traces")

        try:
            if source == "autoswe":
                from inspect_degradation.datasets.autoswe import load_autoswe_jsonl
                jsonl = "data/autoswe-traces.jsonl"
                if Path(jsonl).exists():
                    source_traces = load_autoswe_jsonl(jsonl, min_steps=1)
                else:
                    print(f"Could not find {jsonl}")
            elif source == "nebius":
                from inspect_degradation.datasets.nebius import load_nebius
                source_traces = load_nebius(limit=len(trace_ids) * 3, one_per_instance=True)
            elif source == "swe-smith":
                from inspect_degradation.datasets.swe_smith import load_swe_smith
                source_traces = load_swe_smith(limit=len(trace_ids) * 3, one_per_instance=True)
            elif "multi-swebench" in (source or ""):
                print("Multi-SWE-bench source — action text from cache only")
            print(f"Loaded {len(source_traces)} source traces" if source_traces else "No source traces loaded")
        except Exception as e:
            print(f"Could not load source traces: {e}")

    data = _load_data(cache_path, source_loader=source_traces)
    print(f"Loaded {len(data)} traces, {sum(d['n_steps'] for d in data)} steps")

    # Summary stats.
    total_explore = sum(1 for d in data for s in d["steps"] if s["live_phase"] == "explore")
    total_act = sum(1 for d in data for s in d["steps"] if s["live_phase"] == "act")
    total = total_explore + total_act
    print(f"Phase distribution: {total_explore} explore ({total_explore/total*100:.1f}%), {total_act} act ({total_act/total*100:.1f}%)")

    Handler.data = data
    print(f"\nOpen http://localhost:{PORT}")
    server = HTTPServer(("localhost", PORT), Handler)
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
