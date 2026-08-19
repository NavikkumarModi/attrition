"""A single self-contained HTML dashboard for a population-simulation trace.

Takes the same plain list-of-dicts trace shape every `simulate_population*`
function and `TraceStore.read()` already produce. No new dependency: the
aggregation happens in Python, gets embedded as inline JSON, and four
`<canvas>` panels are drawn with vanilla JavaScript -- no CDN, no server, no
fonts. The file opens correctly straight from `file://` in any browser.
"""

import json

__all__ = ["render_dashboard"]

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background: #0b0f14; color: #e6edf3; font-family: -apple-system,
         BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0;
         padding: 32px; }}
  h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 4px; }}
  .sub {{ color: #8b98a5; font-size: 13px; margin: 0 0 28px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
          gap: 20px; }}
  .card {{ background: #131a22; border: 1px solid #1f2933; border-radius: 10px;
          padding: 16px; }}
  .card h2 {{ font-size: 13px; font-weight: 600; color: #8b98a5;
             text-transform: uppercase; letter-spacing: .04em; margin: 0 0 12px; }}
  canvas {{ width: 100%; display: block; }}
  .empty {{ color: #8b98a5; font-size: 13px; padding: 40px 0; text-align: center; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="sub">{n_rows} decisions across {n_agents} agents, {n_rounds} rounds</p>
<div class="grid">
  <div class="card"><h2>Cumulative system value</h2><canvas id="value" height="220"></canvas></div>
  <div class="card"><h2>Arms alive over time</h2><canvas id="burden" height="220"></canvas></div>
  <div class="card"><h2>Value by agent</h2><canvas id="agents" height="220"></canvas></div>
  {graph_card}
</div>
<script>
const TRACE = {trace_json};
const AGG = {agg_json};
const GRAPH = {graph_json};

function fitCanvas(c) {{
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth || 340, h = c.height;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  return {{ctx, w, h}};
}}

function drawLine(canvasId, xs, ys, color) {{
  const c = document.getElementById(canvasId);
  if (!c || xs.length === 0) return;
  const {{ctx, w, h}} = fitCanvas(c);
  const pad = 28;
  const xmin = Math.min(...xs), xmax = Math.max(...xs, xmin + 1);
  const ymin = Math.min(0, ...ys), ymax = Math.max(...ys, ymin + 1);
  const X = x => pad + (x - xmin) / (xmax - xmin) * (w - 2 * pad);
  const Y = y => h - pad - (y - ymin) / (ymax - ymin) * (h - 2 * pad);
  ctx.strokeStyle = "#2a3540"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, Y(0)); ctx.lineTo(w - pad, Y(0)); ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  xs.forEach((x, i) => {{ const px = X(x), py = Y(ys[i]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }});
  ctx.stroke();
  ctx.fillStyle = color;
  xs.forEach((x, i) => {{ ctx.beginPath(); ctx.arc(X(x), Y(ys[i]), 2.5, 0, 7); ctx.fill(); }});
}}

function drawStep(canvasId, xs, ys, color) {{
  const c = document.getElementById(canvasId);
  if (!c || xs.length === 0) return;
  const {{ctx, w, h}} = fitCanvas(c);
  const pad = 28;
  const xmin = Math.min(...xs), xmax = Math.max(...xs, xmin + 1);
  const ymax = Math.max(...ys, 1);
  const X = x => pad + (x - xmin) / (xmax - xmin) * (w - 2 * pad);
  const Y = y => h - pad - y / ymax * (h - 2 * pad);
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  ctx.moveTo(X(xs[0]), Y(ys[0]));
  for (let i = 1; i < xs.length; i++) {{
    ctx.lineTo(X(xs[i]), Y(ys[i - 1])); ctx.lineTo(X(xs[i]), Y(ys[i]));
  }}
  ctx.stroke();
}}

function drawBars(canvasId, labels, values, color) {{
  const c = document.getElementById(canvasId);
  if (!c || labels.length === 0) return;
  const {{ctx, w, h}} = fitCanvas(c);
  const padLeft = 40, padRight = 16, padBottom = 44, n = labels.length;
  const vmax = Math.max(...values, 1);
  const plotW = w - padLeft - padRight;
  const barW = plotW / n * 0.7;
  const fontSize = Math.max(7, Math.min(10, (plotW / n) / 6));
  ctx.font = `${{fontSize}}px sans-serif`; ctx.fillStyle = "#8b98a5";
  labels.forEach((label, i) => {{
    const cx = padLeft + plotW * (i + 0.5) / n;
    const barH = (h - padBottom - 8) * values[i] / vmax;
    ctx.fillStyle = color;
    ctx.fillRect(cx - barW / 2, h - padBottom - barH, barW, barH);
    ctx.fillStyle = "#8b98a5";
    ctx.save();
    ctx.translate(cx, h - padBottom + 14);
    ctx.rotate(-Math.PI / 7);
    ctx.textAlign = "right";
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }});
}}

function drawGraph(canvasId, agents, edges) {{
  const c = document.getElementById(canvasId);
  if (!c || agents.length === 0) return;
  const {{ctx, w, h}} = fitCanvas(c);
  const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 30;
  const pos = {{}};
  agents.forEach((a, i) => {{
    const angle = 2 * Math.PI * i / agents.length - Math.PI / 2;
    pos[a] = [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
  }});
  ctx.strokeStyle = "#2a3540"; ctx.lineWidth = 1;
  edges.forEach(([a, b]) => {{
    if (!pos[a] || !pos[b]) return;
    ctx.beginPath(); ctx.moveTo(...pos[a]); ctx.lineTo(...pos[b]); ctx.stroke();
  }});
  ctx.font = "10px sans-serif"; ctx.textAlign = "center";
  agents.forEach(a => {{
    const [x, y] = pos[a];
    ctx.fillStyle = "#4f8cff"; ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.fill();
    ctx.fillStyle = "#e6edf3"; ctx.fillText(a, x, y - 12);
  }});
}}

if (AGG.ts.length === 0) {{
  document.querySelector(".grid").innerHTML =
    '<div class="card"><p class="empty">No trace rows.</p></div>';
}} else {{
  drawLine("value", AGG.ts, AGG.cum_value, "#4f8cff");
  drawStep("burden", AGG.ts, AGG.alive, "#f2994a");
  drawBars("agents", AGG.agent_labels, AGG.agent_values, "#27ae60");
  if (GRAPH) drawGraph("graph", GRAPH.agents, GRAPH.edges);
}}
</script>
</body>
</html>
"""


def _aggregate(trace):
    by_t_value = {}
    by_t_alive = {}
    by_agent_value = {}
    for row in trace:
        by_t_value[row["t"]] = by_t_value.get(row["t"], 0.0) + row["value"]
        by_t_alive[row["t"]] = row["alive"]
        by_agent_value[row["agent"]] = by_agent_value.get(row["agent"], 0.0) + row["value"]
    ts = sorted(by_t_value)
    cum, total = [], 0.0
    for t in ts:
        total += by_t_value[t]
        cum.append(total)
    alive = [by_t_alive[t] for t in ts]
    agent_labels = sorted(by_agent_value)
    agent_values = [by_agent_value[a] for a in agent_labels]
    return {"ts": ts, "cum_value": cum, "alive": alive,
            "agent_labels": agent_labels, "agent_values": agent_values}


def render_dashboard(trace, path="dashboard.html", title=None, graph=None):
    """Write a self-contained HTML dashboard for `trace` to `path`.

    `trace`: the plain list-of-dicts shape `simulate_population*` and
    `TraceStore.read()` already produce.
    `graph`: optional `network.AgentGraph` -- when given, adds a fifth panel
    drawing the agent network, so the effect of peer visibility (see
    `population.simulate_population_simultaneous`'s `graph` argument) can
    actually be seen, not just read as two numbers.
    """
    agg = _aggregate(trace)
    n_agents = len({row["agent"] for row in trace})
    n_rounds = len(agg["ts"])
    graph_card = ""
    graph_json = "null"
    if graph is not None:
        graph_card = ('<div class="card"><h2>Agent network</h2>'
                      '<canvas id="graph" height="220"></canvas></div>')
        graph_json = json.dumps({
            "agents": graph.agent_ids,
            "edges": [[a, b] for a in graph.agent_ids
                     for b in graph.neighbors(a) if a < b],
        })
    html = _TEMPLATE.format(
        title=title or "Population run",
        n_rows=len(trace), n_agents=n_agents, n_rounds=n_rounds,
        graph_card=graph_card,
        trace_json=json.dumps(trace),
        agg_json=json.dumps(agg),
        graph_json=graph_json,
    )
    with open(path, "w") as f:
        f.write(html)
    return path
