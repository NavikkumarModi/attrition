"""A single self-contained HTML dashboard for a population-simulation trace.

Takes the same plain list-of-dicts trace shape every `simulate_population*`
function and `TraceStore.read()` already produce. No new dependency: the
aggregation happens in Python, gets embedded as inline JSON, and the panels
are drawn with vanilla-canvas JavaScript -- no CDN, no server, no fonts. The
file opens correctly straight from `file://` in any browser.

Deliberately a single fixed dark theme rather than a light/dark toggle: this
is a generated report page, not a product surface with a persistent viewer
preference to respect, and the categorical palette (see module docstring
below) is validated against exactly the dark surface it renders on.
"""

import json

__all__ = ["render_dashboard"]

# Validated categorical palette (dataviz skill reference palette, dark steps):
# fixed hue order, never cycled per-instance -- see attrition/dashboard.py's
# module docstring. Worst adjacent CVD deltaE 8.4, worst adjacent
# normal-vision deltaE 19.3 (OKLab x100) on the #1a1a19 dark surface.
_SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500",
          "#d55181", "#008300", "#9085e9", "#e66767"]

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --page: #0d0d0d; --surface: #1a1a19; --text: #ffffff;
    --text-sec: #c3c2b7; --text-muted: #898781; --grid: #2c2c2a;
    --axis: #383835; --border: rgba(255,255,255,0.10);
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--page); color: var(--text); font-family: system-ui,
         -apple-system, "Segoe UI", sans-serif; margin: 0; padding: 40px; }}
  h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 16px; letter-spacing: -.01em; }}
  .stats {{ display: flex; gap: 12px; margin: 0 0 32px; flex-wrap: wrap; }}
  .stat {{ background: var(--surface); border: 1px solid var(--border);
          border-radius: 10px; padding: 12px 18px; min-width: 110px; }}
  .stat .label {{ font-size: 11px; color: var(--text-muted);
                 text-transform: uppercase; letter-spacing: .04em; margin: 0 0 4px; }}
  .stat .value {{ font-size: 22px; font-weight: 600; color: var(--text); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
          gap: 20px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
          border-radius: 12px; padding: 20px; }}
  .card h2 {{ font-size: 13px; font-weight: 600; color: var(--text-sec);
             text-transform: uppercase; letter-spacing: .04em; margin: 0 0 4px; }}
  .card .card-sub {{ font-size: 12px; color: var(--text-muted); margin: 0 0 14px; }}
  canvas {{ width: 100%; display: block; cursor: crosshair; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 12px; }}
  .legend .item {{ display: flex; align-items: center; gap: 6px; font-size: 12px;
                  color: var(--text-sec); }}
  .legend .swatch {{ width: 10px; height: 10px; border-radius: 3px; flex: none; }}
  .empty {{ color: var(--text-muted); font-size: 13px; padding: 48px 0; text-align: center; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="stats">
  <div class="stat"><div class="label">Decisions</div><div class="value">{n_rows}</div></div>
  <div class="stat"><div class="label">Agents</div><div class="value">{n_agents}</div></div>
  <div class="stat"><div class="label">Rounds</div><div class="value">{n_rounds}</div></div>
</div>
<div class="grid">
  <div class="card">
    <h2>Cumulative system value</h2>
    <p class="card-sub">Total realised value across all agents, running sum by round</p>
    <canvas id="value" height="240"></canvas>
  </div>
  <div class="card">
    <h2>Arms alive over time</h2>
    <p class="card-sub">Pool size remaining, by round</p>
    <canvas id="burden" height="240"></canvas>
  </div>
  <div class="card">
    <h2>Value by agent</h2>
    <p class="card-sub">Total realised value, summed over the whole run</p>
    <canvas id="agents" height="240"></canvas>
    <div class="legend" id="agents-legend"></div>
  </div>
  {graph_card}
</div>
<script>
const TRACE = {trace_json};
const AGG = {agg_json};
const GRAPH = {graph_json};

const PALETTE = {{
  page: "#0d0d0d", surface: "#1a1a19", text: "#ffffff", textSec: "#c3c2b7",
  textMuted: "#898781", grid: "#2c2c2a", axis: "#383835",
  border: "rgba(255,255,255,0.10)",
  series: {series_json},
}};
const FONT = "11px system-ui, -apple-system, sans-serif";

function withAlpha(hex, a) {{
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16),
       b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

function fmtNum(v) {{
  if (Math.abs(v) >= 1000) return (v / 1000).toFixed(1) + "k";
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}}

function niceTicks(min, max, count) {{
  if (min === max) {{ min -= 1; max += 1; }}
  const rawStep = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = niceMin; v <= niceMax + step * 0.5; v += step) {{
    ticks.push(Math.round(v * 1e6) / 1e6);
  }}
  return ticks;
}}

function pickXTicks(xs, count) {{
  if (xs.length <= count) return xs.map((_, i) => i);
  const xmin = xs[0], xmax = xs[xs.length - 1];
  const step = Math.max(1, Math.round((xmax - xmin) / (count - 1)));
  const idx = [];
  for (let v = xmin; v <= xmax; v += step) {{
    let best = 0, bestD = Infinity;
    xs.forEach((x, i) => {{ const d = Math.abs(x - v); if (d < bestD) {{ bestD = d; best = i; }} }});
    idx.push(best);
  }}
  if (idx[idx.length - 1] !== xs.length - 1) idx.push(xs.length - 1);
  return [...new Set(idx)];
}}

function roundRectTop(ctx, x, y, w, h, r) {{
  r = Math.max(0, Math.min(r, w / 2, h));
  ctx.beginPath();
  ctx.moveTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h);
  ctx.closePath();
}}

function fitCanvas(c) {{
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth || 340, h = c.height;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {{ctx, w, h}};
}}

function drawTooltip(ctx, canvasW, x, y, text, above) {{
  ctx.font = FONT;
  const tw = ctx.measureText(text).width + 16;
  const th = 24;
  let tx = x + 10;
  if (tx + tw > canvasW) tx = x - tw - 10;
  let ty = above ? y - th - 8 : y + 8;
  ctx.fillStyle = PALETTE.surface;
  ctx.strokeStyle = PALETTE.border;
  ctx.lineWidth = 1;
  roundRectTop(ctx, tx, ty, tw, th, 6);
  ctx.lineTo(tx, ty + th);
  ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = PALETTE.text;
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  ctx.fillText(text, tx + 8, ty + th / 2 + 1);
}}

// -------------------------------------------------------------- line/step
function drawSeriesChart(canvasId, xs, ys, color, {{step = false, label}} = {{}}) {{
  const c = document.getElementById(canvasId);
  if (!c || xs.length === 0) return;
  const {{ctx, w, h}} = fitCanvas(c);
  const padL = 44, padR = 12, padT = 10, padB = 26;
  const xmin = Math.min(...xs), xmax = Math.max(...xs, xmin + 1);
  const ymin0 = Math.min(0, ...ys), ymax0 = Math.max(...ys, ymin0 + 1);
  const yTicks = niceTicks(ymin0, ymax0, 4);
  const ymin = yTicks[0], ymax = yTicks[yTicks.length - 1];
  const X = x => padL + (x - xmin) / (xmax - xmin) * (w - padL - padR);
  const Y = y => h - padB - (y - ymin) / (ymax - ymin) * (h - padT - padB);

  function base() {{
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = PALETTE.grid; ctx.lineWidth = 1;
    ctx.fillStyle = PALETTE.textMuted; ctx.font = FONT;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    yTicks.forEach(t => {{
      const y = Y(t);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(fmtNum(t), padL - 8, y);
    }});
    ctx.strokeStyle = PALETTE.axis;
    ctx.beginPath(); ctx.moveTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    pickXTicks(xs, 6).forEach(i => ctx.fillText(String(xs[i]), X(xs[i]), h - padB + 6));

    const path = new Path2D();
    if (step) {{
      path.moveTo(X(xs[0]), Y(ys[0]));
      for (let i = 1; i < xs.length; i++) {{
        path.lineTo(X(xs[i]), Y(ys[i - 1]));
        path.lineTo(X(xs[i]), Y(ys[i]));
      }}
    }} else {{
      xs.forEach((x, i) => {{
        const px = X(x), py = Y(ys[i]);
        i === 0 ? path.moveTo(px, py) : path.lineTo(px, py);
      }});
    }}
    const area = new Path2D(path);
    area.lineTo(X(xs[xs.length - 1]), Y(ymin));
    area.lineTo(X(xs[0]), Y(ymin));
    area.closePath();
    ctx.fillStyle = withAlpha(color, 0.10);
    ctx.fill(area);
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.stroke(path);

    const lx = xs.length - 1;
    ctx.fillStyle = PALETTE.surface;
    ctx.beginPath(); ctx.arc(X(xs[lx]), Y(ys[lx]), 5, 0, 7); ctx.fill();
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(X(xs[lx]), Y(ys[lx]), 3.5, 0, 7); ctx.fill();
  }}
  base();

  c.addEventListener("mousemove", e => {{
    const rect = c.getBoundingClientRect();
    const mx = (e.clientX - rect.left);
    let best = 0, bestD = Infinity;
    xs.forEach((x, i) => {{ const d = Math.abs(X(x) - mx); if (d < bestD) {{ bestD = d; best = i; }} }});
    base();
    const px = X(xs[best]), py = Y(ys[best]);
    ctx.save();
    ctx.strokeStyle = PALETTE.axis; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, h - padB); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = PALETTE.surface;
    ctx.beginPath(); ctx.arc(px, py, 5, 0, 7); ctx.fill();
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(px, py, 3.5, 0, 7); ctx.fill();
    const text = `${{label || "t"}}=${{xs[best]}}   ${{fmtNum(ys[best])}}`;
    drawTooltip(ctx, w, px, py, text, py < h / 2 ? false : true);
  }});
  c.addEventListener("mouseleave", base);
}}

// ----------------------------------------------------------------- bars
function drawBarChart(canvasId, legendId, labels, values, colors) {{
  const c = document.getElementById(canvasId);
  if (!c || labels.length === 0) return;
  const {{ctx, w, h}} = fitCanvas(c);
  const padL = 44, padR = 12, padT = 28, padB = 34;
  const n = labels.length;
  const yTicks = niceTicks(0, Math.max(...values, 1), 4);
  const ymax = yTicks[yTicks.length - 1];
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const slot = plotW / n;
  const barW = Math.min(48, slot * 0.6);
  const Y = v => h - padB - v / ymax * plotH;

  function base() {{
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = PALETTE.grid; ctx.lineWidth = 1;
    ctx.fillStyle = PALETTE.textMuted; ctx.font = FONT;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    yTicks.forEach(t => {{
      const y = Y(t);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(fmtNum(t), padL - 8, y);
    }});
    ctx.strokeStyle = PALETTE.axis;
    ctx.beginPath(); ctx.moveTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();

    labels.forEach((label, i) => {{
      const cx = padL + slot * (i + 0.5);
      const barH = Math.max(1, (h - padB) - Y(values[i]));
      ctx.fillStyle = colors[i];
      roundRectTop(ctx, cx - barW / 2, Y(values[i]), barW, barH, 4);
      ctx.fill();
      ctx.fillStyle = PALETTE.text; ctx.font = "600 11px system-ui, sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(fmtNum(values[i]), cx, Math.max(14, Y(values[i]) - 6));
    }});
  }}
  base();

  const legend = document.getElementById(legendId);
  if (legend) {{
    legend.innerHTML = labels.map((label, i) =>
      `<span class="item"><span class="swatch" style="background:${{colors[i]}}"></span>${{label}}</span>`
    ).join("");
  }}

  c.addEventListener("mousemove", e => {{
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    let i = Math.floor((mx - padL) / slot);
    if (i < 0 || i >= n) {{ base(); return; }}
    base();
    const cx = padL + slot * (i + 0.5);
    const top = Y(values[i]);
    ctx.fillStyle = withAlpha(colors[i], 0.15);
    ctx.fillRect(padL + slot * i, padT, slot, h - padB - padT);
    drawTooltip(ctx, w, cx, top, `${{labels[i]}}: ${{fmtNum(values[i])}}`, true);
  }});
  c.addEventListener("mouseleave", base);
}}

// --------------------------------------------------------------- graph
function drawGraph(canvasId, agents, edges) {{
  const c = document.getElementById(canvasId);
  if (!c || agents.length === 0) return;
  const {{ctx, w, h}} = fitCanvas(c);
  const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 36;
  const pos = {{}};
  agents.forEach((a, i) => {{
    const angle = 2 * Math.PI * i / agents.length - Math.PI / 2;
    pos[a] = [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
  }});

  function base(hover) {{
    ctx.clearRect(0, 0, w, h);
    edges.forEach(([a, b]) => {{
      if (!pos[a] || !pos[b]) return;
      const active = hover && (hover === a || hover === b);
      ctx.strokeStyle = active ? PALETTE.series[1] : PALETTE.grid;
      ctx.lineWidth = active ? 2 : 1;
      ctx.beginPath(); ctx.moveTo(...pos[a]); ctx.lineTo(...pos[b]); ctx.stroke();
    }});
    ctx.font = "600 11px system-ui, sans-serif"; ctx.textAlign = "center";
    agents.forEach(a => {{
      const [x, y] = pos[a];
      const active = hover === a;
      ctx.fillStyle = PALETTE.surface;
      ctx.beginPath(); ctx.arc(x, y, active ? 9 : 7, 0, 7); ctx.fill();
      ctx.fillStyle = active ? PALETTE.series[1] : PALETTE.series[0];
      ctx.beginPath(); ctx.arc(x, y, active ? 7 : 5.5, 0, 7); ctx.fill();
      ctx.fillStyle = active ? PALETTE.text : PALETTE.textSec;
      ctx.fillText(a, x, y - 14);
    }});
  }}
  base(null);

  c.addEventListener("mousemove", e => {{
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let hover = null;
    for (const a of agents) {{
      const [x, y] = pos[a];
      if (Math.hypot(x - mx, y - my) < 12) {{ hover = a; break; }}
    }}
    base(hover);
  }});
  c.addEventListener("mouseleave", () => base(null));
}}

if (AGG.ts.length === 0) {{
  document.querySelector(".grid").innerHTML =
    '<div class="card"><p class="empty">No trace rows.</p></div>';
}} else {{
  drawSeriesChart("value", AGG.ts, AGG.cum_value, PALETTE.series[0], {{label: "round"}});
  drawSeriesChart("burden", AGG.ts, AGG.alive, PALETTE.series[1], {{step: true, label: "round"}});
  drawBarChart("agents", "agents-legend", AGG.agent_labels, AGG.agent_values,
              AGG.agent_labels.map((_, i) => i < 8 ? PALETTE.series[i] : PALETTE.textMuted));
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
        graph_card = (
            '<div class="card">'
            '<h2>Agent network</h2>'
            '<p class="card-sub">Who can see whom\'s choices each round</p>'
            '<canvas id="graph" height="240"></canvas>'
            '</div>')
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
        series_json=json.dumps(_SERIES),
    )
    with open(path, "w") as f:
        f.write(html)
    return path
