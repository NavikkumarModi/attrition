"""Regenerate the PNGs embedded in the top-level README, from live runs --
not hand-drawn -- matching how paper/make_figures.py generates the paper's
figures from live experiments.

    headline.png     the "one-table version" comparison, as a chart
    dashboard.png    a screenshot of a real render_dashboard() output
    pharma_demo.png  hero image: the pharma peer-influence population --
                     agent network, value/burden curves, real verdict text

`dashboard.png` needs a local Chrome/Chromium binary (headless screenshot);
if none is found this script regenerates the other two only and says so.

Run from the repo root:  PYTHONPATH=. python docs/make_readme_images.py
"""

import http.server
import math
import os
import shutil
import subprocess
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attrition import (AgentGraph, ConsumableBandit, ECI, Greedy, MockLLMClient,
                       PHARMA_PERSONAS, Population, SimultaneousPool,
                       ThompsonSampling, UCB, Conservative, compare,
                       derive_antibiotic_parameters, render_dashboard,
                       simulate_population_simultaneous)

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "images")
BG = "#0b0f14"
CARD = "#131a22"
GRID = "#2a3540"
MUTE = "#8b98a5"
TEXT = "#e6edf3"
BLUE = "#4f8cff"
ORANGE = "#f2994a"
GREEN = "#27ae60"


def make_headline():
    env = lambda seed: ConsumableBandit.random(n=20, k_spread=1.5, delta=0.03,
                                                horizon=60, seed=seed)
    results = compare(env, [Greedy(), ThompsonSampling(), UCB(),
                            Conservative(e_bound=2.0), ECI()], seeds=25)
    names = list(results.keys())
    values = [results[n]["value"] for n in names]
    regrets = [results[n]["regret"] for n in names]
    colors = ["#e5484d" if n == "greedy" else
              ("#27ae60" if n == "eci" else "#4f8cff") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color("#2a3540")
        ax.tick_params(colors="#8b98a5", labelsize=9)
        ax.title.set_color("#e6edf3")
    axes[0].bar(names, values, color=colors)
    axes[0].set_title("value  (higher is better)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, regrets, color=colors)
    axes[1].set_title("regret  (lower looks better -- but is not)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("The regret column and the value column are ordered backwards",
                 color="#e6edf3", fontsize=11)
    fig.tight_layout()
    out = os.path.join(IMAGES, "headline.png")
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


def _style_axis(ax):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTE, labelsize=8)
    ax.title.set_color(TEXT)
    ax.title.set_fontsize(10)


def _draw_network(ax, agent_ids, graph):
    n = len(agent_ids)
    pos = {}
    for i, a in enumerate(agent_ids):
        angle = 2 * math.pi * i / n - math.pi / 2
        pos[a] = (math.cos(angle), math.sin(angle))
    for a in agent_ids:
        for b in graph.neighbors(a):
            if a < b:
                (x1, y1), (x2, y2) = pos[a], pos[b]
                ax.plot([x1, x2], [y1, y2], color=GRID, linewidth=1, zorder=1)
    xs = [pos[a][0] for a in agent_ids]
    ys = [pos[a][1] for a in agent_ids]
    ax.scatter(xs, ys, s=220, color=BLUE, zorder=2, edgecolors=BG, linewidths=1.5)
    for a in agent_ids:
        x, y = pos[a]
        ax.annotate(a, (x, y), xytext=(0, 14), textcoords="offset points",
                   ha="center", fontsize=7.5, color=TEXT)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("prescriber network", fontsize=9, color=TEXT)


def make_pharma_demo():
    """Hero image: the pharma peer-influence population -- agent network,
    real value/burden curves from a live run, and a dynamically computed
    verdict line (not a canned caption).
    """
    v, p, e, _ = derive_antibiotic_parameters()
    prescribers = [PHARMA_PERSONAS["dr-conservative"], PHARMA_PERSONAS["dr-balanced"],
                   PHARMA_PERSONAS["dr-aggressive"], PHARMA_PERSONAS["pharmacist-formulary"]]
    agent_ids = [p_.name for p_ in prescribers]
    m = len(prescribers)
    graph = AgentGraph.complete(agent_ids)
    # seed=2 chosen only because it's one of the runs where the pool actually
    # depletes across multiple rounds rather than converging after round 0 --
    # still a real, reproducible run, not hand-edited data.
    pool = SimultaneousPool(v, p, e, delta=0.35, horizon=9, n_agents=m, seed=2)
    population = Population.from_personas(prescribers,
                                          client=MockLLMClient(seed=0, conformity=0.3))
    result = simulate_population_simultaneous(pool, population, rounds=9, graph=graph)
    trace = result["trace"]

    by_t_value, by_t_alive = {}, {}
    for row in trace:
        by_t_value[row["t"]] = by_t_value.get(row["t"], 0.0) + row["value"]
        by_t_alive[row["t"]] = row["alive"]
    ts = sorted(by_t_value)
    cum, total = [], 0.0
    for t in ts:
        total += by_t_value[t]
        cum.append(total)
    alive = [by_t_alive[t] for t in ts]

    fig = plt.figure(figsize=(11, 4.2), facecolor=BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1.3], wspace=0.35,
                          left=0.07, right=0.96, top=0.78, bottom=0.14)

    ax_net = fig.add_subplot(gs[0, 0])
    ax_net.set_facecolor(BG)
    _draw_network(ax_net, agent_ids, graph)

    ax_val = fig.add_subplot(gs[0, 1])
    _style_axis(ax_val)
    ax_val.plot(ts, cum, color=BLUE, marker="o", markersize=3, linewidth=2)
    ax_val.set_title("cumulative system value")
    ax_val.set_xlabel("round", color=MUTE, fontsize=8)

    ax_burden = fig.add_subplot(gs[0, 2])
    _style_axis(ax_burden)
    ax_burden.step(ts, alive, where="post", color=ORANGE, linewidth=2)
    ax_burden.set_title("arms alive (resistance pool depleting)")
    ax_burden.set_xlabel("round", color=MUTE, fontsize=8)

    fig.suptitle("Pharma population: antibiotic stewardship under peer visibility",
                color=TEXT, fontsize=13, fontweight="bold", y=0.97)
    fig.text(0.5, 0.885,
             f"4 prescriber personas, mechanistically-derived resistance "
             f"dynamics -- system_value={result['system_value']:.2f}  "
             f"system_regret={result['system_regret']:.2f}",
             color=MUTE, fontsize=9, ha="center")

    out = os.path.join(IMAGES, "pharma_demo.png")
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


def _find_chrome():
    for candidate in (
        shutil.which("google-chrome"), shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def make_dashboard():
    chrome = _find_chrome()
    if chrome is None:
        print("no Chrome/Chromium found -- skipping dashboard.png "
              "(headline.png was still regenerated)")
        return

    v, p, e, _ = derive_antibiotic_parameters()
    prescribers = [PHARMA_PERSONAS["dr-conservative"], PHARMA_PERSONAS["dr-balanced"],
                   PHARMA_PERSONAS["dr-aggressive"], PHARMA_PERSONAS["pharmacist-formulary"]]
    m = len(prescribers)
    graph = AgentGraph.complete([p_.name for p_ in prescribers])
    pool = SimultaneousPool(v, p, e, delta=0.35, horizon=9, n_agents=m, seed=0)
    population = Population.from_personas(prescribers,
                                          client=MockLLMClient(seed=0, conformity=0.3))
    result = simulate_population_simultaneous(pool, population, rounds=9, graph=graph)

    html_path = os.path.join(IMAGES, "_dashboard_tmp.html")
    render_dashboard(result["trace"], path=html_path, title="Peer-influence run",
                     graph=graph)

    port = 8934
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(
        *a, directory=IMAGES, **kw)
    server = http.server.HTTPServer(("localhost", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        out = os.path.join(IMAGES, "dashboard.png")
        subprocess.run([
            chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
            "--window-size=900,700", f"--screenshot={out}",
            f"http://localhost:{port}/_dashboard_tmp.html",
        ], check=True, capture_output=True)
        print(f"wrote {out}")
    finally:
        server.shutdown()
        os.remove(html_path)


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    make_headline()
    make_pharma_demo()
    make_dashboard()
