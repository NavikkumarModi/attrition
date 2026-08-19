"""Regenerate the two PNGs embedded in the top-level README, from live runs
-- not hand-drawn -- matching how paper/make_figures.py generates the paper's
figures from live experiments.

    headline.png    the "one-table version" comparison, as a chart
    dashboard.png   a screenshot of a real render_dashboard() output

`dashboard.png` needs a local Chrome/Chromium binary (headless screenshot);
if none is found this script regenerates `headline.png` only and says so.

Run from the repo root:  PYTHONPATH=. python docs/make_readme_images.py
"""

import http.server
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
    make_dashboard()
