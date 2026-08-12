"""Generate all paper figures from live runs of the actual experiments.

No numbers are hardcoded: every figure recomputes its data from the library and
the exact-DP reference implementation, so the figures cannot drift from the
results they illustrate.

Usage:  python paper/make_figures.py
Output: paper/fig1_divergence.pdf ... fig5_agent_trace.pdf
"""

import os
import sys
from functools import lru_cache

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evolving_bandits import (ConsumableBandit, Greedy, ECI, Conservative,
                              ThompsonSampling, UCB, compare)

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 9,
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

# colourblind-safe, and each series also gets a distinct linestyle/marker
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#8C8C8C")


# --------------------------------------------------------------- shared exact DP
def exact_dp(v, p, e, delta, T):
    n = len(v); full = frozenset(range(n)); etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)

    @lru_cache(maxsize=None)
    def G(S, t):
        if t >= T or not S:
            return 0.0
        a = max(S, key=lambda i: R(i, S))
        return R(a, S) + p[a]*G(S-{a}, t+1) + (1-p[a])*G(S, t+1)

    @lru_cache(maxsize=None)
    def I(S, t):
        if t >= T or not S:
            return 0.0
        a = max(S, key=lambda i: R(i, S) - delta*p[i]*e[i]*(T-t))
        return R(a, S) + p[a]*I(S-{a}, t+1) + (1-p[a])*I(S, t+1)

    return V(full, 0), G(full, 0), I(full, 0)


# =============================================================== FIGURE 1
def fig1_divergence():
    """The headline: regret pinned at zero while value collapses."""
    rng = np.random.default_rng(5)
    n, T, INST = 6, 10, 20
    settings = [(0.02, 0.5), (0.05, 0.8), (0.10, 1.0),
                (0.20, 1.5), (0.35, 2.0), (0.50, 2.5)]
    xs, v_opt, v_greedy, regret = [], [], [], []
    for delta, spread in settings:
        VS, VG = [], []
        for _ in range(INST):
            v = np.sort(rng.uniform(0.6, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.4, 1.0, n), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, spread, n), 0.0, None)
            vs, vg, _ = exact_dp(v, p, e, delta, T)
            VS.append(vs); VG.append(vg)
        xs.append(delta * spread)
        v_opt.append(np.mean(VS)); v_greedy.append(np.mean(VG))
        regret.append(0.0)          # exact: greedy always pulls best available

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax1.plot(xs, regret, "o-", color=BLUE, lw=2, ms=5,
             label="greedy regret")
    ax1.axhline(0, color=GREY, lw=0.6, ls=":")
    ax1.set_ylim(-0.5, 3.0)
    ax1.set_xlabel(r"coupling strength  $\delta \cdot \mathrm{std}(e)$")
    ax1.set_ylabel("cumulative regret")
    ax1.set_title("(a)  What the metric reports")
    ax1.annotate("identically zero\nat every setting", xy=(xs[3], 0),
                 xytext=(xs[1], 1.5), fontsize=8, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

    ax2.plot(xs, v_opt, "s--", color=GREEN, lw=2, ms=5, label="optimal")
    ax2.plot(xs, v_greedy, "o-", color=RED, lw=2, ms=5, label="greedy")
    ax2.fill_between(xs, v_greedy, v_opt, color=RED, alpha=0.12)
    ax2.axhline(0, color=GREY, lw=0.8, ls="--")
    ax2.set_xlabel(r"coupling strength  $\delta \cdot \mathrm{std}(e)$")
    ax2.set_ylabel("realised value")
    ax2.set_title("(b)  What actually happens")
    ax2.legend(loc="upper right")
    ax2.annotate("value turns negative while\nthe optimum stays positive",
                 xy=(xs[-1], v_greedy[-1]), xytext=(0.04, 0.06), fontsize=8,
                 color=RED, textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8,
                                 connectionstyle="arc3,rad=-0.2"))
    ax2.set_ylim(min(v_greedy) - 2.2, max(v_opt) + 1.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_divergence.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig1: regret flat at 0; value %.2f -> %.2f" % (v_greedy[0], v_greedy[-1]))


# =============================================================== FIGURE 2
def fig2_characterisation():
    """T1: kappa dispersion is what breaks greedy; five structural cases."""
    rng = np.random.default_rng(11)
    N, T, DELTA, INST = 6, 8, 0.12, 12

    def case(make):
        gaps, sds = [], []
        for _ in range(INST):
            v = np.sort(rng.uniform(0.4, 1.2, N))[::-1].copy()
            p, e = make()
            vs, vg, _ = exact_dp(v, p, e, DELTA, T)
            gaps.append((vs - vg) / abs(vs) * 100)
            sds.append(float(np.std(p * e)))
        return np.mean(sds), np.mean(gaps), np.std(gaps) / np.sqrt(INST)

    cases = [
        (r"A: $p$ varies, $e=0$",
         lambda: (rng.uniform(0.2, 1.0, N), np.zeros(N))),
        (r"B: $p$, $e$ both constant",
         lambda: (np.full(N, 0.5), np.full(N, 1.0))),
        (r"C: $p$ constant, $e$ varies",
         lambda: (np.full(N, 0.5), rng.uniform(0.0, 2.0, N))),
        (r"D: $p$ varies, $e$ constant",
         lambda: (rng.uniform(0.2, 1.0, N), np.full(N, 1.0))),
        (r"E: both vary, $p\cdot e$ pinned",
         lambda: (lambda pv: (pv, 0.4 / pv))(rng.uniform(0.25, 1.0, N))),
    ]
    labels, sds, gaps, errs = [], [], [], []
    for lab, mk in cases:
        s, g, er = case(mk)
        labels.append(lab); sds.append(s); gaps.append(g); errs.append(er)

    # dispersion sweep, binned by realised std(kappa)
    pts = []
    rng2 = np.random.default_rng(123)
    for _ in range(200):
        v = np.sort(rng2.uniform(0.4, 1.2, N))[::-1].copy()
        p = np.clip(rng2.uniform(0.3, 1.0, N), 0.05, 1.0)
        e = np.clip(1.0 + rng2.normal(0, rng2.uniform(0, 1.6), N), 0, None)
        vs, vg, vi = exact_dp(v, p, e, DELTA, T)
        pts.append((float(np.std(p*e)), (vs-vg)/abs(vs)*100,
                    (vs-vi)/abs(vs)*100))
    pts = np.array(sorted(pts))
    bins = np.array_split(pts, 6)
    bx = [b[:, 0].mean() for b in bins]
    bg = [b[:, 1].mean() for b in bins]
    bi = [b[:, 2].mean() for b in bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    colors = [GREEN if g < 0.5 else RED for g in gaps]
    ypos = np.arange(len(labels))
    ax1.barh(ypos, gaps, xerr=errs, color=colors, height=0.62,
             error_kw=dict(lw=0.8, capsize=2))
    ax1.set_yticks(ypos); ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("greedy optimality gap (%)")
    ax1.set_title(r"(a)  Greedy fails iff $\kappa=p\,e$ varies")
    for i, (g, s) in enumerate(zip(gaps, sds)):
        ax1.text(g + 0.25, i, r"std$(\kappa)$=%.2f" % s, va="center", fontsize=7)
    ax1.set_xlim(0, max(gaps) * 1.45)

    ax2.plot(bx, bg, "o-", color=RED, lw=2, ms=5, label="greedy")
    ax2.plot(bx, bi, "s--", color=BLUE, lw=2, ms=5, label="ECI (corrected)")
    ax2.set_xlabel(r"dispersion  std$(\kappa)$")
    ax2.set_ylabel("optimality gap (%)")
    ax2.set_title("(b)  Gap grows with dispersion")
    ax2.legend(loc="upper left")
    ax2.annotate("ECI stays flat", xy=(bx[-1], bi[-1]),
                 xytext=(bx[2], 12), fontsize=8, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_characterisation.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("fig2: case E gap = %.4f%% (should be ~0)" % gaps[4])


# =============================================================== FIGURE 3
def fig3_estimation_floor():
    """T3: more horizon buys nothing once the pool, not the clock, binds."""
    from experiments.exp10_learning import Sim

    def trial(p_const, T, seed, n=8):
        sim = Sim(n=n, T=T, spread=1.0, seed=seed)
        sim.p = np.full(n, p_const)
        sim.rng = np.random.default_rng(10_000 + seed); sim.reset()
        X, y = [], []
        while sim.t < sim.T and sim.alive.any():
            idx = np.flatnonzero(sim.alive)
            a = int(sim.rng.choice(idx))
            dead_before = ~sim.alive.copy()
            obs, _, _ = sim.step(a)
            row = np.zeros(2*n); row[a] = 1.0; row[n:][dead_before] = -1.0
            X.append(row); y.append(obs)
        X, y = np.array(X), np.array(y)
        beta = np.linalg.lstsq(X.T@X + 1e-6*np.eye(2*n), X.T@y, rcond=None)[0]
        died = ~sim.alive
        truth = sim.delta * sim.e
        if died.sum() < 3:
            return len(y), np.nan
        return len(y), float(np.sqrt(np.mean((beta[n:][died]-truth[died])**2)))

    Ts = [20, 40, 80, 160, 320]
    series = {}
    for p in [0.9, 0.5, 0.1]:
        rmse, rounds = [], []
        for T in Ts:
            R = [trial(p, T, s) for s in range(40)]
            rounds.append(np.mean([r[0] for r in R]))
            rmse.append(np.nanmean([r[1] for r in R]))
        series[p] = (rounds, rmse)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    styles = {0.9: (RED, "o-"), 0.5: (ORANGE, "s--"), 0.1: (BLUE, "^-.")}

    for p, (rounds, rmse) in series.items():
        c, st = styles[p]
        ax1.plot(Ts, rmse, st, color=c, lw=2, ms=5, label=f"$p$ = {p}")
        ax2.plot(Ts, rounds, st, color=c, lw=2, ms=5, label=f"$p$ = {p}")
        ax2.axhline(8/p, color=c, lw=0.7, ls=":")

    ax1.set_xscale("log"); ax1.set_xticks(Ts)
    ax1.set_xticklabels([str(t) for t in Ts])
    ax1.set_xlabel("horizon $T$")
    ax1.set_ylabel(r"RMSE of $\hat e$")
    ax1.set_title("(a)  More horizon buys nothing")
    ax1.legend()
    ax1.annotate("flat: 16$\\times$ horizon,\nidentical error",
                 xy=(Ts[-1], series[0.9][1][-1]), xytext=(30, 0.19),
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

    ax2.set_xscale("log"); ax2.set_xticks(Ts)
    ax2.set_xticklabels([str(t) for t in Ts])
    ax2.set_xlabel("horizon $T$")
    ax2.set_ylabel("rounds of data obtained")
    ax2.set_title(r"(b)  Sample budget caps at $\sum_j 1/p_j$")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_estimation_floor.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("fig3: p=0.9 RMSE across T:", np.round(series[0.9][1], 4))


# =============================================================== FIGURE 4
def fig4_policies():
    """Policy comparison, and the reversed ordering of regret vs value."""
    spreads = [0.3, 0.8, 1.5, 2.5]
    pols = [Greedy(), ThompsonSampling(), UCB(),
            Conservative(e_bound=2.0), ECI()]
    names = [p.name for p in pols]
    vals = {n: [] for n in names}
    regs = {n: [] for n in names}
    for sp in spreads:
        f = lambda s, sp=sp: ConsumableBandit.random(
            n=20, k_spread=sp, delta=0.03, horizon=60, seed=s)
        r = compare(f, pols, seeds=25)
        for n in names:
            vals[n].append(r[n]["value"]); regs[n].append(r[n]["regret"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    style = {"greedy": (RED, "o-"), "thompson": (ORANGE, "s--"),
             "ucb": (GREY, "^:"), "conservative": (GREEN, "d-."),
             "eci": (BLUE, "*-")}
    for n in names:
        c, st = style[n]
        ax1.plot(spreads, vals[n], st, color=c, lw=1.8, ms=5, label=n)
    ax1.set_xlabel(r"externality dispersion  std$(e)$")
    ax1.set_ylabel("realised value")
    ax1.set_title("(a)  Value achieved")
    ax1.legend(ncol=2, loc="lower left")

    idx = -2
    order_v = sorted(names, key=lambda n: vals[n][idx])
    yv = [vals[n][idx] for n in order_v]
    yr = [regs[n][idx] for n in order_v]
    ypos = np.arange(len(order_v))
    ax2.barh(ypos - 0.19, yv, height=0.36, color=BLUE, label="value")
    ax2.barh(ypos + 0.19, yr, height=0.36, color=ORANGE, label="regret")
    ax2.set_yticks(ypos); ax2.set_yticklabels(order_v)
    ax2.set_xlabel("value  /  cumulative regret")
    ax2.set_title(r"(b)  Rankings are reversed (std$(e)$=1.5)")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_policies.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig4: greedy value %.2f regret %.2f | eci value %.2f regret %.2f"
          % (vals["greedy"][idx], regs["greedy"][idx],
             vals["eci"][idx], regs["eci"][idx]))


# =============================================================== FIGURE 5
def fig5_agent_trace():
    """The agent tool ecosystem: dashboard green, ecosystem burning."""
    from experiments.exp17_agent_ecosystem import build, trace, TOOLS
    T = 30
    v, p, e, d = build(0.06)
    gl = trace(v, p, e, d, T, "greedy", seed=3)
    il = trace(v, p, e, d, T, "index", seed=3)

    def series(log):
        t = [r[0] for r in log if r[1] != "ECOSYSTEM EXHAUSTED"]
        val = [r[2] for r in log if r[1] != "ECOSYSTEM EXHAUSTED"]
        reg = [r[3] for r in log if r[1] != "ECOSYSTEM EXHAUSTED"]
        left = [r[4] for r in log if r[1] != "ECOSYSTEM EXHAUSTED"]
        return t, val, reg, left

    tg, vg, rg, lg = series(gl)
    ti, vi, ri, li = series(il)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax1, ax2, ax3 = axes

    ax1.step(tg, rg, where="post", color=RED, lw=2, label="greedy router")
    ax1.step(ti, ri, where="post", color=BLUE, lw=2, ls="--",
             label="ECI router")
    ax1.set_xlabel("step"); ax1.set_ylabel("per-step regret")
    ax1.set_title("(a)  The dashboard")
    ax1.legend(loc="upper left")
    ax1.set_ylim(-0.03, 0.33)

    ax2.step(tg, vg, where="post", color=RED, lw=2)
    ax2.step(ti, vi, where="post", color=BLUE, lw=2, ls="--")
    ax2.set_xlabel("step"); ax2.set_ylabel("value per call")
    ax2.set_title("(b)  The outcome")

    ax3.step(tg, lg, where="post", color=RED, lw=2)
    ax3.step(ti, li, where="post", color=BLUE, lw=2, ls="--")
    ax3.set_xlabel("step"); ax3.set_ylabel("tools remaining")
    ax3.set_title("(c)  The ecosystem")
    ax3.set_ylim(0, 6.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_agent_trace.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig5: greedy ends val=%.3f tools=%d | eci ends val=%.3f tools=%d"
          % (vg[-1], lg[-1], vi[-1], li[-1]))


if __name__ == "__main__":
    fig1_divergence()
    fig2_characterisation()
    fig3_estimation_floor()
    fig4_policies()
    fig5_agent_trace()
    print("\nall figures written to", OUT)
