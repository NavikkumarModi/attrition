"""Experiment 17 -- zero-regret ruin in an agent tool ecosystem.

The abstract model, instantiated where the AI community will recognise it.

Setting. An orchestrator routes tasks across API-backed tools. Each tool has a
quality v_a and a fragility p_a: heavy use trips a rate limit, exhausts a quota,
or gets the key revoked, permanently removing the tool for the session.

The externality. Tools are not independent. They share upstream infrastructure
-- a common vendor, a shared IP pool, a shared quota tier. When a tool is burned,
downstream capacity degrades for every remaining tool by delta*e_a, where e_a is
how much shared infrastructure tool a sits on. A premium tool on a dedicated
endpoint has low e_a; a cheap tool sharing a public quota pool has high e_a.

This is not a contrived mapping. Shared rate limits across a provider's API
surface, noisy-neighbour effects on shared tenancy, and reputation-based
throttling of an IP range are all standard and all have exactly this structure.

What the experiment shows. A router optimising observed task success -- the
metric an ops dashboard actually displays -- achieves ZERO regret against the
best available tool at every step, while burning the ecosystem underneath itself.
The dashboard stays green throughout.
"""

from functools import lru_cache
import numpy as np


TOOLS = [
    # name,                  quality, fragility, shared-infra coefficient
    ("premium-search",          1.15,      0.15,   0.2),
    ("bulk-search",             1.05,      0.75,   2.4),
    ("premium-codegen",         1.00,      0.10,   0.1),
    ("bulk-scraper",            0.98,      0.85,   2.8),
    ("internal-db",             0.80,      0.05,   0.0),
    ("cached-lookup",           0.62,      0.02,   0.0),
]


def build(delta):
    v = np.array([t[1] for t in TOOLS])
    p = np.array([t[2] for t in TOOLS])
    e = np.array([t[3] for t in TOOLS])
    return v, p, e, delta


def analyse(v, p, e, delta, T):
    n = len(v); full = frozenset(range(n)); etot = float(e.sum())
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def Vstar(S, t):
        if t >= T or not S: return 0.0
        return max(R(a,S) + p[a]*Vstar(S-{a},t+1) + (1-p[a])*Vstar(S,t+1)
                   for a in S)

    def value_of(pick):
        @lru_cache(maxsize=None)
        def W(S, t):
            if t >= T or not S: return 0.0
            a = pick(S, t)
            return R(a,S) + p[a]*W(S-{a},t+1) + (1-p[a])*W(S,t+1)
        return W(full, 0)

    greedy = value_of(lambda S, t: max(S, key=lambda i: R(i, S)))
    index = value_of(lambda S, t: max(
        S, key=lambda i: R(i, S) - delta*p[i]*e[i]*(T-t)))
    return Vstar(full, 0), greedy, index


def trace(v, p, e, delta, T, policy, seed=0):
    """Simulate one session, logging what a dashboard would display."""
    rng = np.random.default_rng(seed)
    n = len(v); alive = np.ones(n, dtype=bool)
    etot = float(e.sum())
    log = []
    for t in range(T):
        if not alive.any():
            log.append((t, "ECOSYSTEM EXHAUSTED", 0.0, 0.0, 0))
            break
        idx = np.flatnonzero(alive)
        burden = delta * float(e[~alive].sum())
        if policy == "greedy":
            a = int(idx[np.argmax(v[idx])])
        else:
            sc = v[idx] - burden - delta*p[idx]*e[idx]*(T-t)
            a = int(idx[np.argmax(sc)])
        realised = v[a] - burden
        best_available = float(np.max(v[idx] - burden))
        log.append((t, TOOLS[a][0], realised, best_available - realised,
                    int(alive.sum())))
        if rng.random() < p[a]:
            alive[a] = False
    return log


if __name__ == "__main__":
    T = 30
    print("AGENT TOOL ECOSYSTEM -- zero-regret ruin\n")
    print(f"{'coupling':>9} {'V*':>8} {'greedy':>8} {'index':>8} "
          f"{'greedy loss':>12} {'greedy regret':>14}")
    print("-" * 64)
    for delta in [0.0, 0.01, 0.03, 0.06, 0.10]:
        v, p, e, d = build(delta)
        vs, vg, vi = analyse(v, p, e, d, T)
        print(f"{delta:9.2f} {vs:8.2f} {vg:8.2f} {vi:8.2f} "
              f"{vs-vg:11.2f} {0.0:13.4f}")

    print("\n\nSESSION TRACE at coupling=0.06, greedy router")
    print("(regret column is what an ops dashboard shows)\n")
    v, p, e, d = build(0.06)
    print(f"{'step':>4} {'tool called':>18} {'realised':>9} {'regret':>8} "
          f"{'tools left':>11}")
    print("-" * 56)
    for row in trace(v, p, e, d, T, "greedy", seed=3)[:14]:
        t, name, real, reg, left = row
        print(f"{t:4d} {name:>18} {real:9.3f} {reg:8.4f} {left:11d}")

    print("\n\nSame session under the index router\n")
    print(f"{'step':>4} {'tool called':>18} {'realised':>9} {'regret':>8} "
          f"{'tools left':>11}")
    print("-" * 56)
    for row in trace(v, p, e, d, T, "index", seed=3)[:14]:
        t, name, real, reg, left = row
        print(f"{t:4d} {name:>18} {real:9.3f} {reg:8.4f} {left:11d}")
