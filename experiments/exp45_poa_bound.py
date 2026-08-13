"""Experiment 45 -- toward a bound on the price of anarchy under consumption.

exp32 measured PoA up to 1.297 but gave no functional form. To state a bound we
need to know what it scales with.

CANDIDATE FORM. Each of m agents ignores the externality it imposes on the other
m-1, so the marginal damage it under-prices is a fraction (m-1)/m of the total.
That suggests

    PoA - 1  ~  C * delta * std(kappa) * (m-1)/m * T

with the (m-1)/m factor saturating at 1 for large m -- decentralisation cannot get
worse than fully ignoring everyone else.

Theorem 1 pins one end exactly: at zero kappa dispersion greedy is optimal, so
every agent playing greedy is playing optimally and PoA = 1 regardless of m.

Sweeps: agent count, dispersion, coupling scale, horizon.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np

__all__ = ["poa", "sweep_poa"]


def planner(v, p, e, delta, T, m):
    """Exact optimum for a controller making m pulls per round."""
    n = len(v)
    etot = float(np.sum(e))
    burden = lambda S: delta * (etot - sum(e[i] for i in S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        k = min(m, len(S))
        best = -np.inf
        for combo in combinations(sorted(S), k):
            b = burden(S)
            imm = sum(v[a] - b for a in combo)
            cont = 0.0
            for mask in range(1 << k):
                pr, dead = 1.0, []
                for j, a in enumerate(combo):
                    if mask >> j & 1:
                        pr *= p[a]; dead.append(a)
                    else:
                        pr *= (1 - p[a])
                if pr > 0:
                    cont += pr * V(S - frozenset(dead), t + 1)
            best = max(best, imm + cont)
        return best

    return V(frozenset(range(n)), 0)


def decentralised(v, p, e, delta, T, m, seeds=250):
    """m independent greedy agents, round-robin within each round."""
    n = len(v)
    totals = []
    for s in range(seeds):
        rng = np.random.default_rng(9000 + s)
        alive = np.ones(n, dtype=bool)
        tot = 0.0
        for _ in range(T):
            if not alive.any():
                break
            for _ in range(m):
                idx = np.flatnonzero(alive)
                if idx.size == 0:
                    break
                b = delta * float(e[~alive].sum())
                a = int(idx[np.argmax(v[idx])])
                tot += v[a] - b
                if rng.random() < p[a]:
                    alive[a] = False
        totals.append(tot)
    return float(np.mean(totals))


def poa(v, p, e, delta, T, m):
    pl = planner(v, p, e, delta, T, m)
    de = decentralised(v, p, e, delta, T, m)
    return pl / max(de, 1e-9), pl, de


def sweep_poa(param, values, inst=6, seed=5, **fixed):
    rows = []
    for val in values:
        kw = dict(n=6, T=5, delta=0.12, spread=1.0, m=2)
        kw.update(fixed)
        kw[param] = val
        rng = np.random.default_rng(seed)
        rs, sds = [], []
        for _ in range(inst):
            n = kw["n"]
            v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng.uniform(0.3, 0.9, n), 0.05, 1.0)
            if kw["spread"] == 0.0:
                e = 0.4 / p                       # kappa exactly constant
            else:
                e = np.clip(1.0 + rng.normal(0, kw["spread"], n), 0.0, None)
            r, _, _ = poa(v, p, e, kw["delta"], kw["T"], kw["m"])
            rs.append(r); sds.append(float(np.std(p * e)))
        rows.append((val, float(np.mean(sds)), float(np.mean(rs))))
    return rows


def show(title, label, rows):
    print(f"\n{title}\n")
    print(f"{label:>9} {'std(kappa)':>11} {'PoA':>8} {'PoA - 1':>10}")
    print("-" * 42)
    for val, sd, r in rows:
        print(f"{val:9.3f} {sd:11.3f} {r:8.4f} {r-1:10.4f}")


if __name__ == "__main__":
    print("PRICE OF ANARCHY: WHAT DOES IT SCALE WITH?")

    show("Zero dispersion must give PoA = 1 exactly (Theorem 1)",
         "spread", sweep_poa("spread", [0.0]))

    show("Dispersion of kappa", "spread",
         sweep_poa("spread", [0.0, 0.3, 0.7, 1.2, 1.8]))

    show("Agent count m", "m", sweep_poa("m", [1, 2, 3]))

    show("Coupling scale delta", "delta",
         sweep_poa("delta", [0.03, 0.08, 0.15, 0.30]))

    show("Horizon T", "T", sweep_poa("T", [3, 4, 5, 6]))

    print("\n  Theorem 1 fixes PoA = 1 at zero dispersion for any m: if greedy is")
    print("  optimal then every agent playing greedy is playing optimally, and")
    print("  decentralisation costs nothing. Any bound must vanish there.")
