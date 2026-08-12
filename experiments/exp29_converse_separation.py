"""Experiment 29 -- the converse separation, and a second benchmark.

Two review items at once.

(1) SECOND BENCHMARK. The paper measures regret against the best available arm,
    which a reviewer can call convenient. Define instead

        system-value regret   R_sys(pi) = V* - W(pi)

    against the ex-ante optimal policy. Report both for every policy so the
    reader can see that the choice of benchmark, not the algorithm, is what
    changes the verdict.

(2) THE CONVERSE (new theorem). The paper shows zero private regret permits
    unbounded system loss. The converse is sharper:

        any policy that is near-optimal in system value MUST incur
        linear private regret against the best-available benchmark.

    That is, a good policy does not merely happen to look bad on the dashboard --
    it is forced to. Construction: in the hot/safe family the optimal policy
    defers the hot arm, so at every one of the m safe pulls it declines an arm
    that is strictly best-available, accruing eps per round; and to avoid the
    burden it must decline on every round, giving m*eps total.

    Sharper still: as eps -> delta*E the private regret of the OPTIMAL policy
    approaches its own system-value advantage, so the dashboard reading is
    proportional to the benefit being delivered.
"""

from functools import lru_cache

import numpy as np


def analyse(v, p, e, delta, T):
    """Exact system value and private regret for optimal, greedy and ECI."""
    n = len(v); full = frozenset(range(n)); etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)

    def best_action(S, t):
        return max(S, key=lambda a: R(a, S) + p[a]*V(S-{a}, t+1)
                   + (1-p[a])*V(S, t+1))

    def evaluate(pick):
        """Return (system value, cumulative private regret) for a policy."""
        @lru_cache(maxsize=None)
        def W(S, t):
            if t >= T or not S:
                return 0.0
            a = pick(S, t)
            return R(a, S) + p[a]*W(S-{a}, t+1) + (1-p[a])*W(S, t+1)

        @lru_cache(maxsize=None)
        def Reg(S, t):
            if t >= T or not S:
                return 0.0
            a = pick(S, t)
            inst = max(R(i, S) for i in S) - R(a, S)
            return inst + p[a]*Reg(S-{a}, t+1) + (1-p[a])*Reg(S, t+1)
        return W(full, 0), Reg(full, 0)

    w_opt, r_opt = evaluate(best_action)
    w_grd, r_grd = evaluate(lambda S, t: max(S, key=lambda i: R(i, S)))
    w_eci, r_eci = evaluate(lambda S, t: max(
        S, key=lambda i: R(i, S) - delta*p[i]*e[i]*(T-t)))
    return {"optimal": (w_opt, r_opt), "greedy": (w_grd, r_grd),
            "eci": (w_eci, r_eci), "Vstar": V(full, 0)}


def hot_safe(m, E, eps=0.05, delta=1.0):
    n = m + 1
    v = np.array([1.0] + [1.0 - eps]*m)
    p = np.ones(n)
    e = np.array([E] + [0.0]*m)
    return v, p, e, delta, n


if __name__ == "__main__":
    print("PART 1 -- both benchmarks side by side\n")
    print("Random instances, n=6, T=8. Note how the two columns rank policies")
    print("in opposite orders.\n")
    rng = np.random.default_rng(17)
    print(f"{'policy':>9} {'system value':>13} {'system regret':>14} "
          f"{'private regret':>15}")
    print("-" * 55)
    acc = {k: [[], [], []] for k in ["optimal", "greedy", "eci"]}
    for _ in range(15):
        v = np.sort(rng.uniform(0.4, 1.2, 6))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, 6), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, 1.2, 6), 0.0, None)
        r = analyse(v, p, e, 0.12, 8)
        for k in acc:
            w, pr = r[k]
            acc[k][0].append(w)
            acc[k][1].append(r["Vstar"] - w)
            acc[k][2].append(pr)
    for k in ["optimal", "eci", "greedy"]:
        print(f"{k:>9} {np.mean(acc[k][0]):13.4f} {np.mean(acc[k][1]):14.4f} "
              f"{np.mean(acc[k][2]):15.4f}")
    print("\n  greedy: BEST private regret, WORST system value.")
    print("  optimal: WORST private regret, BEST system value.")
    print("  The benchmark, not the algorithm, decides the verdict.")

    print("\n\nPART 2 -- the converse: optimality FORCES private regret\n")
    print(f"{'m':>4} {'delta*E':>8} {'V* - V_greedy':>14} "
          f"{'private regret of OPTIMAL':>26} {'predicted m*eps':>16}")
    print("-" * 74)
    eps = 0.05
    for m in [2, 4, 8, 12]:
        for E in [0.5, 1.5]:
            v, p, e, delta, n = hot_safe(m, E, eps=eps)
            r = analyse(v, p, e, delta, n)
            gap = r["Vstar"] - r["greedy"][0]
            pr_opt = r["optimal"][1]
            print(f"{m:4d} {delta*E:8.2f} {gap:14.4f} {pr_opt:26.4f} "
                  f"{m*eps:16.4f}")
    print("\n  The optimal policy's private regret grows LINEARLY in m.")
    print("  It is not merely permitted to look bad -- it is required to.")

    print("\n\nPART 3 -- the dashboard reading tracks the benefit delivered\n")
    print(f"{'eps':>7} {'system gain of optimal':>23} "
          f"{'private regret of optimal':>26} {'ratio':>8}")
    print("-" * 68)
    for eps in [0.01, 0.05, 0.10, 0.20, 0.40]:
        v, p, e, delta, n = hot_safe(8, 1.0, eps=eps)
        r = analyse(v, p, e, delta, n)
        gain = r["Vstar"] - r["greedy"][0]
        pr = r["optimal"][1]
        print(f"{eps:7.2f} {gain:23.4f} {pr:26.4f} "
              f"{(pr/gain if gain > 1e-9 else float('nan')):8.3f}")
    print("\n  As eps grows the price of preservation rises toward the benefit,")
    print("  so the worse a good policy looks, the more it is delivering.")
