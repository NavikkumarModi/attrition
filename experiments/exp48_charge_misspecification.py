"""Experiment 48 -- free-riding is charge misspecification, and that generalises.

Theorem 6 collapsed the multi-agent setting into a single-learner one, which makes
free-riding much easier to analyse than it first appeared. An agent that prices the
externality but discounts it by its own expected share is charging

    lambda * delta * kappa_a * (T - t),      lambda = 1/m

so free-riding is exactly ECI with a mis-scaled charge. That reframing turns a
multi-agent question into a single-agent one and answers a broader question at the
same time: how robust is ECI to getting the externality scale wrong?

    lambda = 0    greedy: ignores the externality entirely
    lambda = 1/m  free-riding with m agents
    lambda = 1    correct social pricing
    lambda > 1    over-charging: excessive caution

The practically important question is the shape of the curve near lambda = 1. If it
is flat, ECI tolerates a badly estimated scale and the specification burden of
Theorem 3 is light. If it is sharp, the scale must be known well, and Theorem 3
bites harder than the ordinal results suggested.
"""

from functools import lru_cache

import numpy as np

__all__ = ["value_at_lambda", "sweep_lambda"]


def value_at_lambda(v, p, e, delta, T, lam):
    """Exact value of the policy that charges lam * delta * kappa_a * (T-t)."""
    n = len(v)
    full = frozenset(range(n))
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def W(S, t):
        if t >= T or not S:
            return 0.0
        a = max(S, key=lambda i: R(i, S) - lam * delta * p[i] * e[i] * (T - t))
        return R(a, S) + p[a]*W(S-{a}, t+1) + (1-p[a])*W(S, t+1)

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)

    return W(full, 0), V(full, 0)


def sweep_lambda(lams, n=6, T=8, delta=0.15, spread=1.2, inst=10, seed=23):
    rng = np.random.default_rng(seed)
    problems = []
    for _ in range(inst):
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, spread, n), 0.0, None)
        problems.append((v, p, e))
    rows = []
    for lam in lams:
        losses = []
        for v, p, e in problems:
            w, vs = value_at_lambda(v, p, e, delta, T, lam)
            losses.append(vs - w)          # raw loss, not a percentage
        rows.append((lam, float(np.mean(losses))))
    return rows


if __name__ == "__main__":
    print("FREE-RIDING AS CHARGE MISSPECIFICATION\n")
    print("An agent discounting the externality by its own share charges")
    print("lambda = 1/m of the correct amount. The same sweep answers how")
    print("robust ECI is to any misspecified scale.\n")

    lams = [0.0, 0.125, 0.25, 0.333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    rows = sweep_lambda(lams)
    best = min(r[1] for r in rows)
    print(f"{'lambda':>8} {'raw loss':>10} {'excess over best':>17} {'reading':>22}")
    print("-" * 62)
    label = {0.0: "greedy", 0.125: "8 agents", 0.25: "4 agents",
             0.333: "3 agents", 0.5: "2 agents", 1.0: "correct pricing"}
    for lam, loss in rows:
        print(f"{lam:8.3f} {loss:10.4f} {loss-best:17.4f} "
              f"{label.get(lam, ''):>22}")

    print("\n\nSHAPE NEAR THE OPTIMUM -- does the scale need to be known well?\n")
    fine = sweep_lambda([0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.6, 2.2])
    b = min(r[1] for r in fine)
    print(f"{'lambda':>8} {'raw loss':>10} {'excess':>10}")
    print("-" * 30)
    for lam, loss in fine:
        print(f"{lam:8.2f} {loss:10.4f} {loss-b:10.4f}")

    print("\n\nUNDER- VERSUS OVER-CHARGING: which error is safer?\n")
    print(f"{'factor off':>12} {'under (1/f)':>13} {'over (f)':>11} "
             f"{'which is worse':>16}")
    print("-" * 56)
    for f in [1.5, 2.0, 3.0, 5.0]:
        under = sweep_lambda([1.0 / f])[0][1]
        over = sweep_lambda([f])[0][1]
        worse = "under" if under > over else "over"
        print(f"{f:12.1f} {under:13.4f} {over:11.4f} {worse:>16}")
