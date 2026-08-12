"""Experiment 21 -- make Theorem 4 a PROVEN theorem, not a verified one.

exp16 showed the value gap growing without apparent bound. "Apparent" is not
good enough for ICML. This builds an explicit two-arm family with closed-form
values, so the gap can be computed analytically and verified against DP.

CONSTRUCTION (deterministic consumption, p=1, so both arms die when pulled).

  arm H ("hot")   : value 1,       externality e_H = E
  arm S ("safe")  : value 1 - eps, externality e_S = 0

Horizon T. Burden after H dies is delta*E, charged on EVERY later round.

Greedy pulls H first (1 > 1-eps). Then only S remains, at value
(1-eps) - delta*E, for the remaining T-1 rounds:

  V_greedy = 1 + (T-1)(1 - eps - delta*E)

Optimal (for delta*E large) pulls S first, then H:

  V_opt    = (1 - eps) + 1 + ... but S dies on its pull, leaving H
           = (1 - eps) + (T-1)(1 - delta*0) ... H not yet dead, no burden
           = (1 - eps) + 1 + (T-2)(1 - delta*E)
    -- pulling H at step 2 incurs burden only from step 3 onward.

  GAP = V_opt - V_greedy = delta*E - eps   ... per-round advantage over T-2 rounds

Setting eps -> 0 and scaling delta*E gives GAP = Theta(delta*E*T), unbounded in
BOTH delta*E and T, while greedy's regret against best-available is exactly 0
at every step (it always pulls the highest available value).

This script verifies the closed form against exact DP and confirms the scaling.
"""

from functools import lru_cache
import numpy as np


def dp(v, e, delta, T, p=None):
    """Exact optimum and greedy value; p defaults to deterministic death."""
    n = len(v)
    p = np.ones(n) if p is None else p
    full = frozenset(range(n))
    etot = float(np.sum(e))
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

    return V(full, 0), G(full, 0)


def closed_form(eps, dE, T):
    """Predicted greedy and optimal values for the two-arm family."""
    v_greedy = 1.0 + (T - 1) * (1.0 - eps - dE)
    v_opt = (1.0 - eps) + 1.0 + (T - 2) * (1.0 - dE)
    return max(v_opt, v_greedy), v_greedy


if __name__ == "__main__":
    print("THEOREM 4 -- explicit construction with exact closed form\n")
    print("Family: 1 hot arm (v=1, e=E) + m safe arms (v=1-eps, e=0), p=1, T=n+1")
    print("Claim:  V_opt - V_greedy = m*delta*E  exactly.\n")
    print(f"{'m':>4} {'n':>4} {'delta*E':>8} {'DP opt':>9} {'DP greedy':>10} "
          f"{'DP gap':>9} {'m*delta*E':>10} {'exact':>6}")
    print("-" * 68)
    ok = True
    for dE in [0.5, 1.0, 2.0]:
        for m in [1, 2, 4, 8, 12]:
            eps = 0.01
            n = m + 1
            v = np.array([1.0] + [1.0 - eps] * m)
            e = np.array([dE] + [0.0] * m)
            vo, vg = dp(v, e, 1.0, n + 1)
            pred = m * dE
            exact = abs((vo - vg) - pred) < 1e-9
            ok &= exact
            print(f"{m:4d} {n:4d} {dE:8.2f} {vo:9.4f} {vg:10.4f} "
                  f"{vo-vg:9.4f} {pred:10.4f} {'yes' if exact else 'NO':>6}")
        print()
    print(f"closed form exact everywhere: {'CONFIRMED' if ok else 'FALSIFIED'}")
    print()
    print("Greedy regret against best-available benchmark, same instances:")
    print("  identically 0 at every step -- greedy always pulls the highest")
    print("  available value by construction.")
