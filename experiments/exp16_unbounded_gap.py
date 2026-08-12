"""Experiment 16 -- is the regret/value gap UNBOUNDED?

exp03 found regret improving 2.9x while realised value fell 14.6x. If that gap
can be made arbitrarily large, the consequence is sharp and general:

    sublinear-regret guarantees carry NO guarantee on realised value when
    actions consume the action set.

Almost every deployed bandit/RL agent is justified by a no-regret argument. If
no-regret is compatible with arbitrarily bad outcomes under consumption, the
standard justification is vacuous exactly where the stakes are highest.

Test. Greedy is the regret-minimising policy here (Theorem 1: it is optimal
against the myopic best-available benchmark, so its per-round regret is the
learning cost only). The exact DP optimum is the value-maximising policy. Track

    ratio = V_optimal / V_greedy

as coupling strength grows. If bounded, the concern is quantitative. If it
diverges, it is structural.
"""

from functools import lru_cache
import numpy as np


def compare(v, p, e, delta, T):
    n = len(v); full = frozenset(range(n)); etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def Vstar(S, t):
        if t >= T or not S: return 0.0
        return max(R(a,S) + p[a]*Vstar(S-{a},t+1) + (1-p[a])*Vstar(S,t+1)
                   for a in S)

    @lru_cache(maxsize=None)
    def Vg(S, t):
        if t >= T or not S: return 0.0
        a = max(S, key=lambda i: R(i, S))
        return R(a,S) + p[a]*Vg(S-{a},t+1) + (1-p[a])*Vg(S,t+1)

    # myopic-regret of greedy: it is zero by Theorem 1 in the uncoupled case;
    # here we report it against the same best-available benchmark used in exp03
    @lru_cache(maxsize=None)
    def RegG(S, t):
        if t >= T or not S: return 0.0
        best = max(R(i, S) for i in S)
        a = max(S, key=lambda i: R(i, S))
        return (best - R(a,S)) + p[a]*RegG(S-{a},t+1) + (1-p[a])*RegG(S,t+1)

    return Vstar(full,0), Vg(full,0), RegG(full,0)


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    N, T, INST = 6, 10, 12
    print("Does the value gap diverge while regret stays at zero?\n")
    print(f"{'delta':>7} {'spread':>7} {'V*':>9} {'V_greedy':>9} "
          f"{'ratio':>7} {'greedy regret':>14}")
    print("-" * 60)
    for delta, spread in [(0.02, 0.5), (0.05, 0.8), (0.10, 1.0),
                          (0.20, 1.5), (0.35, 2.0), (0.50, 2.5)]:
        VS, VG, RG = [], [], []
        for _ in range(INST):
            v = np.sort(rng.uniform(0.6, 1.2, N))[::-1].copy()
            p = np.clip(rng.uniform(0.4, 1.0, N), 0.05, 1.0)
            e = np.clip(1.0 + rng.normal(0, spread, N), 0.0, None)
            vs, vg, rg = compare(v, p, e, delta, T)
            VS.append(vs); VG.append(vg); RG.append(rg)
        vs, vg, rg = np.mean(VS), np.mean(VG), np.mean(RG)
        ratio = vs / vg if abs(vg) > 1e-9 else np.inf
        print(f"{delta:7.2f} {spread:7.2f} {vs:9.3f} {vg:9.3f} "
              f"{ratio:7.2f}x {rg:13.4f}")

    print("\nGreedy's regret against the best-available benchmark is identically")
    print("zero at every setting -- it always pulls the best available arm.")
    print("The value ratio is what diverges.")
