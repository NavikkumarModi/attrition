r"""Experiment 18 -- verify the exchange argument for Lemma 1a.

The proposed proof: take an optimal policy that pulls b at (S,t) while a with
v_a > v_b is available. Couple the death randomness, swap the two pulls, show
the swap is weakly improving.

The claim rests on an *interchange* property. Rather than assume it, verify the
exact quantity the argument needs:

    Q(a,S,t) := v_a + p_a V(S\{a},t+1) + (1-p_a) V(S,t+1)

Lemma 1a says argmax Q = argmax v. Equivalently, for any a,b in S with
v_a >= v_b:

    Q(a,S,t) - Q(b,S,t) >= 0                                        (EX)

Expanding, with D_x := V(S,t+1) - V(S\{x},t+1):

    (EX)  <=>  (v_a - v_b)  >=  p_a D_a - p_b D_b

So the exchange argument needs: the difference in expected option loss between
two arms never exceeds their value gap. That is a concrete, checkable inequality
and it is the precise content of the lemma.

This measures the SLACK  (v_a - v_b) - (p_a D_a - p_b D_b)  over all pairs and
states, and identifies exactly when it binds (equals zero).
"""

from functools import lru_cache
from itertools import combinations
import numpy as np


def build_V(v, p, T):
    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        return max(v[a] + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)
    return V


def audit(n, T, rng, tol=1e-9):
    v = np.sort(rng.uniform(0.1, 1.5, n))[::-1].copy()
    p = np.clip(rng.uniform(0.05, 1.0, n), 0.02, 1.0)
    V = build_V(v, p, T)
    viol, slacks, binding = 0, [], 0
    for size in range(2, n + 1):
        for Ss in combinations(range(n), size):
            S = frozenset(Ss)
            for t in range(T):
                D = {x: V(S, t+1) - V(S - {x}, t+1) for x in S}
                for a in S:
                    for b in S:
                        if v[a] < v[b] - 1e-12:
                            continue
                        slack = (v[a] - v[b]) - (p[a]*D[a] - p[b]*D[b])
                        slacks.append(slack)
                        if slack < -tol:
                            viol += 1
                        if abs(slack) < 1e-9:
                            binding += 1
    return viol, np.array(slacks), binding


if __name__ == "__main__":
    rng = np.random.default_rng(19)
    tot_v, tot_c, tot_b = 0, 0, 0
    all_min = np.inf
    for inst in range(60):
        n = int(rng.integers(3, 7)); T = int(rng.integers(2, 8))
        viol, sl, binding = audit(n, T, rng)
        tot_v += viol; tot_c += len(sl); tot_b += binding
        all_min = min(all_min, float(sl.min()))
    print("EXCHANGE INEQUALITY (EX):  (v_a - v_b) >= p_a D_a - p_b D_b")
    print(f"  pairs checked      : {tot_c}")
    print(f"  violations         : {tot_v}")
    print(f"  minimum slack      : {all_min:+.10f}")
    print(f"  binding (slack=0)  : {tot_b}  ({100*tot_b/tot_c:.1f}%)")
    print()
    print("  -> (EX) holds:", "CONFIRMED" if tot_v == 0 else "FALSIFIED")

    # where does it bind? test the p=1 fully-consumed regime specifically
    print("\nBinding structure: p=1 (every pull consumes) vs p<1")
    for label, lo, hi in [("p=1 exactly", 1.0, 1.0), ("p in [0.05,1)", 0.05, 0.99)][:2] \
            if False else [("p=1 exactly", 1.0, 1.0), ("p in [0.05,0.99]", 0.05, 0.99)]:
        rng2 = np.random.default_rng(3)
        binds, tot = 0, 0
        for _ in range(25):
            n = int(rng2.integers(3, 6)); T = int(rng2.integers(2, 7))
            v = np.sort(rng2.uniform(0.1, 1.5, n))[::-1].copy()
            p = np.full(n, 1.0) if lo == hi else np.clip(
                rng2.uniform(lo, hi, n), 0.02, 1.0)
            V = build_V(v, p, T)
            for size in range(2, n+1):
                for Ss in combinations(range(n), size):
                    S = frozenset(Ss)
                    for t in range(T):
                        D = {x: V(S, t+1) - V(S-{x}, t+1) for x in S}
                        for a in S:
                            for b in S:
                                if v[a] < v[b] - 1e-12: continue
                                s = (v[a]-v[b]) - (p[a]*D[a] - p[b]*D[b])
                                tot += 1
                                if abs(s) < 1e-9: binds += 1
        print(f"  {label:>18}: {100*binds/tot:5.1f}% of pairs bind")
