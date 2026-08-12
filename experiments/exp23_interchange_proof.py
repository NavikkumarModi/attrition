"""Experiment 23 -- the last gap: Lemma 1a's coupling argument.

(EX) was verified over 44,860 pairs but not proven. The proposed route is an
interchange argument. This tests the exact property that argument requires, so
the proof can be written rather than gestured at.

CLAIM (Interchange). Fix a state S and arms a, b in S with v_a >= v_b. Let pi
be any policy that pulls b at time t and a at time t+1. Let pi' be pi with those
two pulls swapped. Then V(pi') >= V(pi), under a coupling of the death
randomness.

Why this should hold. Couple the death coins: use the same uniform draw for
whichever arm is pulled at each step. Then after both pulls, the SET of surviving
arms has the same distribution under pi and pi' -- the pair {a,b} experiences the
same two coins in either order. The only difference is WHICH reward arrives
first, and since rewards are stationary and the burden state is identical at
time t, the difference is exactly (v_a - v_b) >= 0 discounted by nothing.

The subtlety the proof must handle: the burden created by a death at step t is
charged from step t+1 onward, so the ORDER of deaths matters for the burden path.
This tests whether that effect can reverse the inequality.

Test: enumerate all (S, a, b) and compare the two-step swap directly.
"""

from functools import lru_cache
from itertools import combinations
import numpy as np


def build(v, p, e, delta, T):
    n = len(v)
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)

    def two_step(S, t, first, second):
        """Value of pulling `first` then `second` (if still alive), then optimal."""
        total = R(first, S)
        out = 0.0
        for died1, w1 in [(True, p[first]), (False, 1-p[first])]:
            if w1 == 0:
                continue
            S1 = S - {first} if died1 else S
            if second not in S1:
                out += w1 * V(S1, t+1)
                continue
            sub = R(second, S1)
            for died2, w2 in [(True, p[second]), (False, 1-p[second])]:
                if w2 == 0:
                    continue
                S2 = S1 - {second} if died2 else S1
                sub += w2 * V(S2, t+2)
            out += w1 * sub
        return total + out

    return V, two_step, R


if __name__ == "__main__":
    rng = np.random.default_rng(131)
    viol = 0
    checked = 0
    min_slack = np.inf
    worst = None

    for inst in range(80):
        n = int(rng.integers(3, 6))
        T = int(rng.integers(3, 8))
        v = np.sort(rng.uniform(0.2, 1.4, n))[::-1].copy()
        p = np.clip(rng.uniform(0.1, 1.0, n), 0.05, 1.0)
        # include the coupled case -- this is where order of deaths matters
        e = np.clip(rng.uniform(0.0, 2.5, n), 0.0, None)
        delta = float(rng.choice([0.0, 0.05, 0.15]))
        V, two_step, R = build(v, p, e, delta, T)

        for size in range(2, n+1):
            for Ss in combinations(range(n), size):
                S = frozenset(Ss)
                for t in range(max(T-1, 1)):
                    for a in S:
                        for b in S:
                            if a == b:
                                continue
                            if R(a, S) < R(b, S) - 1e-12:
                                continue          # want the higher-value arm as `a`
                            ab = two_step(S, t, a, b)
                            ba = two_step(S, t, b, a)
                            checked += 1
                            slack = ab - ba
                            if slack < min_slack:
                                min_slack = slack
                                worst = (delta, float(np.std(p*e)), a, b)
                            if slack < -1e-9:
                                viol += 1

    print("INTERCHANGE CLAIM: pulling the higher-value arm FIRST is weakly better\n")
    print(f"  ordered pairs checked : {checked}")
    print(f"  violations            : {viol}")
    print(f"  minimum slack         : {min_slack:+.10f}")
    print()
    if viol == 0:
        print("  -> interchange holds: CONFIRMED")
        print("     The two-step swap never hurts, including when delta > 0,")
        print("     so death-order effects on the burden path do not reverse it.")
    else:
        print("  -> interchange FAILS. worst case (delta, std(kappa), a, b):", worst)
        print("     Lemma 1a cannot be proven by simple interchange; the")
        print("     coupling must account for burden-path asymmetry.")
