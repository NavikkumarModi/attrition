"""Experiment 13 -- isolate the option-loss term for the Theorem 1 proof.

Rewrite the Bellman comparison. With B(S) and V(S,t+1) independent of the
chosen arm, maximising

    v_a - B(S) + p_a V(S\{a},t+1) + (1-p_a) V(S,t+1)

is equivalent to maximising

    v_a - p_a * D_a(S,t),      D_a := V(S,t+1) - V(S\{a},t+1)

D_a is the marginal value of still holding arm a. Decompose it:

    D_a  =  delta * e_a * L(S,t)   +   O_a(S,t)
            \_____ burden _____/       \_ option _/

The burden part contributes p_a * delta * e_a * L = delta * kappa_a * L, which
is CONSTANT across arms exactly when kappa is constant -- that is the easy half
of Theorem 1. The proof therefore hinges entirely on the option term:

    LEMMA 1a:  argmax_a [ v_a - p_a * O_a(S,t) ]  =  argmax_a v_a

i.e. the option correction never reorders the arms. This measures O_a directly
in the base model (e=0) and checks the lemma exhaustively.
"""
from functools import lru_cache
from itertools import combinations
import numpy as np

def base_V(v, p, T):
    n = len(v)
    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S: return 0.0
        return max(v[a] + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)
    return V

rng = np.random.default_rng(41)
viol = 0; tot = 0
max_ratio_spread = 0.0
rows = []

for inst in range(200):
    n = int(rng.integers(3, 7)); T = int(rng.integers(2, 9))
    v = np.sort(rng.uniform(0.1, 1.5, n))[::-1].copy()
    p = np.clip(rng.uniform(0.05, 1.0, n), 0.02, 1.0)
    V = base_V(v, p, T)
    # check the lemma at every reachable (S,t)
    for size in range(2, n+1):
        for Ss in combinations(range(n), size):
            S = frozenset(Ss)
            for t in range(T):
                if t+1 > T: continue
                O = {a: V(S, t+1) - V(S - {a}, t+1) for a in S}
                corrected = {a: v[a] - p[a]*O[a] for a in S}
                best_plain = max(S, key=lambda a: v[a])
                best_corr = max(S, key=lambda a: corrected[a])
                tot += 1
                if abs(corrected[best_corr] - corrected[best_plain]) > 1e-9:
                    viol += 1
                    if len(rows) < 5:
                        rows.append((list(Ss), t, v[list(Ss)].round(3),
                                     p[list(Ss)].round(3)))
                po = np.array([p[a]*O[a] for a in sorted(S)])
                if po.max() > 1e-12:
                    max_ratio_spread = max(max_ratio_spread,
                                           float(po.max()-po.min()))

print(f"LEMMA 1a checked at {tot} reachable (S,t) states across 200 instances")
print(f"  violations: {viol}")
print(f"  max spread of p_a*O_a across arms in a state: {max_ratio_spread:.4f}")
print(f"  -> p_a*O_a is NOT constant, yet never reorders: "
      f"{'CONFIRMED' if viol==0 else 'FALSIFIED'}")
if rows:
    print("\n counterexamples:")
    for r in rows: print("  ", r)
