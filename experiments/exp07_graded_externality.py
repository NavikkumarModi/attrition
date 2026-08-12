"""Experiment 07 -- graded externalities, and a decisive test of C1.

Session 3 conjectured: greedy is optimal iff the externality is
"action-symmetric". That was stated loosely. The graded model sharpens it.

Model. Each arm carries an externality coefficient e_a >= 0. When arm a dies
endogenously it adds `delta * e_a` of permanent burden subtracted from every
subsequent reward. S5 was the binary special case e_a in {0, 1}.

Refined conjecture (C1'):
    greedy is optimal  <=>  the EXPECTED externality per pull, p_a * e_a,
                            is constant across arms.

Rationale: pulling arm a creates burden `delta * e_a` with probability `p_a`.
What differentiates arms for planning purposes is the product, not either
factor alone. This predicts five cases:

    A  p varies, e = 0          product constant (0)  -> greedy optimal
    B  p constant, e constant   product constant      -> greedy optimal
    C  p constant, e varies     product varies        -> greedy BREAKS
    D  p varies,  e constant    product varies        -> greedy BREAKS
    E  p varies,  e varies, but p*e held CONSTANT     -> greedy optimal

A, B, C already have support from exp04 (S1, S4, S5). D and E are new, and E
is the decisive test: it can only come out right if the product is the correct
invariant, since both factors vary individually.

Also measured: does the optimality gap grow with the spread of p_a * e_a?
"""

from functools import lru_cache
import numpy as np


def solve(v, p_vec, e_vec, delta, T):
    """Exact DP with graded externality. Returns (V*, greedy value)."""
    n = len(v)
    full = frozenset(range(n))
    e_total = float(np.sum(e_vec))

    def burden(S):
        dead = e_total - sum(e_vec[i] for i in S)
        return delta * dead

    def reward(a, S):
        return float(v[a] - burden(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        best = -np.inf
        for a in S:
            pa = p_vec[a]
            val = reward(a, S) + pa * V(S - {a}, t + 1) + (1 - pa) * V(S, t + 1)
            best = max(best, val)
        return best

    @lru_cache(maxsize=None)
    def Vg(S, t):
        if t >= T or not S:
            return 0.0
        a = max(S, key=lambda i: reward(i, S))
        pa = p_vec[a]
        return reward(a, S) + pa * Vg(S - {a}, t + 1) + (1 - pa) * Vg(S, t + 1)

    return V(full, 0), Vg(full, 0)


rng = np.random.default_rng(11)
N, T, DELTA, INST = 6, 8, 0.12, 10


def vals():
    return np.sort(rng.uniform(0.4, 1.2, size=N))[::-1].copy()


def case(name, make, inst=INST):
    gaps, spreads = [], []
    for _ in range(inst):
        v = vals()
        p_vec, e_vec = make()
        vs, vg = solve(v, p_vec, e_vec, DELTA, T)
        gaps.append((vs - vg) / abs(vs) * 100)
        spreads.append(float(np.std(p_vec * e_vec)))
    broken = sum(g > 1e-7 for g in gaps)
    print(f"{name:46} {np.mean(spreads):9.4f} {np.mean(gaps):8.3f}% "
          f"{broken:>4}/{inst}")
    return np.mean(spreads), np.mean(gaps)


print("C1' test: greedy optimal <=> p_a * e_a constant across arms\n")
print(f"{'case':46} {'std(p*e)':>9} {'mean gap':>9} {'broken':>8}")
print("-" * 76)

case("A  p varies, e=0",
     lambda: (rng.uniform(0.2, 1.0, N), np.zeros(N)))

case("B  p constant, e constant",
     lambda: (np.full(N, 0.5), np.full(N, 1.0)))

case("C  p constant, e varies",
     lambda: (np.full(N, 0.5), rng.uniform(0.0, 2.0, N)))

case("D  p varies, e constant",
     lambda: (rng.uniform(0.2, 1.0, N), np.full(N, 1.0)))


def case_e():
    """p varies, e varies, product pinned to a constant."""
    p_vec = rng.uniform(0.25, 1.0, N)
    k = 0.4
    e_vec = k / p_vec           # so p_a * e_a == k for every arm
    return p_vec, e_vec


case("E  p and e both vary, product HELD CONSTANT", case_e)

print("\n\nGap vs spread of p*e (C prediction: monotone increasing)\n")
print(f"{'target std(p*e)':>16} {'actual':>9} {'mean gap':>9}")
print("-" * 38)
for scale in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]:
    gaps, sds = [], []
    for _ in range(INST):
        v = vals()
        p_vec = np.full(N, 0.5)
        e_vec = np.clip(1.0 + rng.normal(0, scale, N) / 0.5, 0, None)
        vs, vg = solve(v, p_vec, e_vec, DELTA, T)
        gaps.append((vs - vg) / abs(vs) * 100)
        sds.append(float(np.std(p_vec * e_vec)))
    print(f"{scale:16.2f} {np.mean(sds):9.4f} {np.mean(gaps):8.3f}%")
