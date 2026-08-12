"""Experiment 08 -- the corrected index.

exp07 established that p_a * e_a is the invariant: greedy is optimal iff it is
constant, and the gap grows with its dispersion. That immediately suggests the
correction. Pulling arm a creates expected permanent burden delta*p_a*e_a,
which is charged against every remaining round. So the right index is

    I(a, t) = v_a  -  delta * p_a * e_a * (T - t)

Greedy uses v_a alone; this adds the discounted externality the pull will
impose on the rest of the horizon. If this index recovers the DP optimum, the
result is not merely "greedy breaks" but "here is the closed-form policy that
fixes it".
"""
from functools import lru_cache
import numpy as np

def solve_all(v, p, e, delta, T):
    n = len(v); full = frozenset(range(n)); etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S: return 0.0
        return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)

    def policy_value(pick):
        @lru_cache(maxsize=None)
        def W(S, t):
            if t >= T or not S: return 0.0
            a = pick(S, t)
            return R(a, S) + p[a]*W(S-{a}, t+1) + (1-p[a])*W(S, t+1)
        return W(full, 0)

    greedy = policy_value(lambda S, t: max(S, key=lambda i: R(i, S)))
    index  = policy_value(lambda S, t: max(
        S, key=lambda i: R(i, S) - delta*p[i]*e[i]*(T-t)))
    return V(full, 0), greedy, index

rng = np.random.default_rng(23)
N, T, DELTA, INST = 6, 8, 0.12, 12

print(f"{'spread':>8} {'V*':>8} {'greedy':>8} {'index':>8} "
      f"{'greedy gap':>11} {'index gap':>10} {'captured':>9}")
print("-" * 70)
for scale in [0.1, 0.2, 0.4, 0.8, 1.2]:
    gg, ig, sd = [], [], []
    for _ in range(INST):
        v = np.sort(rng.uniform(0.4, 1.2, N))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, N), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, scale, N), 0, None)
        vs, vg, vi = solve_all(v, p, e, DELTA, T)
        gg.append((vs-vg)/abs(vs)*100); ig.append((vs-vi)/abs(vs)*100)
        sd.append(float(np.std(p*e)))
    g, i = np.mean(gg), np.mean(ig)
    cap = 100*(g-i)/g if g > 1e-9 else 0.0
    print(f"{np.mean(sd):8.3f} {'':>8} {'':>8} {'':>8} "
          f"{g:10.3f}% {i:9.3f}% {cap:8.1f}%")
