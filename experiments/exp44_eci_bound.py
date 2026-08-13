r"""Experiment 44 -- toward a bound on the ECI gap.

ECI is the main algorithmic contribution and the inventory lists it as empirical.
To prove a bound we first need to know what the gap actually scales with.

DECOMPOSITION. Maximising the Bellman expression is equivalent to maximising
v_a - p_a D_a, with D_a = V(S,t+1) - V(S\\{a},t+1). Splitting D_a into a burden and
an option component, D_a = delta e_a L(S,t) + O_a(S,t), the ECI score
v_a - delta kappa_a (T-t) makes two approximations:

    (E1)  it replaces the true burden horizon L(S,t) by the naive (T-t)
    (E2)  it drops the option term O_a entirely

so the per-arm score error is  delta kappa_a (L - (T-t)) + p_a O_a.

A one-step argument then bounds the value gap by 2 T max_a |error_a|, which is
correct but likely loose. Rather than assume a form, this measures how the gap
scales in each parameter separately, so the bound can be stated in the variables
that actually drive it.

Theorem 1 already pins one end: the gap is exactly zero when kappa is constant. So
the bound must vanish with kappa dispersion -- but earlier runs showed the ECI gap
roughly FLAT in dispersion while greedy's grew, which is worth resolving.
"""

from functools import lru_cache

import numpy as np


def gap(v, p, e, delta, T):
    """Return (greedy gap, ECI gap) as fractions of the optimum."""
    n = len(v); full = frozenset(range(n)); etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)

    def pol(pick):
        @lru_cache(maxsize=None)
        def W(S, t):
            if t >= T or not S:
                return 0.0
            a = pick(S, t)
            return R(a, S) + p[a]*W(S-{a}, t+1) + (1-p[a])*W(S, t+1)
        return W(full, 0)

    vs = V(full, 0)
    g = pol(lambda S, t: max(S, key=lambda i: R(i, S)))
    i_ = pol(lambda S, t: max(S, key=lambda i: R(i, S)
                              - delta*p[i]*e[i]*(T-t)))
    return (vs - g) / abs(vs), (vs - i_) / abs(vs)


def cell(n=6, T=8, delta=0.12, spread=1.0, inst=12, seed=0):
    rng = np.random.default_rng(seed)
    G, I, SD = [], [], []
    for _ in range(inst):
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        e = np.clip(1.0 + rng.normal(0, spread, n), 0.0, None)
        a, b = gap(v, p, e, delta, T)
        G.append(a); I.append(b); SD.append(float(np.std(p*e)))
    return float(np.mean(G)), float(np.mean(I)), float(np.mean(SD))


def sweep(param, values, **fixed):
    rows = []
    for val in values:
        kw = dict(n=6, T=8, delta=0.12, spread=1.0, seed=17)
        kw.update(fixed)
        kw[param] = val
        g, i, sd = cell(**kw)
        rows.append((val, sd, g, i))
    return rows


def show(title, param, rows, note=""):
    print(f"\n{title}\n")
    print(f"{param:>9} {'std(kappa)':>11} {'greedy gap':>12} {'ECI gap':>10} "
          f"{'ratio':>8}")
    print("-" * 54)
    for val, sd, g, i in rows:
        r = i / g if g > 1e-9 else float("nan")
        print(f"{val:9.3f} {sd:11.3f} {100*g:11.2f}% {100*i:9.2f}% {r:8.3f}")
    if note:
        print(f"  {note}")


if __name__ == "__main__":
    print("WHAT DOES THE ECI GAP SCALE WITH?")

    show("Dispersion of kappa (T1 says the gap must vanish at zero dispersion)",
         "spread", sweep("spread", [0.0, 0.2, 0.5, 1.0, 1.6, 2.4]))

    show("Externality scale delta", "delta",
         sweep("delta", [0.02, 0.06, 0.12, 0.25, 0.45]))

    show("Horizon T", "T", sweep("T", [4, 6, 8, 12, 16]))

    show("Pool size n", "n", sweep("n", [4, 5, 6, 7, 8]))

    print("\n  Theorem 1 fixes the left end exactly: zero dispersion, zero gap.")
    print("  The question is which parameter the gap grows in, and whether the")
    print("  growth is slow enough for a useful bound.")
