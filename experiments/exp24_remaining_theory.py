"""Experiment 24 -- close the remaining open theory.

Three items outstanding:

  (i)   T1 NECESSITY. Sufficiency is proven (Lemma 1a', exp23). Necessity needs
        a construction: kappa not constant => greedy strictly suboptimal.

  (ii)  T2 EXACTNESS. The ECI index captures 52-99% of the gap empirically. Is
        there a regime where it is EXACTLY optimal? Conjecture: yes, in the pure
        sequencing regime (p=1, T=n), where every arm is pulled exactly once and
        the only decision is order. There the burden charge delta*kappa_a*(T-t)
        is exact rather than approximate, because the number of remaining pulls
        is known with certainty.

  (iii) CONJECTURE 1a. Gap monotone in dispersion of kappa.
"""

from functools import lru_cache
from itertools import permutations
import numpy as np


def dp_values(v, p, e, delta, T):
    n = len(v); full = frozenset(range(n)); etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))
    R = lambda a, S: float(v[a] - B(S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S: return 0.0
        return max(R(a,S) + p[a]*V(S-{a},t+1) + (1-p[a])*V(S,t+1) for a in S)

    def pol(pick):
        @lru_cache(maxsize=None)
        def W(S, t):
            if t >= T or not S: return 0.0
            a = pick(S, t)
            return R(a,S) + p[a]*W(S-{a},t+1) + (1-p[a])*W(S,t+1)
        return W(full, 0)

    greedy = pol(lambda S,t: max(S, key=lambda i: R(i,S)))
    eci = pol(lambda S,t: max(S, key=lambda i: R(i,S) - delta*p[i]*e[i]*(T-t)))
    return V(full,0), greedy, eci


def main():
    # ---------------------------------------------------------------- (i)
    print("(i) T1 NECESSITY -- kappa varies => greedy strictly suboptimal\n")
    print("Construction: hot arm (v=1, kappa=E) + m safe arms (v=1-eps, kappa=0)")
    print("Closed form gap = m*delta*E > 0 whenever kappa is not constant.\n")
    print(f"{'m':>3} {'kappa_hot':>10} {'kappa_safe':>11} {'std(kappa)':>11} "
          f"{'gap':>9} {'predicted':>10} {'exact':>6}")
    print("-" * 66)
    ok_i = True
    for m in [1, 3, 6, 10]:
        for E in [0.5, 1.5]:
            eps, delta = 0.01, 1.0
            n = m + 1
            v = np.array([1.0] + [1.0-eps]*m)
            p = np.ones(n)
            e = np.array([E] + [0.0]*m)
            vo, vg, _ = dp_values(v, p, e, delta, n+1)
            kap = p*e
            pred = m * delta * E
            exact = abs((vo-vg) - pred) < 1e-9
            ok_i &= exact
            print(f"{m:3d} {kap[0]:10.2f} {kap[1]:11.2f} {np.std(kap):11.4f} "
                  f"{vo-vg:9.4f} {pred:10.4f} {'yes' if exact else 'NO':>6}")
    print(f"\n  -> necessity: {'CONFIRMED' if ok_i else 'FALSIFIED'}")
    print("     Any kappa dispersion admits an instance with strictly positive gap.")
    
    
    # ---------------------------------------------------------------- (ii)
    print("\n\n(ii) T2 EXACTNESS -- is ECI exactly optimal in pure sequencing?\n")
    print("Regime: p=1 (every pull consumes), T=n (all arms pulled exactly once).\n")
    print(f"{'n':>3} {'std(kappa)':>11} {'V*':>9} {'greedy':>9} {'ECI':>9} "
          f"{'ECI gap':>9} {'exact':>6}")
    print("-" * 60)
    rng = np.random.default_rng(202)
    ok_ii = True
    for trial in range(10):
        n = int(rng.integers(3, 7))
        v = np.sort(rng.uniform(0.3, 1.3, n))[::-1].copy()
        p = np.ones(n)
        e = np.clip(rng.uniform(0.0, 2.0, n), 0, None)
        delta = 0.15
        vo, vg, vi = dp_values(v, p, e, delta, n)
        gap = vo - vi
        exact = gap < 1e-9
        ok_ii &= exact
        print(f"{n:3d} {np.std(p*e):11.4f} {vo:9.4f} {vg:9.4f} {vi:9.4f} "
              f"{gap:9.6f} {'yes' if exact else 'NO':>6}")
    print(f"\n  -> ECI exactly optimal in pure sequencing: "
          f"{'CONFIRMED' if ok_ii else 'FALSIFIED'}")
    
    # brute-force cross-check on one instance
    print("\n  brute-force check against all n! orderings (n=5):")
    rng2 = np.random.default_rng(9)
    v = np.sort(rng2.uniform(0.3, 1.3, 5))[::-1].copy()
    e = rng2.uniform(0.0, 2.0, 5); delta = 0.2
    best, best_ord = -np.inf, None
    for perm in permutations(range(5)):
        tot, burden = 0.0, 0.0
        for a in perm:
            tot += v[a] - burden
            burden += delta * e[a]
        if tot > best: best, best_ord = tot, perm
    # ECI ordering
    order_eci, rem, burden = [], list(range(5)), 0.0
    for t in range(5):
        a = max(rem, key=lambda i: v[i] - burden - delta*1.0*e[i]*(5-t))
        order_eci.append(a); rem.remove(a); burden += delta*e[a]
    tot_eci, burden = 0.0, 0.0
    for a in order_eci:
        tot_eci += v[a] - burden; burden += delta*e[a]
    print(f"    best of 120 orderings : {best:.6f}  {best_ord}")
    print(f"    ECI ordering          : {tot_eci:.6f}  {tuple(order_eci)}")
    print(f"    match: {'YES' if abs(best-tot_eci) < 1e-9 else 'NO'}")
    
    
    # ---------------------------------------------------------------- (iii)
    print("\n\n(iii) CONJECTURE 1a -- gap monotone in dispersion of kappa\n")
    print(f"{'target':>8} {'std(kappa)':>11} {'greedy gap %':>13} {'ECI gap %':>11}")
    print("-" * 48)
    rng3 = np.random.default_rng(55)
    prev = -1.0
    mono = True
    for scale in [0.0, 0.1, 0.2, 0.4, 0.7, 1.1, 1.6]:
        gg, ii, sd = [], [], []
        for _ in range(12):
            n, T, delta = 6, 8, 0.12
            v = np.sort(rng3.uniform(0.4, 1.2, n))[::-1].copy()
            p = np.clip(rng3.uniform(0.3, 1.0, n), 0.05, 1.0)
            e = np.clip(1.0 + rng3.normal(0, scale, n), 0, None)
            vo, vg, vi = dp_values(v, p, e, delta, T)
            gg.append((vo-vg)/abs(vo)*100); ii.append((vo-vi)/abs(vo)*100)
            sd.append(float(np.std(p*e)))
        g = np.mean(gg)
        if g < prev - 1e-9: mono = False
        prev = g
        print(f"{scale:8.2f} {np.mean(sd):11.4f} {g:12.3f}% {np.mean(ii):10.3f}%")
    print(f"\n  -> monotone in dispersion: {'CONFIRMED' if mono else 'FALSIFIED'}")


if __name__ == "__main__":
    main()
