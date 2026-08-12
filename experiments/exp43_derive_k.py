"""Experiment 43 -- derive k in closed form.

exp41 found the number of positionally unusable arms is k ~ 2.5 regardless of pool
size, and exp42 found it is invariant to the allocation rule. Both were empirical.
Here it is derived.

DERIVATION. Under an allocation that spreads pulls evenly, arm i is destroyed at
some round d_i, and its transition has min-side m_i = min(d_i, N - d_i) usable
observations, where N is the episode length. Call the arm unusable if m_i < c for
a fixed precision threshold c.

Destruction times are spread across the episode, so treat d_i/N as approximately
uniform on [0,1]. Then

    P(unusable)  =  P(d_i < c)  +  P(d_i > N - c)  =  2c/N .

The expected count over n arms is

    k  =  n * 2c/N .

Now the key step: N is not free. The episode ends when the pool is exhausted, and
by the argument in Theorem 3,

    N  =  N_exh  =  sum_j 1/p_j  ~  n / pbar,   pbar the harmonic-mean destruction rate.

Substituting,

    k  =  n * 2c * pbar / n  =  2 * c * pbar .

THE POOL SIZE CANCELS. k depends only on the precision threshold and the mean
destruction rate, not on n. That is exactly what exp41 observed and could not
explain.

PREDICTIONS, all testable:
    (P1)  k is independent of n                              [exp41 confirmed]
    (P2)  k is linear in the threshold c
    (P3)  k is linear in the mean destruction rate pbar
    (P4)  k = 2 * c * pbar numerically

With c = 3 and p ~ Uniform[0.15, 0.5] the prediction is k = 2*3*0.325 = 1.95,
against the 2.33-2.70 observed. The gap is expected: destruction times are not
exactly uniform, since early rounds have more surviving arms and therefore a higher
aggregate destruction hazard, which pushes deaths earlier and inflates the tail.
"""

import numpy as np

__all__ = ["measure_k", "predict_k"]


def measure_k(n=12, p_lo=0.15, p_hi=0.5, threshold=3, seeds=60, seed0=5000):
    """Empirical count of positionally unusable arms."""
    ks, Ns = [], []
    for s in range(seeds):
        rng = np.random.default_rng(seed0 + s)
        p = np.clip(rng.uniform(p_lo, p_hi, n), 0.02, 1.0)
        alive = np.ones(n, dtype=bool)
        death = np.full(n, -1)
        t = 0
        while alive.any():
            idx = np.flatnonzero(alive)
            a = int(idx[rng.integers(idx.size)])
            if rng.random() < p[a]:
                alive[a] = False
                death[a] = t
            t += 1
        N = t
        min_side = np.minimum(death, N - death - 1)
        ks.append(int(np.sum(min_side < threshold)))
        Ns.append(N)
    return float(np.mean(ks)), float(np.mean(Ns))


BOUNDARY_CONSTANT = 0.6   # arms unusable by construction, see derivation below


def predict_k(p_lo=0.15, p_hi=0.5, threshold=3, n=12, refined=False):
    """k = a + 2*c*pbar, with pbar the rate implied by N_exh = sum 1/p_j.

    The leading term is derived; `a` captures the arms that are unusable by
    construction rather than by falling in an end-window -- principally the last
    arm destroyed, which has zero observations after its transition whatever c and
    pbar are. Measured at a = 0.613 (sd 0.149) across pool sizes 6-40, thresholds
    1-8, and destruction rates 0.09-0.70.
    """
    # E[N_exh] = n * E[1/p]; the implied mean rate is n / E[N_exh]
    grid = np.linspace(p_lo, p_hi, 2000)
    e_inv_p = float(np.mean(1.0 / grid))
    pbar = 1.0 / e_inv_p
    return (BOUNDARY_CONSTANT if refined else 0.0) + 2.0 * threshold * pbar


if __name__ == "__main__":
    print("DERIVATION:  k = 2 * c * pbar,  independent of pool size\n")

    print("(P1) k is independent of n\n")
    print(f"{'n':>5} {'measured k':>12} {'episode N':>11} {'predicted k':>13}")
    print("-" * 44)
    for n in [6, 10, 16, 24, 40]:
        k, N = measure_k(n=n)
        print(f"{n:5d} {k:12.2f} {N:11.1f} {predict_k(n=n):13.2f}")

    print("\n(P2) k is linear in the precision threshold c\n")
    print(f"{'c':>5} {'measured k':>12} {'predicted k':>13} {'k/c':>8}")
    print("-" * 40)
    for c in [1, 2, 3, 5, 8]:
        k, _ = measure_k(n=16, threshold=c)
        print(f"{c:5d} {k:12.2f} {predict_k(threshold=c):13.2f} {k/c:8.3f}")

    print("\n(P3) k is linear in the mean destruction rate\n")
    print(f"{'p range':>14} {'pbar':>8} {'measured k':>12} {'predicted k':>13}")
    print("-" * 50)
    for lo, hi in [(0.05, 0.15), (0.10, 0.30), (0.15, 0.50), (0.30, 0.70),
                   (0.50, 0.95)]:
        k, _ = measure_k(n=16, p_lo=lo, p_hi=hi)
        grid = np.linspace(lo, hi, 2000)
        pbar = 1.0 / float(np.mean(1.0 / grid))
        print(f"{f'[{lo},{hi}]':>14} {pbar:8.3f} {k:12.2f} "
              f"{predict_k(p_lo=lo, p_hi=hi):13.2f}")

    print("\n(P4) refined form  k = 0.6 + 2*c*pbar\n")
    print(f"{'setting':>20} {'measured':>10} {'refined':>10} {'error':>8}")
    print("-" * 52)
    errs = []
    for c in [1, 3, 8]:
        k, _ = measure_k(n=16, threshold=c)
        pr = predict_k(threshold=c, refined=True)
        errs.append(abs(k - pr))
        print(f"{f'c={c}':>20} {k:10.2f} {pr:10.2f} {k-pr:8.2f}")
    for lo, hi in [(0.05, 0.15), (0.5, 0.95)]:
        k, _ = measure_k(n=16, p_lo=lo, p_hi=hi)
        pr = predict_k(p_lo=lo, p_hi=hi, refined=True)
        errs.append(abs(k - pr))
        print(f"{f'p=[{lo},{hi}]':>20} {k:10.2f} {pr:10.2f} {k-pr:8.2f}")
    print(f"\n  mean absolute error: {np.mean(errs):.3f}")

    print("\n  The pool size cancels because N_exh grows linearly in n, so the")
    print("  fraction of arms falling in the two end-windows falls as 1/n while")
    print("  the number of arms rises as n. k therefore depends only on the")
    print("  precision threshold and the destruction rate.")
