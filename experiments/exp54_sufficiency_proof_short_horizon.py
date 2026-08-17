"""Experiment 54 -- a complete, rigorous proof of Theorem 1 sufficiency for
the pre-exhaustion regime (T < n), following a reframing proposed in external
review: define the externality-only continuation cost C(S,h) (all v_a set to
0) and show the FIRST-CHOICE-invariance directly via an exact closed form.

THE LEMMA (proven below, verified to machine precision).
For h < |S|, phi(S,h) := [C(S,h) - C(S-{x},h)] / e_x is (a) independent of
which x in S is used to compute it, and (b) exactly equal to -delta*h.

PROOF, by induction on h.
Base case h=0: C(S,0)=0 for all S, so phi(S,0)=0=-delta*0. Trivial.

Inductive step. Assume phi(S',h-1) = -delta*(h-1) for every S' with
h-1 < |S'| (this covers both S and S-{x}, since S-{x} has one fewer arm and we
are working in the regime h < |S|, so h-1 < |S|-1 = |S-{x}|).

First, because phi(S,h-1) is independent of x (by the inductive hypothesis),
the Bellman minimisation collapses:
    C(S,h) = B(S) + min_a [p_a C(S-a,h-1) + (1-p_a) C(S,h-1)]
           = B(S) + C(S,h-1) + min_a [-p_a e_a phi(S,h-1)]
           = B(S) + C(S,h-1) - kappa * phi(S,h-1)
using p_a e_a = kappa (the constant-kappa hypothesis) -- EVERY choice of a
gives the identical value, so the minimum is achieved everywhere, not just
attained by one arm.

Then, using B(S) - B(S-x) = -delta*e_x (direct from the definition of B) and
C(S,h-1) - C(S-x,h-1) = e_x * phi(S,h-1) = -delta*(h-1)*e_x (inductive
hypothesis, applied to arm x specifically):

    C(S,h) - C(S-x,h) = [B(S)-B(S-x)] + [C(S,h-1)-C(S-x,h-1)]
                       = -delta*e_x - delta*(h-1)*e_x
                       = -delta*e_x*h

Dividing by e_x gives phi(S,h) = -delta*h, independent of x. This closes the
induction. QED.

THE THEOREM (T1 sufficiency, T < n case; proven below).
For h < |S|, Q_C(S,a,h) = Q_C(S,b,h) whenever kappa_a = kappa_b, a,b in S.

PROOF. Using C(S-x,h-1) = C(S,h-1) + delta*(h-1)*e_x (from the Lemma, valid
since h-1 < |S| - 1 required S-x's own regime -- true here since h < |S|):

    Q_C(S,a,h) - Q_C(S,b,h)
      = p_a C(S-a,h-1) - p_b C(S-b,h-1) + (p_b-p_a) C(S,h-1)
      = (p_a-p_b) C(S,h-1) + delta(h-1)[p_a e_a - p_b e_b] + (p_b-p_a) C(S,h-1)
      = delta(h-1) [kappa_a - kappa_b]
      = 0   when kappa_a = kappa_b.

Combined with v_a >= v_b implying Q(S,a,h) - Q(S,b,h) = v_a - v_b >= 0 (the
private-value term adds on directly since the externality-continuation term is
now provably identical, not just empirically so), greedy is optimal at every
state and horizon T < n.

EMPIRICAL EXTENSION (not proven). The same Q_C-invariance is found to hold
exactly, at machine precision, in the exhaustion regime h >= |S| too, even
though phi(S,h) is no longer simply -delta*h there. This strongly suggests
the theorem holds in full generality, via an infinite hierarchy of
higher-order ratio invariants that empirically also hold exactly but whose
general induction has not been completed. Recorded honestly as evidence
toward a fully general proof, not as one.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np

__all__ = ["build_C", "phi", "verify_lemma", "verify_theorem"]


def build_C(p, e, delta):
    n = len(p)
    etot = float(np.sum(e))
    B = lambda S: delta * (etot - sum(e[i] for i in S))

    @lru_cache(maxsize=None)
    def C(S, h):
        if h == 0 or not S:
            return 0.0
        return B(S) + min(p[a]*C(S-{a}, h-1) + (1-p[a])*C(S, h-1) for a in S)

    return C, B


def phi(C, S, h, e):
    x0 = next(iter(S))
    denom = e[x0]
    if len(S) == 1:
        return C(S, h) / denom
    return (C(S, h) - C(S - {x0}, h)) / denom


def verify_lemma(seeds=15, seed0=0):
    """phi(S,h) = -delta*h exactly, for h < |S|, across all subsets and all
    valid h, on `seeds` random constant-kappa instances."""
    rng = np.random.default_rng(seed0)
    max_err = 0.0
    for _ in range(seeds):
        n = int(rng.integers(3, 7))
        kappa = rng.uniform(0.2, 2.0)
        p = np.clip(rng.uniform(0.1, 0.9, n), 0.05, 1.0)
        e = kappa / p
        delta = rng.uniform(0.1, 1.0)
        C, B = build_C(p, e, delta)
        for size in range(1, n + 1):
            for combo in combinations(range(n), size):
                S = frozenset(combo)
                for h in range(size):
                    phi_val = phi(C, S, h, e)
                    max_err = max(max_err, abs(phi_val - (-delta*h)))
    return max_err


def verify_theorem(seeds=20, seed0=0):
    """Q_C(S,a,h) = Q_C(S,b,h) for kappa_a=kappa_b, h < |S| strict, across all
    subsets, all valid h, all arm pairs, on `seeds` random instances."""
    rng = np.random.default_rng(seed0)
    max_err, checked = 0.0, 0
    for _ in range(seeds):
        n = int(rng.integers(3, 7))
        kappa = rng.uniform(0.2, 2.0)
        p = np.clip(rng.uniform(0.1, 0.9, n), 0.05, 1.0)
        e = kappa / p
        delta = rng.uniform(0.1, 1.0)
        C, B = build_C(p, e, delta)
        for size in range(2, n + 1):
            for combo in combinations(range(n), size):
                S = frozenset(combo)
                for h in range(1, size):
                    def QC(a, hh):
                        return B(S) + p[a]*C(S-{a}, hh-1) + (1-p[a])*C(S, hh-1)
                    arms = list(S)
                    base = QC(arms[0], h)
                    for a in arms[1:]:
                        checked += 1
                        max_err = max(max_err, abs(QC(a, h) - base))
    return max_err, checked


if __name__ == "__main__":
    print("LEMMA: phi(S,h) = -delta*h exactly, for h < |S|\n")
    err = verify_lemma()
    print(f"  max error across all (S,h,instance): {err:.2e}")

    print("\nTHEOREM: Q_C(S,a,h) = Q_C(S,b,h) for kappa_a=kappa_b, h < |S|\n")
    err, n = verify_theorem()
    print(f"  {n} comparisons checked, max error: {err:.2e}")

    print("\nEXTENSION EVIDENCE (not proven): does the invariance persist")
    print("past h >= |S| (exhaustion regime)?\n")
    rng = np.random.default_rng(0)
    n = 4
    kappa = 1.0
    p = np.clip(rng.uniform(0.1, 0.9, n), 0.05, 1.0)
    e = kappa / p
    delta = 0.4
    full = frozenset(range(n))
    C, B = build_C(p, e, delta)
    for h in [4, 5, 6, 7]:
        def QC(a, hh):
            return B(full) + p[a]*C(full-{a}, hh-1) + (1-p[a])*C(full, hh-1)
        vals = [QC(a, h) for a in full]
        print(f"  h={h} (>= n={n}): spread = {max(vals)-min(vals):.2e}")
