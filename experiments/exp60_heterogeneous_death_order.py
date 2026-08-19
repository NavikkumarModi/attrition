"""Experiment 60 -- resolving the paper's stated-open heterogeneous-rate
generalization of the exchangeability and memorylessness lemmas.

THE PROBLEM (as stated in the paper). Under homogeneous destruction rate p,
the death order is a uniform random permutation (Lemma: Exchangeability) and
inter-death gaps are i.i.d. Geometric(p) (Lemma: Memorylessness), giving an
exact formula for k(p,c). Heterogeneous p_a breaks both lemmas: higher-p arms
die faster and are biased toward earlier ranks, and gaps are no longer i.i.d.
The paper states: "A general closed form would need the order statistics of
death times under this rate-dependent sorting -- a genuinely different and
harder problem than the homogeneous case just solved, and not yet derived."

RESOLVED: both pieces, each with a short proof, each independently verified.

PART 1 -- Death order (generalized exchangeability).
Under uniform-random selection among alive arms, with arm a dying with
probability p_a when selected, the death order follows a Plackett-Luce
sequential distribution:
    P(order = i_1,...,i_n) = prod_{k=1}^{n} p_{i_k} / S_k
where S_k is the sum of p_a over arms still alive just before the k-th death
(S_1 = sum of all p_a; S_k = S_{k-1} - p_{i_{k-1}}).

PROOF (renewal argument). Let f_i := P(arm i dies before any other, from an
alive set A). Conditioning on the first round's pick (uniform among |A|
arms) and outcome:
    f_i = (1/|A|)*p_i + f_i * (1/|A|) * sum_{j in A}(1-p_j)
        = (1/|A|)*p_i + f_i * (|A| - S_A)/|A|
Solving gives f_i = p_i / S_A exactly -- rounds where nobody dies simply
restart the identical sub-problem and cancel from the equation. By the
Markov property, conditional on who dies first, the process among survivors
is a FRESH instance of the same dynamics with the same rates restricted to
the smaller alive set (no memory of how the state was reached), so the same
argument applies recursively -- giving the full sequential product exactly.

PART 2 -- Death timing (generalized memorylessness).
Given a fixed alive set A with sum S_A and size |A|, each round independently
produces a death (any arm) with probability S_A/|A| (pick uniformly, then
that arm's own Bernoulli(p) coin) -- so the WAITING TIME until the next
death, from a fixed alive set, is exactly Geometric(S_A/|A|), and gaps
across the sequence of alive sets determined by the death order are
independent geometrics with the rate updated at each death.

TOGETHER these give the complete generating mechanism for the joint
(order, timing) distribution under heterogeneous rates -- the piece the
paper states was "not yet derived." A simple universal closed form for
k(p_1,...,p_n,c) analogous to the homogeneous k=1+5p case is not claimed;
what is resolved is the underlying order-statistics structure the paper
identifies as the missing ingredient.
"""

from collections import Counter
from functools import lru_cache
from itertools import permutations

import numpy as np

__all__ = ["plackett_luce_order_prob", "simulate_death_order",
           "verify_order_distribution", "verify_gap_distribution"]


def plackett_luce_order_prob(order, p):
    """Exact probability of a specific full death order under heterogeneous p."""
    remaining = dict(enumerate(p))
    prob = 1.0
    total = sum(p)
    for a in order[:-1]:
        prob *= remaining[a] / total
        total -= remaining[a]
    return prob


def simulate_death_order(p, rng):
    n = len(p)
    alive = list(range(n))
    order = []
    while alive:
        idx = rng.integers(len(alive))
        a = alive[idx]
        if rng.random() < p[a]:
            order.append(a)
            alive.pop(idx)
    return tuple(order)


def verify_order_distribution(p, n_sims=500_000, seed=0):
    """Returns max absolute error between simulated and Plackett-Luce
    predicted probabilities, across all n! orderings."""
    rng = np.random.default_rng(seed)
    counts = Counter()
    for _ in range(n_sims):
        counts[simulate_death_order(p, rng)] += 1
    max_err = 0.0
    for order in permutations(range(len(p))):
        emp = counts.get(order, 0) / n_sims
        pred = plackett_luce_order_prob(order, p)
        max_err = max(max_err, abs(emp - pred))
    return max_err


def verify_gap_distribution(p_alive, n_sims=300_000, seed=5, max_gap=6):
    """Returns max absolute error between simulated and predicted
    Geometric(S/m) PMF for the gap until the next death from a fixed
    alive set."""
    rng = np.random.default_rng(seed)
    m = len(p_alive)
    S = float(np.sum(p_alive))
    rate = S / m
    gaps = np.zeros(n_sims, dtype=int)
    for i in range(n_sims):
        g = 0
        while True:
            g += 1
            idx = rng.integers(m)
            if rng.random() < p_alive[idx]:
                break
        gaps[i] = g
    max_err = 0.0
    for k in range(1, max_gap + 1):
        emp = float(np.mean(gaps == k))
        pred = (1 - rate) ** (k - 1) * rate
        max_err = max(max_err, abs(emp - pred))
    return max_err


if __name__ == "__main__":
    print("Part 1: death order follows Plackett-Luce exactly\n")
    p = [0.15, 0.35, 0.55]
    err = verify_order_distribution(p, n_sims=1_000_000)
    print(f"  p={p}: max error across all 6 orderings = {err:.5f}")

    print("\nPart 2: inter-death gap ~ Geometric(S_A/|A|) from a fixed alive set\n")
    p_alive = np.array([0.1, 0.25, 0.4, 0.6])
    err2 = verify_gap_distribution(p_alive)
    print(f"  p_alive={p_alive}: max PMF error = {err2:.5f}")
