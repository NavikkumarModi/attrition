"""Experiment 04 -- when does greedy-optimality break?

Session 2 proved greedy optimal for the base model (independent arms,
stationary rewards, geometric death). This battery probes the boundary.

Scenarios:
  S1  heterogeneous irreversibility   p_a varies per arm
  S2  discounting                     future worth less
  S3  rotting rewards                 arm value decays with each pull
  S4  commons degradation             consuming arms degrades ALL rewards
  S5  competitive release             consuming an arm RAISES a rival's value

S4 and S5 introduce cross-arm coupling, which the base model lacks entirely.
Coupling is the realistic case in the motivating domains (competitive release
in oncology, shared-capacity suppliers in CMC), so if greedy survives coupling
the model is genuinely uninteresting; if it breaks, that is where the research
problem lives.
"""

from functools import lru_cache
import numpy as np


# ---------------------------------------------------------------- generic DP

def solve(values_fn, n, T, p_vec, gamma=1.0):
    """Exact DP. values_fn(a, S) -> immediate reward of pulling a in state S.

    Returns (V*, optimal first action, greedy-policy value).
    """
    full = frozenset(range(n))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        best = -np.inf
        for a in S:
            r = values_fn(a, S)
            pa = p_vec[a]
            cont = pa * V(S - {a}, t + 1) + (1 - pa) * V(S, t + 1)
            best = max(best, r + gamma * cont)
        return best

    @lru_cache(maxsize=None)
    def Vg(S, t):
        """Value of the greedy policy: always pull highest immediate reward."""
        if t >= T or not S:
            return 0.0
        a = max(S, key=lambda i: values_fn(i, S))
        r = values_fn(a, S)
        pa = p_vec[a]
        return r + gamma * (pa * Vg(S - {a}, t + 1) + (1 - pa) * Vg(S, t + 1))

    v_star = V(full, 0)
    a_star = max(full, key=lambda a: values_fn(a, full) + gamma * (
        p_vec[a] * V(full - {a}, 1) + (1 - p_vec[a]) * V(full, 1)))
    return v_star, a_star, Vg(full, 0)


def report(name, rows):
    print(f"\n=== {name} ===")
    print(f"{'inst':>4} {'V*':>9} {'greedy':>9} {'gap':>9} {'gap%':>7} {'greedy opt?':>12}")
    print("-" * 56)
    broken = 0
    for i, (vs, vg, ok) in enumerate(rows):
        gap = vs - vg
        pct = 100 * gap / abs(vs) if vs != 0 else 0.0
        flag = "yes" if gap < 1e-9 else "NO"
        if gap >= 1e-9:
            broken += 1
        print(f"{i:4d} {vs:9.4f} {vg:9.4f} {gap:9.5f} {pct:7.3f} {flag:>12}")
    print(f"  -> greedy suboptimal in {broken}/{len(rows)} instances")
    return broken


rng = np.random.default_rng(7)
N_INST = 8


def rand_vals(n):
    return np.sort(rng.uniform(0.2, 1.0, size=n))[::-1].copy()


# ---------------------------------------------------------------- S1 hetero p
rows = []
for _ in range(N_INST):
    n, T = 6, 7
    v = rand_vals(n)
    p_vec = rng.uniform(0.05, 1.0, size=n)
    vs, a, vg = solve(lambda a, S: v[a], n, T, p_vec)
    rows.append((vs, vg, a == 0))
report("S1  heterogeneous irreversibility", rows)

# ---------------------------------------------------------------- S2 discount
rows = []
for _ in range(N_INST):
    n, T = 6, 8
    v = rand_vals(n)
    p_vec = np.full(n, rng.choice([0.3, 0.7, 1.0]))
    vs, a, vg = solve(lambda a, S: v[a], n, T, p_vec, gamma=0.8)
    rows.append((vs, vg, a == 0))
report("S2  discounting (gamma=0.8)", rows)

# ---------------------------------------------------------------- S3 rotting
# value falls with the number of arms already consumed by that arm's own use;
# approximated here as decay in the number of DEAD arms is S4, so here we make
# each arm's value decay in how long it has been available (time-based rot).
rows = []
for _ in range(N_INST):
    n, T = 6, 8
    v = rand_vals(n)
    p_vec = np.full(n, 0.5)
    # rot: value shrinks with elapsed rounds -> encoded via |S| proxy is weak,
    # so use explicit time by folding t into the state through T-stage recursion
    # (handled by making values depend on |S| as a monotone proxy for time)
    vs, a, vg = solve(lambda a, S: v[a] * (0.9 ** (n - len(S))), n, T, p_vec)
    rows.append((vs, vg, a == 0))
report("S3  rotting (decay in arms consumed)", rows)

# ---------------------------------------------------------------- S4 commons
rows = []
for _ in range(N_INST):
    n, T = 6, 8
    v = rand_vals(n)
    p_vec = np.full(n, rng.choice([0.4, 0.8, 1.0]))
    c = 0.9  # severity: reward scaled by fraction of arms still alive
    vs, a, vg = solve(lambda a, S: v[a] * (1 - c * (n - len(S)) / n),
                      n, T, p_vec)
    rows.append((vs, vg, a == 0))
report("S4  commons degradation (consuming arms degrades all)", rows)

# ---------------------------------------------------------------- S5 release
# Arm n-1 is 'resistant': its value RISES as other arms are consumed, but it is
# a poor arm to pull. Consuming sensitive arms releases it. Reward of every arm
# is penalised by the released mass.
rows = []
for _ in range(N_INST):
    n, T = 6, 8
    v = rand_vals(n)
    p_vec = np.full(n, rng.choice([0.5, 1.0]))
    delta = 0.35

    def vals(a, S, v=v, n=n, delta=delta):
        dead_sensitive = len([i for i in range(n - 1) if i not in S])
        burden = delta * dead_sensitive
        return max(v[a] - burden, 0.0)

    vs, a, vg = solve(vals, n, T, p_vec)
    rows.append((vs, vg, a == 0))
report("S5  competitive release (consumption raises a rival's burden)", rows)
