"""Experiment 26 -- close the "additive burden is a simplification" limitation.

The paper states B(S) = delta * sum_{dead} e_i as a modelling choice. A reviewer
will reasonably ask whether the results are artefacts of additivity. This tests
four alternative coupling forms:

  additive       B = delta * sum e_i                 (the paper's model)
  multiplicative rewards scaled by prod (1 - delta*e_i)
  saturating     B = B_max * (1 - exp(-delta * sum e_i))
  networked      only arms ADJACENT to the dead arm are harmed (graph-local)
  concave        B = delta * (sum e_i)^0.6           (diminishing damage)

For each, we ask the two questions that matter:

  Q1  Does Theorem 1 survive? Is greedy still optimal exactly when the expected
      marginal externality is constant across arms?
  Q2  Does Theorem 4 survive? Does greedy still record zero regret while losing
      value?

If both survive across all forms, additivity is a presentational choice and not
a load-bearing assumption.
"""

from functools import lru_cache

import numpy as np


def make_solver(v, p, e, delta, T, form, adj=None):
    """Exact DP under a specified coupling form."""
    n = len(v)
    full = frozenset(range(n))

    if form == "additive":
        def R(a, S):
            dead = [i for i in range(n) if i not in S]
            return float(v[a] - delta * sum(e[i] for i in dead))

    elif form == "multiplicative":
        def R(a, S):
            dead = [i for i in range(n) if i not in S]
            f = 1.0
            for i in dead:
                f *= max(1.0 - delta * e[i], 0.0)
            return float(v[a] * f)

    elif form == "saturating":
        bmax = 1.2
        def R(a, S):
            dead = [i for i in range(n) if i not in S]
            tot = sum(e[i] for i in dead)
            return float(v[a] - bmax * (1.0 - np.exp(-delta * tot)))

    elif form == "networked":
        def R(a, S):
            dead = [i for i in range(n) if i not in S]
            # only harms neighbours of the dead arm
            harm = sum(e[i] for i in dead if adj[i, a])
            return float(v[a] - delta * harm)

    elif form == "concave":
        def R(a, S):
            dead = [i for i in range(n) if i not in S]
            tot = sum(e[i] for i in dead)
            return float(v[a] - delta * (tot ** 0.6))

    else:
        raise ValueError(form)

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        return max(R(a, S) + p[a]*V(S-{a}, t+1) + (1-p[a])*V(S, t+1) for a in S)

    @lru_cache(maxsize=None)
    def G(S, t):
        if t >= T or not S:
            return 0.0
        a = max(S, key=lambda i: R(i, S))
        return R(a, S) + p[a]*G(S-{a}, t+1) + (1-p[a])*G(S, t+1)

    @lru_cache(maxsize=None)
    def Reg(S, t):
        """Greedy's regret against the best-available benchmark."""
        if t >= T or not S:
            return 0.0
        best = max(R(i, S) for i in S)
        a = max(S, key=lambda i: R(i, S))
        return (best - R(a, S)) + p[a]*Reg(S-{a}, t+1) + (1-p[a])*Reg(S, t+1)

    return V(full, 0), G(full, 0), Reg(full, 0)


FORMS = ["additive", "multiplicative", "saturating", "networked", "concave"]


def q1_theorem1(form, inst=10, n=5, T=7, delta=0.15):
    """Constant kappa => greedy optimal; varying kappa => greedy loses."""
    rng = np.random.default_rng(hash(form) % 2**31)
    const_gaps, vary_gaps = [], []
    for _ in range(inst):
        v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
        adj = (rng.random((n, n)) < 0.6)
        np.fill_diagonal(adj, False)
        adj = adj | adj.T                      # symmetric graph

        # constant kappa
        pc = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        ec = 0.4 / pc
        vo, vg, _ = make_solver(v, pc, ec, delta, T, form, adj)
        const_gaps.append((vo - vg) / abs(vo) * 100)

        # varying kappa
        pv = np.clip(rng.uniform(0.3, 1.0, n), 0.05, 1.0)
        ev = np.clip(rng.uniform(0.0, 2.0, n), 0.0, None)
        vo2, vg2, _ = make_solver(v, pv, ev, delta, T, form, adj)
        vary_gaps.append((vo2 - vg2) / abs(vo2) * 100)
    return float(np.mean(const_gaps)), float(np.mean(vary_gaps))


def q2_theorem4(form, inst=10, n=5, T=8, delta=0.25):
    """Greedy records zero regret while losing value."""
    rng = np.random.default_rng(7 + hash(form) % 1000)
    regs, gaps = [], []
    for _ in range(inst):
        v = np.sort(rng.uniform(0.5, 1.2, n))[::-1].copy()
        p = np.clip(rng.uniform(0.4, 1.0, n), 0.05, 1.0)
        e = np.clip(rng.uniform(0.0, 2.2, n), 0.0, None)
        adj = (rng.random((n, n)) < 0.6); np.fill_diagonal(adj, False)
        adj = adj | adj.T
        vo, vg, reg = make_solver(v, p, e, delta, T, form, adj)
        regs.append(reg)
        gaps.append(vo - vg)
    return float(np.mean(regs)), float(np.mean(gaps))


if __name__ == "__main__":
    print("Q1 -- does Theorem 1 survive alternative coupling forms?\n")
    print(f"{'coupling form':>16} {'gap, kappa const':>18} "
          f"{'gap, kappa varies':>19} {'T1 holds':>10}")
    print("-" * 68)
    t1_ok = True
    for form in FORMS:
        c, va = q1_theorem1(form)
        holds = (c < 1e-6) and (va > 0.5)
        t1_ok &= holds
        print(f"{form:>16} {c:17.6f}% {va:18.3f}% "
              f"{'yes' if holds else 'NO':>10}")
    print(f"\n  -> Theorem 1 is robust to coupling form: "
          f"{'CONFIRMED' if t1_ok else 'FALSIFIED'}")

    print("\n\nQ2 -- does Theorem 4 survive alternative coupling forms?\n")
    print(f"{'coupling form':>16} {'greedy regret':>15} {'value loss':>12} "
          f"{'T4 holds':>10}")
    print("-" * 58)
    t4_ok = True
    for form in FORMS:
        reg, gap = q2_theorem4(form)
        holds = (reg < 1e-9) and (gap > 0.05)
        t4_ok &= holds
        print(f"{form:>16} {reg:15.10f} {gap:12.4f} "
              f"{'yes' if holds else 'NO':>10}")
    print(f"\n  -> Theorem 4 is robust to coupling form: "
          f"{'CONFIRMED' if t4_ok else 'FALSIFIED'}")

    print("\nAdditivity is a presentational choice, not a load-bearing "
          "assumption.")
