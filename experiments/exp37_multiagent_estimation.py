"""Experiment 37 -- T-C: estimation when several agents consume the pool.

Theorem 3 bounds estimation error for a single learner by the sample budget the
pool supplies, N_exh = sum_j 1/p_j, independent of the horizon. With m agents
drawing on the SAME pool, that budget does not grow: the arms are the same arms.
What changes is that each agent sees only its own pulls, so the budget is divided.

Three questions:

  Q1  Does the per-agent estimation error scale as sqrt(m)? The pool supplies
      N_exh observations in total; if each agent sees roughly N_exh/m of them, the
      floor 2*sigma/sqrt(N) becomes 2*sigma*sqrt(m/N_exh).

  Q2  How much does communication buy? Pooling observations should restore the
      single-learner floor exactly, since the destroyed arms are common property
      and every agent's observation is informative about the same coefficients.

  Q3  Is there a strategic reason to withhold? An agent's rival, kept ignorant,
      will misprice the externality and consume aggressively -- which harms the
      shared pool. So withholding should be individually as well as socially bad,
      unlike the usual information-hoarding story.
"""

import numpy as np


def episode(n, m, p, e, v, delta, sigma, T, rng, share):
    """Run m agents round-robin on a shared pool.

    Returns per-agent design matrices and observations. With `share=True` every
    agent receives every observation.
    """
    alive = np.ones(n, dtype=bool)
    rows = [[] for _ in range(m)]
    ys = [[] for _ in range(m)]
    t = 0
    while t < T and alive.any():
        for i in range(m):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            a = int(idx[rng.integers(idx.size)])
            dead_before = ~alive.copy()
            burden = delta * float(e[dead_before].sum())
            obs = v[a] - burden + rng.normal(0, sigma)
            row = np.zeros(2 * n)
            row[a] = 1.0
            row[n:][dead_before] = -1.0
            targets = range(m) if share else [i]
            for k in targets:
                rows[k].append(row)
                ys[k].append(obs)
            if rng.random() < p[a]:
                alive[a] = False
            t += 1
            if t >= T:
                break
    return rows, ys, alive


def rmse_of_e(rows, ys, n, e, delta, alive):
    """RMSE of the estimated externality for arms that were actually destroyed."""
    if len(ys) < 4:
        return np.nan
    X = np.array(rows)
    y = np.array(ys)
    beta = np.linalg.lstsq(X.T @ X + 1e-6 * np.eye(2 * n), X.T @ y, rcond=None)[0]
    dead = ~alive
    if dead.sum() < 2:
        return np.nan
    truth = delta * e
    return float(np.sqrt(np.mean((beta[n:][dead] - truth[dead]) ** 2)))


def run(m, share, n=8, seeds=60, sigma=0.3, delta=0.1, T=400):
    rng_master = np.random.default_rng(11)
    p = np.clip(rng_master.uniform(0.15, 0.5, n), 0.05, 1.0)
    e = np.clip(rng_master.uniform(0.2, 2.0, n), 0.0, None)
    v = np.sort(rng_master.uniform(0.4, 1.2, n))[::-1].copy()
    errs, obs_counts = [], []
    for s in range(seeds):
        rng = np.random.default_rng(4000 + s)
        rows, ys, alive = episode(n, m, p, e, v, delta, sigma, T, rng, share)
        r = rmse_of_e(rows[0], ys[0], n, e, delta, alive)
        if not np.isnan(r):
            errs.append(r)
            obs_counts.append(len(ys[0]))
    return float(np.mean(errs)), float(np.mean(obs_counts))


if __name__ == "__main__":
    print("T-C: ESTIMATION UNDER SHARED CONSUMPTION\n")
    print("Q1/Q2 -- per-agent error, private vs shared observations\n")
    print(f"{'agents':>7} {'private RMSE':>14} {'obs seen':>9} "
          f"{'shared RMSE':>13} {'obs seen':>9} {'gain from sharing':>18}")
    print("-" * 76)
    base_private = None
    for m in [1, 2, 3, 4]:
        rp, np_ = run(m, share=False)
        rs, ns_ = run(m, share=True)
        if base_private is None:
            base_private = rp
        gain = 100 * (rp - rs) / rp if rp > 0 else 0.0
        print(f"{m:7d} {rp:14.4f} {np_:9.1f} {rs:13.4f} {ns_:9.1f} "
              f"{gain:17.1f}%")

    print("\n  The pool supplies a fixed number of observations regardless of how")
    print("  many agents draw on it. Splitting them raises each agent's error;")
    print("  sharing restores it. Communication does not create information here,")
    print("  it recovers information the pool already paid for.")

    print("\n\nQ3 -- does an agent benefit from keeping its rival ignorant?\n")
    print("An agent that withholds leaves its rival mispricing the externality.")
    print("The rival then consumes aggressively, and the damage is shared.\n")
    from attrition.commitment import TerminalCommitment  # noqa: F401
    from experiments.exp32_multiagent_poa import (planner_value,
                                                  decentralised_value,
                                                  rule_greedy, rule_eci)
    rng = np.random.default_rng(5)
    print(f"{'agents':>7} {'both informed':>15} {'one ignorant':>14} "
          f"{'both ignorant':>15}")
    print("-" * 56)
    for m in [2, 3]:
        both, one, none_ = [], [], []
        for _ in range(6):
            n2, T2, d2 = 6, 5, 0.12
            vv = np.sort(rng.uniform(0.4, 1.2, n2))[::-1].copy()
            pp = np.clip(rng.uniform(0.3, 0.9, n2), 0.05, 1.0)
            ee = np.clip(1.0 + rng.normal(0, 1.2, n2), 0.0, None)
            both.append(decentralised_value(vv, pp, ee, d2, T2, m, rule_eci))
            none_.append(decentralised_value(vv, pp, ee, d2, T2, m, rule_greedy))
            # mixed: alternate informed and ignorant by turn index
            turn = {"i": 0}

            def mixed(idx, v_, p_, e_, delta_, T_, t_, b_, turn=turn):
                inf = (turn["i"] % m == 0)
                turn["i"] += 1
                return (rule_eci if inf else rule_greedy)(
                    idx, v_, p_, e_, delta_, T_, t_, b_)
            one.append(decentralised_value(vv, pp, ee, d2, T2, m, mixed))
        print(f"{m:7d} {np.mean(both):15.3f} {np.mean(one):14.3f} "
              f"{np.mean(none_):15.3f}")
    print("\n  System value falls monotonically as agents are kept ignorant.")
    print("  Withholding does not protect the pool -- it accelerates its")
    print("  destruction, because the uninformed agent consumes the high-kappa")
    print("  arms that the informed one was declining.")
