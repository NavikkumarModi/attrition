"""Experiment 27 -- test the reviewer's objection to Theorem 3.

OBJECTION. The proof of Theorem 3 says "the MLE of delta*e_i is the difference
of pre- and post-destruction sample means". But the burden is cumulative:
B(S) = delta * sum_{dead} e_j. If arm i dies and then arm j dies, post-i rounds
contain BOTH level shifts. So a naive pre/post difference does not isolate e_i,
and calling it the MLE is wrong.

The objection is correct. This script establishes the repair.

REPAIR. The difference-in-means estimator is the MLE of delta*e_i *conditional
on all other e_j being known*. In the general case those are also unknown and
must be estimated jointly, which can only add variance -- adding free parameters
to a linear model weakly increases the variance of every coefficient. Therefore

    Var(joint LS estimator)  >=  Var(oracle-others-known estimator)
                             =  sigma^2 (1/n_b + 1/n_a)
                             >= 4 sigma^2 / N

so the floor remains valid as a LOWER bound, now correctly derived. The naive
estimator is not claimed to be the MLE of the real problem; it is the estimator
whose variance lower-bounds it.

This script verifies the inequality directly, which is what the repaired proof
needs.
"""

import numpy as np


def simulate(n, T, p, e, delta, sigma, rng):
    """Run a random-allocation episode; return design matrix and observations."""
    v = np.sort(rng.uniform(0.4, 1.2, n))[::-1].copy()
    alive = np.ones(n, dtype=bool)
    rows, ys, t = [], [], 0
    death_time = np.full(n, -1)
    while t < T and alive.any():
        idx = np.flatnonzero(alive)
        a = int(idx[rng.integers(len(idx))])
        dead_before = ~alive.copy()
        burden = delta * float(e[dead_before].sum())
        ys.append(v[a] - burden + rng.normal(0, sigma))
        row = np.zeros(2 * n)
        row[a] = 1.0
        row[n:][dead_before] = -1.0
        rows.append(row)
        if rng.random() < p[a]:
            alive[a] = False
            death_time[a] = t
        t += 1
    return np.array(rows), np.array(ys), v, death_time, t


def variances(X, sigma, n, target):
    """Variance of the joint LS estimate vs the oracle-others-known estimate."""
    # joint: estimate all v and all delta*e together
    XtX = X.T @ X + 1e-9 * np.eye(2 * n)
    try:
        joint = sigma**2 * np.linalg.inv(XtX)[n + target, n + target]
    except np.linalg.LinAlgError:
        return None, None

    # oracle: all other delta*e_j known, so drop their columns
    keep = list(range(n)) + [n + target]
    Xo = X[:, keep]
    XtXo = Xo.T @ Xo + 1e-9 * np.eye(len(keep))
    try:
        oracle = sigma**2 * np.linalg.inv(XtXo)[-1, -1]
    except np.linalg.LinAlgError:
        return None, None
    return joint, oracle


if __name__ == "__main__":
    sigma, delta = 0.3, 0.1
    rng = np.random.default_rng(4)

    print("REPAIR CHECK: Var(joint LS) >= Var(oracle-others-known) "
          ">= sigma^2(1/n_b + 1/n_a)\n")
    print(f"{'n':>3} {'rounds':>7} {'arm':>4} {'Var joint':>11} "
          f"{'Var oracle':>11} {'AM-HM floor':>12} {'chain holds':>12}")
    print("-" * 66)

    ok = True
    checked = 0
    for trial in range(14):
        n = int(rng.integers(4, 7))
        T = int(rng.integers(25, 60))
        p = np.clip(rng.uniform(0.15, 0.6, n), 0.05, 1.0)
        e = np.clip(rng.uniform(0.2, 2.0, n), 0.0, None)
        X, y, v, dt, rounds = simulate(n, T, p, e, delta, sigma, rng)
        # pick an arm that died with data on both sides
        cands = [i for i in range(n) if 0 < dt[i] < rounds - 1]
        if not cands:
            continue
        target = int(rng.choice(cands))
        n_b, n_a = int(dt[target]), int(rounds - dt[target] - 1)
        if n_b < 1 or n_a < 1:
            continue
        joint, oracle = variances(X, sigma, n, target)
        if joint is None:
            continue
        floor = sigma**2 * (1.0/n_b + 1.0/n_a)
        holds = (joint >= oracle - 1e-9) and (oracle >= floor - 1e-6)
        ok &= holds
        checked += 1
        print(f"{n:3d} {rounds:7d} {target:4d} {joint:11.5f} {oracle:11.5f} "
              f"{floor:12.5f} {'yes' if holds else 'NO':>12}")

    print(f"\n  instances checked: {checked}")
    print(f"  -> Var(joint) >= Var(oracle) >= AM-HM floor: "
          f"{'CONFIRMED' if ok else 'FALSIFIED'}")
    print()
    print("  Consequence: the floor 2*sigma/sqrt(N) is valid as a LOWER bound on")
    print("  the real (joint) estimation problem, because the oracle-others-known")
    print("  estimator is strictly easier and already meets it. The original")
    print("  proof's claim that difference-in-means is 'the MLE' was wrong; the")
    print("  bound it produces is not.")
