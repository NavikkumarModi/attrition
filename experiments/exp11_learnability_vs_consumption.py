"""Experiment 11 -- irreversibility destroys the data needed to price it.

exp10 found the learning policy captured ~0% of the 47% gap available to the
oracle index. Diagnosis showed why, and it is structural rather than a bug:

To estimate arm i's externality e_i you must compare reward levels BEFORE and
AFTER arm i dies. Death is irreversible, so each arm supplies exactly ONE such
transition. The precision of the estimate therefore depends entirely on how
many rounds are spent in each burden state -- and high p empties the pool
before any averaging can happen.

This sweeps the consumption rate and measures recovery of e.
"""
import numpy as np
from experiments.exp10_learning import Sim

def trial(lo, hi, seed, spread=1.0):
    sim = Sim(spread=spread, seed=seed)
    sim.p = np.clip(np.random.default_rng(seed).uniform(lo, hi, sim.n), 0.02, 1.0)
    sim.rng = np.random.default_rng(seed); sim.reset()
    n = sim.n; X, y = [], []
    n_states, last = 0, 0
    while sim.t < sim.T and sim.alive.any():
        idx = np.flatnonzero(sim.alive)
        a = int(sim.rng.choice(idx))
        dead_before = ~sim.alive.copy()
        obs, _, _ = sim.step(a)
        row = np.zeros(2*n); row[a] = 1.0; row[n:][dead_before] = -1.0
        X.append(row); y.append(obs)
        d = int((~sim.alive).sum())
        if d != last: n_states += 1; last = d
    X, y = np.array(X), np.array(y)
    M = X.T @ X
    beta = np.linalg.lstsq(M + 1e-6*np.eye(2*n), X.T @ y, rcond=None)[0]
    died = ~sim.alive
    corr = (np.corrcoef(beta[n:][died], (sim.delta*sim.e)[died])[0, 1]
            if died.sum() >= 3 else np.nan)
    return len(y), len(y)/max(n_states, 1), corr

if __name__ == "__main__":
    print(f"{'p range':>14} {'rounds':>8} {'rounds/state':>13} {'corr(e_hat,e)':>14}")
    print("-" * 52)
    for lo, hi in [(0.9,1.0),(0.5,1.0),(0.25,0.5),(0.10,0.25),(0.04,0.10)]:
        R = [trial(lo, hi, s) for s in range(25)]
        print(f"{f'[{lo},{hi}]':>14} {np.mean([r[0] for r in R]):8.1f} "
              f"{np.mean([r[1] for r in R]):13.1f} "
              f"{np.nanmean([r[2] for r in R]):+13.3f}")
