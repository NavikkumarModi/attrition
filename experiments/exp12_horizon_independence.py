"""Experiment 12 -- decisive test of the impossibility claim.

Theorem 3 (target): estimation error for the externality vector is bounded
below by a quantity that depends on the CONSUMPTION rate, not on the horizon.

Reasoning. To estimate e_i you compare reward levels before and after arm i
dies, so the standard error obeys
    SE(delta*e_i) >= sigma * sqrt(1/n_before + 1/n_after) >= 2*sigma/sqrt(N)
where N is the episode length. But the episode ends when the pool empties, so
    E[N] = sum_i 1/p_i  ->  n/p  under homogeneous p,
independent of T once T exceeds it. Hence

    SE >= 2*sigma*sqrt(p/n),  and this floor is INDEPENDENT OF T.

Sharp prediction: raising T improves estimation at low p (until the pool-limit
binds) and does nothing at high p. If instead accuracy keeps improving with T
at high p, the claim is false.
"""
import numpy as np
from experiments.exp10_learning import Sim

def trial(p_const, T, seed, n=8, spread=1.0):
    sim = Sim(n=n, T=T, spread=spread, seed=seed)
    sim.p = np.full(n, p_const)
    sim.rng = np.random.default_rng(10_000 + seed); sim.reset()
    X, y = [], []
    while sim.t < sim.T and sim.alive.any():
        idx = np.flatnonzero(sim.alive)
        a = int(sim.rng.choice(idx))
        dead_before = ~sim.alive.copy()
        obs, _, _ = sim.step(a)
        row = np.zeros(2*n); row[a] = 1.0; row[n:][dead_before] = -1.0
        X.append(row); y.append(obs)
    X, y = np.array(X), np.array(y)
    beta = np.linalg.lstsq(X.T@X + 1e-6*np.eye(2*n), X.T@y, rcond=None)[0]
    died = ~sim.alive
    truth = sim.delta * sim.e
    if died.sum() < 3:
        return len(y), np.nan, np.nan
    rmse = float(np.sqrt(np.mean((beta[n:][died] - truth[died])**2)))
    corr = float(np.corrcoef(beta[n:][died], truth[died])[0, 1])
    return len(y), rmse, corr

if __name__ == "__main__":
    SEEDS, N = 60, 8
    print("Does more horizon buy better estimates?\n")
    print(f"{'p':>5} {'n/p':>6} {'T':>6} {'rounds':>8} {'RMSE':>8} {'corr':>7}")
    print("-" * 46)
    for p in [0.9, 0.5, 0.1]:
        for T in [20, 40, 80, 160, 320]:
            R = [trial(p, T, s, n=N) for s in range(SEEDS)]
            print(f"{p:5.2f} {N/p:6.1f} {T:6d} "
                  f"{np.mean([r[0] for r in R]):8.1f} "
                  f"{np.nanmean([r[1] for r in R]):8.4f} "
                  f"{np.nanmean([r[2] for r in R]):+7.3f}")
        print()
