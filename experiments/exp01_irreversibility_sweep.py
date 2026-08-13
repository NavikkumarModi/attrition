"""Experiment 01 -- does irreversibility actually break departure-blind LinUCB?

Sweeps endogenous departure probability p from 0 (classical) to 1 (full
irreversibility) with no exogenous churn and no arrivals, so the effect is
attributable to irreversibility alone.
"""
import numpy as np
from attrition.runner import sweep
from attrition.agents import LinUCBAgent, RandomAgent, IPICAgent

ENV = dict(d=5, n_init=60, mu=0.0, lam=0.0, sigma=0.1, horizon=1500)
PS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
SEEDS = 20

def main():
    rows = []
    for label, fac in [
        ("random",  lambda env: RandomAgent(seed=env.seed)),
        ("linucb",  lambda env: LinUCBAgent(d=env.d, alpha=1.0)),
        ("ipic",    lambda env: IPICAgent(d=env.d, p=env.p, alpha=1.0,
                                          horizon=env.horizon)),
    ]:
        res = sweep(fac, PS, n_seeds=SEEDS, **ENV)
        for p, st in res.items():
            rows.append((label, p, st))

    print(f"{'agent':8} {'p':>5} {'regret':>12} {'±se':>7} {'pulls':>7} {'exh':>5}")
    print("-" * 50)
    for label, p, st in rows:
        print(f"{label:8} {p:5.2f} {st['regret_mean']:12.2f} "
              f"{st['regret_se']:7.2f} {st['pulls_mean']:7.0f} "
              f"{st['exhausted_frac']:5.2f}")

if __name__ == "__main__":
    main()
