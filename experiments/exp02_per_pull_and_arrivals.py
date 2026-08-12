"""Experiment 02.

Two corrections to exp01:
 (a) report regret PER PULL -- cumulative regret is confounded by episode
     length, since higher p exhausts the arm pool sooner.
 (b) test the arrivals regime. With no arrivals and a finite pool, every arm
     is consumed regardless of policy, so the problem is pure *sequencing*
     and there is no reason to preserve a good arm. Preservation should only
     pay when an arm can be re-pulled, i.e. p<1 with a live population.
"""
import numpy as np
from evolving_bandits.env import EvolvingBanditEnv
from evolving_bandits.runner import run_episode
from evolving_bandits.agents import LinUCBAgent, IPICAgent, RandomAgent

SEEDS = 10

def cell(fac, n_seeds=SEEDS, **kw):
    per_pull, cum, pulls = [], [], []
    for s in range(n_seeds):
        env = EvolvingBanditEnv(seed=s, **kw)
        agent = fac(env)
        r = run_episode(env, agent)
        if r["pulls"] > 0:
            per_pull.append(r["cum_regret"] / r["pulls"])
            cum.append(r["cum_regret"]); pulls.append(r["pulls"])
    return (np.mean(per_pull), np.std(per_pull)/np.sqrt(len(per_pull)),
            np.mean(cum), np.mean(pulls))

def block(title, env_kw, ps):
    print(f"\n=== {title} ===")
    print(f"{'agent':8} {'p':>5} {'regret/pull':>12} {'±se':>7} {'cum':>9} {'pulls':>7}")
    print("-" * 54)
    for p in ps:
        kw = dict(env_kw); kw["p"] = p
        for label, fac in [
            ("linucb", lambda env: LinUCBAgent(d=env.d, alpha=1.0)),
            ("ipic",   lambda env: IPICAgent(d=env.d, p=env.p, alpha=1.0,
                                             horizon=env.horizon)),
        ]:
            m, se, c, pl = cell(fac, **kw)
            print(f"{label:8} {p:5.2f} {m:12.4f} {se:7.4f} {c:9.1f} {pl:7.0f}")

if __name__ == "__main__":
    block("A. no arrivals (pure sequencing)",
          dict(d=5, n_init=60, mu=0.0, lam=0.0, sigma=0.1, horizon=800),
          [0.0, 0.1, 0.5, 1.0])
    block("B. with arrivals (arms reusable, pool sustained)",
          dict(d=5, n_init=60, mu=0.0, lam=1.0, sigma=0.1, horizon=800),
          [0.0, 0.1, 0.5, 1.0])
    block("C. arrivals + exogenous churn (mixed regime)",
          dict(d=5, n_init=60, mu=0.01, lam=1.5, sigma=0.1, horizon=800),
          [0.0, 0.1, 0.5, 1.0])
