"""Experiment runner.

Regret here is measured against the *myopic* best-available-arm sequence:

    R(T) = sum_t [ max_{a in A_t} <theta*, x_a>  -  <theta*, x_{a_t}> ]

This is a diagnostic benchmark, not the Whittle-index oracle from the spec.
It is well-defined and cheap, and it is enough to answer the first empirical
question: does irreversibility actually break departure-blind algorithms?
"""

import numpy as np
from .env import EvolvingBanditEnv


def run_episode(env, agent):
    inst_regret = []
    pulls = 0
    while env.t < env.horizon and not env.is_exhausted():
        arms = env.available
        best = env.best_available_mean()
        arm = agent.select(arms, env)
        chosen_mean = float(env.theta @ arm.x)
        reward, _ = env.step(arm.idx)
        agent.update(arm, reward)
        inst_regret.append(best - chosen_mean)
        pulls += 1
    return {
        "cum_regret": float(np.sum(inst_regret)),
        "pulls": pulls,
        "regret_curve": np.cumsum(inst_regret),
        "exhausted": env.is_exhausted(),
        "arms_left": len(env.arms),
    }


def sweep(agent_factory, p_values, n_seeds=20, **env_kwargs):
    """Run agent_factory across p values and seeds. Returns dict p -> stats."""
    out = {}
    for p in p_values:
        regrets, pulls, exhausted = [], [], 0
        for s in range(n_seeds):
            env = EvolvingBanditEnv(p=p, seed=s, **env_kwargs)
            agent = agent_factory(env)
            res = run_episode(env, agent)
            regrets.append(res["cum_regret"])
            pulls.append(res["pulls"])
            exhausted += int(res["exhausted"])
        out[p] = {
            "regret_mean": float(np.mean(regrets)),
            "regret_se": float(np.std(regrets) / np.sqrt(n_seeds)),
            "pulls_mean": float(np.mean(pulls)),
            "exhausted_frac": exhausted / n_seeds,
        }
    return out
