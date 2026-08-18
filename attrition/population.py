"""Population simulation: many policies -- LLM-driven or classical -- acting
in a shared consumable environment. This is the OASIS-style driver for the
package: a named group of agents, each with its own decision-making, sharing
one resource pool.

Two loop shapes, matching the two multi-agent models already in this repo:

    simulate_population               turn-taking: agents pull one after
                                       another each round from a shared
                                       `ConsumableBandit`, each seeing the
                                       pool exactly as earlier agents in the
                                       same round left it. By Theorem 6 this
                                       is exactly equivalent to one learner
                                       over m*T pulls, so it has no genuine
                                       price of anarchy -- useful as a
                                       baseline-comparable setting, not as a
                                       coordination-failure demo.

    simulate_population_simultaneous  genuine simultaneous action over a
                                       `SimultaneousPool`: every agent commits
                                       before anyone's pull resolves, so
                                       collisions and blindness are real
                                       effects. Use this when "what happens
                                       when many independent decision-makers
                                       share a resource" is the actual
                                       question -- e.g. several prescribers
                                       drawing on one population's
                                       antimicrobial susceptibility.
"""

import numpy as np

from .consumable import run
from .policies import State

__all__ = ["Population", "simulate_population", "simulate_population_simultaneous",
           "compare_population_to_baselines"]


class Population:
    """A named group of agents, each anything with a `.select(state)` method."""

    def __init__(self, members):
        """members: dict agent_id -> policy."""
        self.members = dict(members)

    @classmethod
    def from_personas(cls, personas, client=None, policy_cls=None):
        """One `LLMPolicy` per persona, all sharing one client."""
        from .llm_policy import LLMPolicy
        policy_cls = policy_cls or LLMPolicy
        return cls({p.name: policy_cls(p, client=client) for p in personas})

    def __iter__(self):
        return iter(self.members.items())

    def __len__(self):
        return len(self.members)


def simulate_population(env, population, known=True, log=True):
    """Run a population turn-taking on a shared `ConsumableBandit`.

    Every agent in `population` pulls once per round, in a fixed order, until
    the environment is done (horizon reached or every arm destroyed).
    """
    env.reset()
    per_agent = {aid: {"value": 0.0, "regret": 0.0, "pulls": 0}
                 for aid in population.members}
    trace = []
    while not env.done():
        for aid, policy in population.members.items():
            if env.done():
                break
            st = env.state(known=known)
            arm = policy.select(st)
            _, realised, regret, destroyed = env.step(arm)
            per_agent[aid]["value"] += realised
            per_agent[aid]["regret"] += regret
            per_agent[aid]["pulls"] += 1
            if log:
                trace.append({"t": env.t - 1, "agent": aid, "arm": arm,
                              "value": realised, "regret": regret,
                              "destroyed": destroyed,
                              "alive": int(env.alive.sum())})
    system_value = sum(a["value"] for a in per_agent.values())
    system_regret = sum(a["regret"] for a in per_agent.values())
    return {"agents": per_agent, "system_value": system_value,
            "system_regret": system_regret, "trace": trace}


def simulate_population_simultaneous(pool, population, rounds=None, log=True):
    """Run a population with every agent committing simultaneously each round
    on a shared `SimultaneousPool`, then resolving together.
    """
    pool.reset()
    rounds = pool.T if rounds is None else int(rounds)
    agent_ids = list(population.members)
    per_agent = {aid: {"value": 0.0, "regret": 0.0, "pulls": 0}
                 for aid in agent_ids}
    trace = []
    for r in range(rounds):
        if not pool.alive.any():
            break
        idx = np.flatnonzero(pool.alive)
        b = pool.burden
        best_available = float(np.max(pool.v[idx] - b))
        actions = []
        for aid in agent_ids:
            st = State(available=idx, t=pool.t, horizon=rounds, v_hat=pool.v,
                      p=pool.p, e=pool.e, delta=pool.delta)
            actions.append(population.members[aid].select(st))
        rewards, destroyed = pool.step(actions)
        for aid, arm, reward in zip(agent_ids, actions, rewards):
            regret = best_available - reward
            per_agent[aid]["value"] += reward
            per_agent[aid]["regret"] += regret
            per_agent[aid]["pulls"] += 1
            if log:
                trace.append({"t": r, "agent": aid, "arm": arm, "value": reward,
                              "regret": regret, "destroyed": arm in destroyed,
                              "alive": int(pool.alive.sum())})
    system_value = sum(a["value"] for a in per_agent.values())
    system_regret = sum(a["regret"] for a in per_agent.values())
    return {"agents": per_agent, "system_value": system_value,
            "system_regret": system_regret, "trace": trace}


def compare_population_to_baselines(env_factory, population, baselines, seeds=10):
    """Compare a population's system value/regret against classical
    single-agent baselines (e.g. `Greedy()`, `ECI()`) on matched seeded
    `ConsumableBandit` environments from `env_factory`, over the same total
    pull budget (`env.horizon`). This is the direct operational form of the
    paper's question applied to a population: does it land near Greedy
    (zero regret, high harm) or near ECI (positive regret, low harm)?

    Note: `population`'s policy instances are reused and reset across seeds;
    for `LLMPolicy` this means `.log` accumulates across the whole sweep
    rather than reflecting a single seed's run.
    """
    pop_vals, pop_regs = [], []
    for s in range(seeds):
        r = simulate_population(env_factory(s), population)
        pop_vals.append(r["system_value"])
        pop_regs.append(r["system_regret"])
    out = {"population": {"value": float(np.mean(pop_vals)),
                          "regret": float(np.mean(pop_regs))}}
    for pol in baselines:
        vals, regs = [], []
        for s in range(seeds):
            res = run(env_factory(s), pol, seed=s)
            vals.append(res["value"])
            regs.append(res["regret"])
        out[pol.name] = {"value": float(np.mean(vals)),
                         "regret": float(np.mean(regs))}
    return out
