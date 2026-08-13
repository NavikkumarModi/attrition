"""Simultaneous-action multi-agent consumption.

The sequential model in exp32 turned out to be degenerate: with agents acting one
after another on a shared pool with full observation, m agents over T rounds is
exactly one learner over m*T pulls (proved in exp46). Decentralisation costs
nothing there, so the "price of anarchy" it appeared to measure was an artefact of
comparing against a planner with a different action space.

Genuine decentralisation requires that agents cannot condition on each other's
current choices. This module implements simultaneous action:

    - all m agents choose an arm at the same instant, seeing only the state at the
      start of the round;
    - all pulls then resolve together;
    - an arm chosen by several agents is pulled several times, and each pull is an
      independent destruction trial, so collisions raise the destruction hazard;
    - the burden created is visible only from the next round.

Two effects appear that sequential play cannot express:

    COLLISION   several agents converge on the same attractive arm, destroying it
                faster than any of them intended. A planner assigns distinct arms.
    BLINDNESS   no agent sees the damage its peers are doing this round, so all
                under-price the burden simultaneously.

This is the model in which a price of anarchy is a real quantity rather than a
comparison artefact.
"""

from functools import lru_cache
from itertools import product

import numpy as np

__all__ = ["SimultaneousPool", "planner_value_simultaneous",
           "decentralised_value_simultaneous", "price_of_anarchy"]


class SimultaneousPool:
    """Shared consumable pool under simultaneous action."""

    def __init__(self, v, p, e, delta=0.12, horizon=5, n_agents=2, seed=0):
        self.v = np.asarray(v, float)
        self.p = np.asarray(p, float)
        self.e = np.asarray(e, float)
        self.n = len(self.v)
        self.delta, self.T, self.m = float(delta), int(horizon), int(n_agents)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.alive = np.ones(self.n, dtype=bool)
        self.t = 0
        return self.alive.copy()

    @property
    def burden(self):
        return self.delta * float(self.e[~self.alive].sum())

    def step(self, actions):
        """All agents act at once. Returns (per-agent rewards, destroyed set)."""
        b = self.burden
        rewards = [float(self.v[a] - b) for a in actions]
        destroyed = set()
        for a in actions:                      # each pull is its own trial
            if self.alive[a] and self.rng.random() < self.p[a]:
                self.alive[a] = False
                destroyed.add(a)
        self.t += 1
        return rewards, destroyed


def _survive_prob(counts, p):
    """P(arm survives) given it was pulled `counts` times this round."""
    return (1.0 - p) ** counts


def planner_value_simultaneous(v, p, e, delta, T, m):
    """Exact system optimum when a coordinator assigns all m arms each round.

    The coordinator may assign the same arm to several agents, but knows the
    resulting destruction hazard and can spread agents across arms instead.
    """
    n = len(v)
    etot = float(np.sum(e))
    burden = lambda S: delta * (etot - sum(e[i] for i in S))

    @lru_cache(maxsize=None)
    def V(S, t):
        if t >= T or not S:
            return 0.0
        arms = sorted(S)
        b = burden(S)
        best = -np.inf
        # assignments are multisets of size m over available arms
        for assign in product(arms, repeat=m):
            counts = {a: assign.count(a) for a in set(assign)}
            imm = sum(v[a] - b for a in assign)
            uniq = sorted(counts)
            cont = 0.0
            for mask in range(1 << len(uniq)):
                pr, dead = 1.0, []
                for j, a in enumerate(uniq):
                    q = 1.0 - _survive_prob(counts[a], p[a])
                    if mask >> j & 1:
                        pr *= q
                        dead.append(a)
                    else:
                        pr *= (1.0 - q)
                if pr > 0:
                    cont += pr * V(S - frozenset(dead), t + 1)
            best = max(best, imm + cont)
        return best

    return V(frozenset(range(n)), 0)


def decentralised_value_simultaneous(v, p, e, delta, T, m, rule="greedy",
                                     seeds=600, tie_break="same"):
    """m agents choosing simultaneously, each optimising its own reward.

    `tie_break="same"` means every agent independently picks the same best arm --
    the pure coordination failure. `tie_break="random"` lets agents randomise among
    near-equal arms, a weak form of implicit coordination.
    """
    n = len(v)
    totals = []
    for s in range(seeds):
        pool = SimultaneousPool(v, p, e, delta, T, m, seed=9000 + s)
        tot = 0.0
        for _ in range(T):
            if not pool.alive.any():
                break
            idx = np.flatnonzero(pool.alive)
            if rule == "greedy":
                score = v[idx]
            elif rule == "eci":
                score = v[idx] - delta * p[idx] * e[idx] * (T - pool.t)
            else:
                raise ValueError(rule)
            if tie_break == "same":
                actions = [int(idx[np.argmax(score)])] * m
            else:
                order = idx[np.argsort(score)[::-1]]
                actions = [int(order[i % len(order)]) for i in range(m)]
            rewards, _ = pool.step(actions)
            tot += sum(rewards)
        totals.append(tot)
    return float(np.mean(totals))


def price_of_anarchy(v, p, e, delta, T, m, rule="greedy", tie_break="same"):
    pl = planner_value_simultaneous(v, p, e, delta, T, m)
    de = decentralised_value_simultaneous(v, p, e, delta, T, m, rule=rule,
                                          tie_break=tie_break)
    return pl / max(de, 1e-9), pl, de
