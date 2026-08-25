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
           "decentralised_value_simultaneous", "price_of_anarchy",
           "DegradingPool", "decentralised_value_degrading"]


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
                                     seeds=600, tie_break="same",
                                     lam=0.0, obs_noise=0.0,
                                     return_details=False):
    """m agents choosing simultaneously, each optimising its own reward.

    `tie_break="same"` means every agent independently picks the same best arm --
    the pure coordination failure. `tie_break="random"` lets agents randomise among
    near-equal arms, a weak form of implicit coordination.

    `rule="lambda_weighted"` generalises `"greedy"` (`lam=0`) and `"eci"`
    (`lam=1, obs_noise=0`) into a single dial: each agent scores an arm as
        v_a - lam * delta * kappa_hat_a * (T - t)
    where `kappa_hat_a = p_a * e_a` is optionally perturbed by mean-zero Gaussian
    noise of scale `obs_noise * std(kappa)` before the agent sees it -- a stand-in
    for the noisy/delayed/biased community-state estimates a real decentralised
    prescriber would have, matching abx_amr_simulator's partial-observability
    knob. This is the R_overall = (1-lam)*individual + lam*community dial from
    that paper, transplanted into a genuinely multi-agent, simultaneous-action
    setting it does not have.

    `return_details=True` returns both the true realised system value (what
    happened to the pool, including the burden term the scoring function never
    sees -- see module docstring's BLINDNESS) and the agents' own reported value
    (the score they computed for the arm they chose) instead of a single float.
    The gap between the two is the regret-harm divergence this rule exists to
    measure.
    """
    n = len(v)
    kappa = p * e
    kappa_scale = float(kappa.std()) if kappa.std() > 0 else 1.0
    totals, reported_totals = [], []
    for s in range(seeds):
        pool = SimultaneousPool(v, p, e, delta, T, m, seed=9000 + s)
        rng = np.random.default_rng(80_000 + s)
        tot, rep = 0.0, 0.0
        for _ in range(T):
            if not pool.alive.any():
                break
            idx = np.flatnonzero(pool.alive)
            if rule == "greedy":
                score = v[idx]
            elif rule == "eci":
                score = v[idx] - delta * p[idx] * e[idx] * (T - pool.t)
            elif rule == "lambda_weighted":
                kappa_hat = p[idx] * e[idx]
                if obs_noise > 0:
                    kappa_hat = kappa_hat + rng.normal(
                        0.0, obs_noise * kappa_scale, size=kappa_hat.shape)
                    kappa_hat = np.clip(kappa_hat, 0.0, None)
                score = v[idx] - lam * delta * kappa_hat * (T - pool.t)
            else:
                raise ValueError(rule)
            score_full = np.full(n, -np.inf)
            score_full[idx] = score
            if tie_break == "same":
                actions = [int(idx[np.argmax(score)])] * m
            else:
                order = idx[np.argsort(score)[::-1]]
                actions = [int(order[i % len(order)]) for i in range(m)]
            rewards, _ = pool.step(actions)
            tot += sum(rewards)
            rep += float(sum(score_full[a] for a in actions))
        totals.append(tot)
        reported_totals.append(rep)
    if return_details:
        return {"system_value": float(np.mean(totals)),
                "reported_value": float(np.mean(reported_totals)),
                "gap": float(np.mean(reported_totals) - np.mean(totals))}
    return float(np.mean(totals))


def price_of_anarchy(v, p, e, delta, T, m, rule="greedy", tie_break="same"):
    pl = planner_value_simultaneous(v, p, e, delta, T, m)
    de = decentralised_value_simultaneous(v, p, e, delta, T, m, rule=rule,
                                          tie_break=tie_break)
    return pl / max(de, 1e-9), pl, de


class DegradingPool:
    """Generalises `SimultaneousPool`'s one-shot, permanent destruction into
    continuous, partially-recoverable degradation -- "degrading action
    sets": actions no longer disappear, they become less valuable, and that
    devaluation itself can fade. See FINDINGS.md#24 for why this exists:
    real antimicrobial resistance accumulates, decays, and only asymptotes
    toward permanent loss, it does not binary-flip the way a destroyed arm
    does in the base model.

    Each arm carries an accrued-externality state `s_a in [0, e_a]` instead
    of a binary alive/dead flag:

        s_a(t+1) = rho * s_a(t) + (1 - rho) * e_a * hit_a(t)

    where `hit_a(t)` is 1 if arm a was pulled this round and the p_a-coin
    came up destructive, else 0. `rho` is the fraction of accrued harm that
    *survives* into the next round:

        rho -> 1   nearly all accrued harm persists -- state creeps toward
                   e_a and, once there, the arm is priced at its full,
                   permanent externality every round it is hit again
        rho = 0    accrued harm fully clears every round -- an arm hit this
                   round is exactly as good as new next round
        0 < rho < 1  the genuinely new regime: harm persists but fades, at
                   a rate set by rho

    Note on direction: an earlier informal write-up of this extension
    labelled rho=0 as "permanent consumption" and rho->1 as "recoverable".
    That has the persistence direction backwards -- rho is a survival/decay
    coefficient on *accrued* state, so it is rho->1, not rho->0, under
    which harm persists longest. Recorded here so the correction is
    visible, not silently made.

    Important: rho->1 does *not* recover `SimultaneousPool` as a limiting
    case, even though both are then "permanent" in some sense -- checked
    empirically, not assumed (a long-horizon run at rho=0.999 produces a
    substantially *higher* system value than the permanent-destruction
    model at the same horizon, not a converging one). The two models
    formalise different notions of irreversibility: `SimultaneousPool`
    permanently *removes an arm from the choice set*, so a coordination
    failure that keeps colliding on a destroyed arm's replacement compounds;
    `DegradingPool` permanently prices an arm at its full externality once
    saturated, but the arm is always still selectable. Real antimicrobial
    resistance is closer to the second: a resistant antibiotic can still be
    prescribed, it is just costly to do so. Both are genuine models of
    irreversibility; they are not nested.

    Every arm remains selectable at every round (`alive` is always all
    True) -- degradation lowers value, it does not remove the option, which
    is the actual behavioural difference from `SimultaneousPool` this class
    exists to study.

    There is no exact planner here (unlike `planner_value_simultaneous`):
    the state is continuous, not a finite alive/dead subset, so the
    dynamic-programming construction that computes the binary-model's exact
    optimum does not carry over without discretising the state space, which
    is not attempted. Comparisons below are therefore decentralised-only
    (greedy vs eci vs lambda_weighted), not against a planner ceiling --
    stated as a real limitation, not worked around.
    """

    def __init__(self, v, p, e, rho=0.0, delta=0.12, horizon=5, n_agents=2, seed=0):
        self.v = np.asarray(v, float)
        self.p = np.asarray(p, float)
        self.e = np.asarray(e, float)
        self.n = len(self.v)
        self.rho = float(rho)
        self.delta, self.T, self.m = float(delta), int(horizon), int(n_agents)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = np.zeros(self.n)
        self.t = 0
        return self.state.copy()

    @property
    def alive(self):
        return np.ones(self.n, dtype=bool)

    @property
    def burden(self):
        return self.delta * float(self.state.sum())

    def step(self, actions):
        """All agents act at once. Returns (per-agent rewards, hit set)."""
        b = self.burden
        rewards = [float(self.v[a] - b) for a in actions]
        hit = np.zeros(self.n)
        for a in actions:                      # each pull is its own trial
            if self.rng.random() < self.p[a]:
                hit[a] = 1.0
        self.state = self.rho * self.state + (1.0 - self.rho) * self.e * hit
        self.t += 1
        return rewards, set(np.flatnonzero(hit))


def decentralised_value_degrading(v, p, e, rho, delta, T, m, rule="greedy",
                                  seeds=600, tie_break="same",
                                  lam=0.0, obs_noise=0.0,
                                  return_details=False):
    """`decentralised_value_simultaneous`'s scoring rules (greedy/eci/
    lambda_weighted), run against `DegradingPool` instead of
    `SimultaneousPool` -- the partial-recovery question: does the
    regret-harm divergence survive when consumption is no longer permanent?
    """
    n = len(v)
    kappa = p * e
    kappa_scale = float(kappa.std()) if kappa.std() > 0 else 1.0
    totals, reported_totals = [], []
    for s in range(seeds):
        pool = DegradingPool(v, p, e, rho, delta, T, m, seed=9000 + s)
        rng = np.random.default_rng(80_000 + s)
        tot, rep = 0.0, 0.0
        for _ in range(T):
            idx = np.arange(n)         # every arm always selectable
            if rule == "greedy":
                score = v[idx]
            elif rule == "eci":
                score = v[idx] - delta * p[idx] * e[idx] * (T - pool.t)
            elif rule == "lambda_weighted":
                kappa_hat = p[idx] * e[idx]
                if obs_noise > 0:
                    kappa_hat = kappa_hat + rng.normal(
                        0.0, obs_noise * kappa_scale, size=kappa_hat.shape)
                    kappa_hat = np.clip(kappa_hat, 0.0, None)
                score = v[idx] - lam * delta * kappa_hat * (T - pool.t)
            else:
                raise ValueError(rule)
            if tie_break == "same":
                actions = [int(idx[np.argmax(score)])] * m
            else:
                order = idx[np.argsort(score)[::-1]]
                actions = [int(order[i % len(order)]) for i in range(m)]
            rewards, _ = pool.step(actions)
            tot += sum(rewards)
            rep += float(sum(score[np.flatnonzero(idx == a)[0]] for a in actions))
        totals.append(tot)
        reported_totals.append(rep)
    if return_details:
        return {"system_value": float(np.mean(totals)),
                "reported_value": float(np.mean(reported_totals)),
                "gap": float(np.mean(reported_totals) - np.mean(totals))}
    return float(np.mean(totals))
