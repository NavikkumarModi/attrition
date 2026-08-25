"""Peer-visibility graphs for populations.

An `AgentGraph` doesn't change the bandit at all -- value, destruction, and
externality are still governed entirely by `ConsumableBandit`/
`SimultaneousPool`. It only changes what an agent sees before it chooses:
`simulate_population_simultaneous(..., graph=...)` restricts each agent's
`peer_history` to its graph neighbors' choices from the previous round. The
graph is an observation channel layered on top of the existing mechanism, not
a replacement for it.

Stdlib `random` for every small/exact construction method (`complete`,
`ring`, `star`, `random`). `random_by_degree` (FINDINGS.md#52 -- the
large-population-scale generator) uses numpy, already a core dependency
of this package, not a new one.
"""

import random

import numpy as np

__all__ = ["AgentGraph"]


class AgentGraph:
    """Plain undirected adjacency structure over agent ids."""

    def __init__(self, agent_ids, edges=()):
        self.agent_ids = list(agent_ids)
        self._adj = {a: set() for a in self.agent_ids}
        for a, b in edges:
            self.add_edge(a, b)

    def add_edge(self, a, b):
        self._adj[a].add(b)
        self._adj[b].add(a)

    def neighbors(self, agent_id):
        return sorted(self._adj.get(agent_id, ()))

    @classmethod
    def complete(cls, agent_ids):
        """Every agent sees every other agent."""
        agent_ids = list(agent_ids)
        edges = [(a, b) for i, a in enumerate(agent_ids)
                 for b in agent_ids[i + 1:]]
        return cls(agent_ids, edges)

    @classmethod
    def ring(cls, agent_ids):
        """Each agent sees its two immediate neighbors in a cycle."""
        agent_ids = list(agent_ids)
        n = len(agent_ids)
        edges = [(agent_ids[i], agent_ids[(i + 1) % n]) for i in range(n)] \
            if n > 1 else []
        return cls(agent_ids, edges)

    @classmethod
    def star(cls, agent_ids, hub=None):
        """One hub agent (e.g. an influential prescriber) sees and is seen by
        everyone; everyone else sees only the hub.
        """
        agent_ids = list(agent_ids)
        hub = agent_ids[0] if hub is None else hub
        edges = [(hub, a) for a in agent_ids if a != hub]
        return cls(agent_ids, edges)

    @classmethod
    def random(cls, agent_ids, p=0.3, seed=0):
        """Erdos-Renyi: each possible edge present independently with prob p.

        O(n^2) by construction -- it tests every possible pair -- which is
        fine at the population sizes this method was built for (a few
        thousand) but becomes both too slow AND the wrong model at real
        large-population scale: a FIXED edge probability `p` implies a
        MEAN DEGREE that grows linearly with population size (mean degree
        = p*(n-1)), so holding `p` fixed while raising `n` from thousands
        to millions does not produce "the same kind of sparse social
        network, bigger" -- it produces a graph with billions of edges
        (checked, not assumed: n=1,000,000, p=0.02 implies ~10 billion
        edges, an absurd social-network claim, not a performance problem
        to optimize around). Use `random_by_degree` instead once
        population size gets into the tens of thousands or more -- it
        holds mean degree fixed instead of edge probability, which is
        both the real-social-network-appropriate parameterisation and the
        only one that stays tractable at scale."""
        agent_ids = list(agent_ids)
        rng = random.Random(seed)
        edges = [(a, b) for i, a in enumerate(agent_ids)
                 for b in agent_ids[i + 1:] if rng.random() < p]
        return cls(agent_ids, edges)

    @classmethod
    def random_by_degree(cls, agent_ids, mean_degree=10.0, seed=0):
        """A sparse random graph parameterised by MEAN DEGREE (expected
        number of neighbors per agent) instead of edge probability -- the
        real-social-network-appropriate parameterisation (people's real
        social-connection counts don't grow with population size the way
        `random`'s fixed edge probability implies) and the one that stays
        tractable at real large-population scale.

        Vectorised (numpy): draws `round(n * mean_degree / 2)` edges
        directly as random index pairs rather than testing every one of
        the `n*(n-1)/2` possible pairs `random` does -- O(n * mean_degree)
        instead of O(n^2). At mean_degree=10, n=1,000,000 that's ~5M edge
        draws (a fraction of a second), not the ~5*10^11 pair-tests
        `random` would need to do the same job (and would produce a wildly
        different, far denser graph doing it, per that method's own
        docstring).

        Self-loops are dropped, not resampled -- a real, small (expected
        ~mean_degree/n of drawn pairs) undercount of the target mean
        degree, stated rather than silently masked by a resampling loop
        that would reintroduce the O(n^2) worst case this method exists to
        avoid.
        """
        agent_ids = list(agent_ids)
        n = len(agent_ids)
        if n < 2:
            return cls(agent_ids, ())
        rng = np.random.default_rng(seed)
        n_edges = max(0, round(n * float(mean_degree) / 2.0))
        a_idx = rng.integers(0, n, size=n_edges)
        b_idx = rng.integers(0, n, size=n_edges)
        edges = [(agent_ids[i], agent_ids[j]) for i, j in zip(a_idx, b_idx) if i != j]
        return cls(agent_ids, edges)
