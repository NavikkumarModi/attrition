"""Peer-visibility graphs for populations.

An `AgentGraph` doesn't change the bandit at all -- value, destruction, and
externality are still governed entirely by `ConsumableBandit`/
`SimultaneousPool`. It only changes what an agent sees before it chooses:
`simulate_population_simultaneous(..., graph=...)` restricts each agent's
`peer_history` to its graph neighbors' choices from the previous round. The
graph is an observation channel layered on top of the existing mechanism, not
a replacement for it.

Stdlib `random` only, no new dependency.
"""

import random

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
        """Erdos-Renyi: each possible edge present independently with prob p."""
        agent_ids = list(agent_ids)
        rng = random.Random(seed)
        edges = [(a, b) for i, a in enumerate(agent_ids)
                 for b in agent_ids[i + 1:] if rng.random() < p]
        return cls(agent_ids, edges)
