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


class _CSRAdjacency:
    """Lazy compressed-sparse-row neighbor lookup (FINDINGS.md#53) --
    `random_by_degree`'s fast-path backend at large population sizes.

    Construction stores only three numpy arrays (an integer id->position
    map, `indptr`, `indices` -- the standard CSR layout) and does NOT
    build a single Python string or set up front. Converting a queried
    node's integer neighbor indices back into the agent-id strings
    `neighbors()` is contracted to return happens lazily, only for ids
    actually queried, and is memoized -- the real fix for the cost
    `AgentGraph`'s general dict-of-sets `__init__`/`add_edge` path pays
    unconditionally (building every node's neighbor SET as real Python
    objects, whether or not that node's neighbors are ever looked up).
    """

    __slots__ = ("_id_to_idx", "_idx_to_id", "_indptr", "_indices", "_cache")

    def __init__(self, agent_ids, indptr, indices):
        self._idx_to_id = agent_ids
        self._id_to_idx = {a: i for i, a in enumerate(agent_ids)}
        self._indptr = indptr
        self._indices = indices
        self._cache = {}

    def neighbors(self, agent_id):
        cached = self._cache.get(agent_id)
        if cached is not None:
            return cached
        idx = self._id_to_idx.get(agent_id)
        if idx is None:
            return []
        seg = self._indices[self._indptr[idx]:self._indptr[idx + 1]]
        result = sorted(self._idx_to_id[j] for j in seg)
        self._cache[agent_id] = result
        return result


class AgentGraph:
    """Plain undirected adjacency structure over agent ids."""

    def __init__(self, agent_ids, edges=()):
        self.agent_ids = list(agent_ids)
        self._adj = {a: set() for a in self.agent_ids}
        self._csr = None
        for a, b in edges:
            self.add_edge(a, b)

    def add_edge(self, a, b):
        self._adj[a].add(b)
        self._adj[b].add(a)
        self._csr = None  # any exact-backend graph never sets _csr anyway;
                          # defensive reset if add_edge is ever called on
                          # a _csr-backed instance (not currently possible
                          # via any public constructor, but never silently
                          # stale if that changes)

    def neighbors(self, agent_id):
        if self._csr is not None:
            return self._csr.neighbors(agent_id)
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

        Construction itself is also vectorised end to end (FINDINGS.md
        #53), not just the edge sampling: the resulting instance is
        backed by a lazy CSR adjacency (`_CSRAdjacency`) instead of
        `AgentGraph`'s general dict-of-sets `__init__`/`add_edge` path,
        which was itself the dominant cost at real scale (~4.5s of a
        ~5.5s total at n=1,000,000, checked, not assumed) -- that path
        builds every node's neighbor set as real Python objects
        unconditionally, whether or not that node is ever queried.
        Construction here is ~1s at n=1,000,000 (a numpy sort plus a
        vectorised grouping, no Python-level loop over edges at all), and
        the per-node id/set materialisation `neighbors()` still returns
        happens lazily, once per node actually queried, not once per node
        that exists.
        """
        agent_ids = list(agent_ids)
        n = len(agent_ids)
        if n < 2:
            return cls(agent_ids, ())
        rng = np.random.default_rng(seed)
        n_edges = max(0, round(n * float(mean_degree) / 2.0))
        a_idx = rng.integers(0, n, size=n_edges)
        b_idx = rng.integers(0, n, size=n_edges)
        keep = a_idx != b_idx
        a_idx, b_idx = a_idx[keep], b_idx[keep]

        # Undirected: each edge contributes an entry in both directions.
        # `kind="quicksort"` (numpy's introsort, the default) rather than
        # the stable mergesort -- neighbor ORDER within a node doesn't
        # matter here (`neighbors()` sorts its output anyway), only
        # correct grouping does, and quicksort is measured ~3.5x faster
        # for this (0.79s vs 2.8s at n=1,000,000, checked).
        src = np.concatenate([a_idx, b_idx])
        dst = np.concatenate([b_idx, a_idx])
        order = np.argsort(src, kind="quicksort")
        src_sorted = src[order]
        indices = dst[order]
        indptr = np.searchsorted(src_sorted, np.arange(n + 1))

        self = cls.__new__(cls)
        self.agent_ids = agent_ids
        self._adj = {}
        self._csr = _CSRAdjacency(agent_ids, indptr, indices)
        return self
