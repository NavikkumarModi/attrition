"""Voter-to-voter social network: word-of-mouth propagation through the
synthetic population itself, distinct from `attrition.election`'s campaign-
to-campaign peer visibility (`examples/15_campaign_peer_influence.py`,
which is about whether campaigns herd toward each other's choices, not
about voters influencing voters). Conflating the two would overclaim what
either shows -- see `examples/16_voter_social_network.py`'s docstring.

Reuses `attrition.network.AgentGraph` (already proven for campaign/
prescriber peer visibility) as the graph structure, and `agents.py`'s
`BehaviouralVoter` (Level 1) as the per-voter state -- this module adds
only the propagation step: a voter who switches this round has a chance of
converting each of its graph neighbors too, independent of whether those
neighbors were directly contacted by a campaign this round. That is a
distinct causal channel from direct contact (word-of-mouth vs. campaign
messaging), reported separately, not folded into a single number.

Scale, stated plainly: graphs here are built over a few hundred voters, a
demonstration scale for exercising the propagation mechanism, not an
attempt at the population engine's "millions" Level-0 aspiration -- see
FINDINGS.md's election-track sections.
"""

import numpy as np

from ..network import AgentGraph
from .agents import BehaviouralVoter
from .persuasion import persuadable_fraction_from_margin
from .population import build_constituency

__all__ = ["simulate_level1_contact_with_network"]


def simulate_level1_contact_with_network(constituency, intensity_sequence,
                                         rho=0.7, n_voters=500, graph=None,
                                         peer_conversion_prob=0.05, seed=0):
    """Level 1 (`agents.simulate_level1_contact`)'s direct-contact
    mechanism, plus one additional propagation step per round: each voter
    who switched this round has probability `peer_conversion_prob` of
    converting each of its graph neighbors, independent of whether that
    neighbor was directly contacted.

    `graph` defaults to `AgentGraph.random(voter_ids, p=0.02, seed=seed)`
    if not supplied -- a sparse Erdos-Renyi graph, not a claim about real
    social-network topology.

    Returns (switched_fraction_by_round, final_switched_fraction,
    direct_only_final) -- the last value re-runs the same voters and same
    contact sequence with propagation disabled, so the network's marginal
    contribution is directly readable, not left to be inferred by
    comparing two separate calls with different RNG draws.
    """
    if isinstance(constituency, str):
        constituency = build_constituency(constituency)
    persuadable = persuadable_fraction_from_margin(constituency)
    mean_resistance = 1.0 - persuadable
    rng = np.random.default_rng(seed)
    conc = 8.0
    a = max(mean_resistance * conc, 0.5)
    b = max((1.0 - mean_resistance) * conc, 0.5)

    voter_ids = [f"v{i}" for i in range(n_voters)]
    if graph is None:
        graph = AgentGraph.random(voter_ids, p=0.02, seed=seed)

    def _run(with_propagation):
        rng_local = np.random.default_rng(seed)
        resistances = rng_local.beta(a, b, size=n_voters)
        voters = {vid: BehaviouralVoter(r)
                 for vid, r in zip(voter_ids, resistances)}
        fractions = []
        for intensity in intensity_sequence:
            for vid, v in voters.items():
                v.pressure = rho * v.pressure + (1.0 - rho) * float(intensity)
                if rng_local.random() < v.pressure * (1.0 - v.switch_resistance):
                    v.switched = True
                elif rng_local.random() < 0.02:
                    v.switched = False
            if with_propagation:
                just_switched = [vid for vid, v in voters.items() if v.switched]
                for vid in just_switched:
                    for neighbor in graph.neighbors(vid):
                        if not voters[neighbor].switched and \
                           rng_local.random() < peer_conversion_prob:
                            voters[neighbor].switched = True
            fractions.append(sum(v.switched for v in voters.values()) / n_voters)
        return fractions

    with_network = _run(with_propagation=True)
    direct_only = _run(with_propagation=False)
    return with_network, with_network[-1], direct_only[-1]
