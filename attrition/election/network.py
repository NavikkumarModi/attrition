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
from .agents import BehaviouralVoter, REALISTIC_PERSUASION_CEILING_PCT
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

    Rescaled by the same `REALISTIC_PERSUASION_CEILING_PCT` (agents.py) the
    non-network path uses -- added in FINDINGS.md#47 after wiring this
    into the API surfaced raw switched fractions above 30%, an order of
    magnitude past the real field-experiment ceiling `simulate_level1_
    contact` already respects. This module predates that calibration
    (FINDINGS.md#41) and was never updated to match; fixed at the
    mechanism, not papered over in the API layer that found it.
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
        # Peer conversion is tracked separately from direct-contact
        # pressure, not folded into the same boolean: direct contact is
        # deterministic threshold-crossing (agents.py's fix -- reverts
        # cleanly if pressure decays), while a peer conversion represents
        # a lasting belief change and should not silently un-convert just
        # because the voter's own pressure later dips. `switched` each
        # round is the OR of both.
        peer_converted = set()
        fractions = []
        for intensity in intensity_sequence:
            for v in voters.values():
                v.pressure = rho * v.pressure + (1.0 - rho) * float(intensity)
                v.switched = v.pressure > v.switch_resistance
            if with_propagation:
                just_switched = [vid for vid, v in voters.items()
                                 if v.switched or vid in peer_converted]
                for vid in just_switched:
                    for neighbor in graph.neighbors(vid):
                        if neighbor not in peer_converted and \
                           not voters[neighbor].switched and \
                           rng_local.random() < peer_conversion_prob:
                            peer_converted.add(neighbor)
            for vid in peer_converted:
                voters[vid].switched = True
            raw_fraction = sum(v.switched for v in voters.values()) / n_voters
            fractions.append(raw_fraction * REALISTIC_PERSUASION_CEILING_PCT / 100.0)
        return fractions

    with_network = _run(with_propagation=True)
    direct_only = _run(with_propagation=False)
    return with_network, with_network[-1], direct_only[-1]
