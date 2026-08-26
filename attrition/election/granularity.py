"""Population granularity: split a seat's synthetic Level 1 population
across its REAL constituent districts (`census.py`'s real 2011 population
weights) instead of running one undifferentiated pool for the whole seat.

Real limitation, stated up front rather than left for a reader to
discover: sub-populations here differ only in SIZE (each district's real
share of the seat's constituent-district population), not in
persuadability. There is no real per-district margin or incumbency data
to derive a differentiated resistance distribution from -- every
sub-population draws from the same seat-wide Beta distribution
`agents.simulate_level1_contact` already uses
(`mean_resistance_for(persuadable_fraction_from_margin(constituency),
constituency.candidate_incumbent[0])`, a real, seat-level anchor, not a
district one). This adds spatial resolution to the REPORTING -- a
per-district switched fraction, not a new causal claim that voters in
different constituent districts are differently persuadable.

Falls back cleanly to the existing undifferentiated `simulate_level1_
contact` for the ~18% of real seats `census.py` has no constituent-
district data for (see that module's docstring for the real, checked
coverage number) -- reported as `district: None`, one "district" covering
the whole requested population, never silently dropped.

Optionally combines with voter-to-voter network propagation
(`use_network=True`, FINDINGS.md#53) -- closing the gap `election_api.py`
previously rejected with a 400 (`use_network` and `use_district_
granularity` "cannot both be set"). A real, explicit scoping choice
stated here rather than left implicit: propagation is LOCAL to each
district -- each sub-population gets its OWN independent random graph
(`attrition.network.AgentGraph.random_by_degree`), with no edges
between districts. This is not asserted to be a complete model of real
word-of-mouth (a voter's real social ties are not neatly bounded by
district lines), but it is the honest, defensible default absent any
real cross-district social-tie data to build a joined graph from --
the same "real data or an explicit, stated modelling choice, never a
silent guess" discipline this module already uses for population size.
"""
import numpy as np

from .agents import simulate_level1_contact
from .census import census_rollup_for_seat
from .network import simulate_level1_contact_with_network
from .population import build_constituency
from ..network import AgentGraph

__all__ = ["simulate_level1_contact_by_district"]


def _allocate_integers(total, weights):
    """Split `total` into `len(weights)` non-negative integers summing
    EXACTLY to `total`, proportional to `weights` -- the largest-
    remainder method (Hamilton apportionment), the same well-known,
    exact-sum-preserving technique real electoral seat-apportionment uses,
    not a rounding scheme that could silently over/under-allocate voters.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    raw = weights * total
    base = np.floor(raw).astype(int)
    remainder = total - base.sum()
    if remainder > 0:
        order = np.argsort(-(raw - base))
        for i in order[:remainder]:
            base[i] += 1
    return base.tolist()


def simulate_level1_contact_by_district(constituency, intensity_sequence, rho=0.7,
                                        n_voters=2000, seed=0, use_network=False,
                                        peer_conversion_prob=0.05, mean_degree=10.0):
    """Level 1 contact simulation, split across this real seat's real
    constituent districts by their real 2011 population share, optionally
    with voter-to-voter network propagation run independently within each
    district (`use_network=True` -- see module docstring for why
    propagation is scoped per-district, not seat-wide).

    Returns `(per_district, combined_final_fraction)`:
      per_district   list of {state, district, n_voters,
                     final_switched_fraction, [direct_only_final_switched_
                     fraction if use_network]} -- `district` is `None` and
                     the list has one entry (covering all `n_voters`) if
                     `census.py` has no constituent-district data for
                     this seat.
      combined_final_fraction   the EXACT population-weighted combination
                     of every sub-population's final switched fraction --
                     exact, not approximate, because each sub-population
                     is disjoint by construction (they partition the
                     requested `n_voters`, unlike `census.py`'s district
                     rollup, whose districts are NOT disjoint across
                     different seats and so cannot be summed this way).
    """
    if isinstance(constituency, str):
        constituency = build_constituency(constituency)

    def _run(count, sub_seed):
        if not use_network:
            _, final = simulate_level1_contact(
                constituency, intensity_sequence, rho=rho, n_voters=count, seed=sub_seed)
            return final, None
        voter_ids = [f"v{k}" for k in range(count)]
        graph = AgentGraph.random_by_degree(voter_ids, mean_degree=mean_degree,
                                            seed=sub_seed)
        _, final, direct_only = simulate_level1_contact_with_network(
            constituency, intensity_sequence, rho=rho, n_voters=count, graph=graph,
            peer_conversion_prob=peer_conversion_prob, seed=sub_seed)
        return final, direct_only

    rollup = census_rollup_for_seat(constituency.state, constituency.name)
    if not rollup:
        final, direct_only = _run(n_voters, seed)
        entry = {"state": constituency.state, "district": None,
                "n_voters": n_voters, "final_switched_fraction": final}
        if direct_only is not None:
            entry["direct_only_final_switched_fraction"] = direct_only
        return [entry], final

    districts = rollup["constituent_districts"]
    weights = [d["population_total"] for d in districts]
    counts = _allocate_integers(n_voters, weights)

    per_district = []
    weighted_sum = 0.0
    for i, (d, count) in enumerate(zip(districts, counts)):
        if count == 0:
            continue
        final, direct_only = _run(count, seed + i)
        entry = {"state": d["state"], "district": d["district"],
                "n_voters": count, "final_switched_fraction": final}
        if direct_only is not None:
            entry["direct_only_final_switched_fraction"] = direct_only
        per_district.append(entry)
        weighted_sum += final * count

    combined_final = weighted_sum / n_voters
    return per_district, combined_final
