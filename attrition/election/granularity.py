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
"""
import numpy as np

from .agents import simulate_level1_contact
from .census import census_rollup_for_seat
from .population import build_constituency

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
                                        n_voters=2000, seed=0):
    """Level 1 contact simulation, split across this real seat's real
    constituent districts by their real 2011 population share.

    Returns `(per_district, combined_final_fraction)`:
      per_district   list of {state, district, n_voters,
                     final_switched_fraction} -- `district` is `None` and
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

    rollup = census_rollup_for_seat(constituency.state, constituency.name)
    if not rollup:
        _, final = simulate_level1_contact(
            constituency, intensity_sequence, rho=rho, n_voters=n_voters, seed=seed)
        return ([{"state": constituency.state, "district": None,
                 "n_voters": n_voters, "final_switched_fraction": final}],
               final)

    districts = rollup["constituent_districts"]
    weights = [d["population_total"] for d in districts]
    counts = _allocate_integers(n_voters, weights)

    per_district = []
    weighted_sum = 0.0
    for i, (d, count) in enumerate(zip(districts, counts)):
        if count == 0:
            continue
        _, final = simulate_level1_contact(
            constituency, intensity_sequence, rho=rho, n_voters=count, seed=seed + i)
        per_district.append({"state": d["state"], "district": d["district"],
                            "n_voters": count, "final_switched_fraction": final})
        weighted_sum += final * count

    combined_final = weighted_sum / n_voters
    return per_district, combined_final
