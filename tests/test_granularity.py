"""Tests for attrition.election.granularity -- splitting a seat's Level 1
population across its real constituent districts (FINDINGS.md#51).
Offline, no network.
"""

import pytest

from attrition.election import (build_constituency, simulate_level1_contact_by_district,
                                census_rollup_for_seat, ALL_PHASE1_CONSTITUENCIES)


def test_district_split_partitions_n_voters_exactly():
    c = build_constituency("AHMEDABAD EAST", state="Gujarat")
    per_district, combined = simulate_level1_contact_by_district(
        c, [0.15, 0.2, 0.25, 0.3, 0.35], n_voters=3000, seed=0)
    assert sum(d["n_voters"] for d in per_district) == 3000
    assert len(per_district) == 2  # Ahmedabad East's two real constituent districts


def test_district_split_weights_match_real_population_shares():
    """Regression check for the largest-remainder allocator: district
    voter shares must track real population shares, not be equal splits
    or arbitrary."""
    c = build_constituency("AHMEDABAD EAST", state="Gujarat")
    rollup = census_rollup_for_seat("Gujarat", "AHMEDABAD EAST")
    pops = {d["district"]: d["population_total"] for d in rollup["constituent_districts"]}
    total_pop = sum(pops.values())

    per_district, _ = simulate_level1_contact_by_district(
        c, [0.2], n_voters=10000, seed=0)
    for d in per_district:
        expected_share = pops[d["district"]] / total_pop
        actual_share = d["n_voters"] / 10000
        assert actual_share == pytest.approx(expected_share, abs=0.01)


def test_combined_fraction_is_the_exact_population_weighted_mean():
    c = build_constituency("AHMEDABAD EAST", state="Gujarat")
    per_district, combined = simulate_level1_contact_by_district(
        c, [0.15, 0.2, 0.25, 0.3, 0.35], n_voters=3000, seed=0)
    n = sum(d["n_voters"] for d in per_district)
    expected = sum(d["final_switched_fraction"] * d["n_voters"] for d in per_district) / n
    assert combined == pytest.approx(expected)


def test_falls_back_to_one_undifferentiated_pool_with_no_rollup():
    unmatched = next(c for c in ALL_PHASE1_CONSTITUENCIES
                     if census_rollup_for_seat(c.state, c.name) is None)
    per_district, combined = simulate_level1_contact_by_district(
        unmatched, [0.2, 0.3], n_voters=1000, seed=0)
    assert len(per_district) == 1
    assert per_district[0]["district"] is None
    assert per_district[0]["n_voters"] == 1000
    assert combined == per_district[0]["final_switched_fraction"]


def test_district_split_reproducible_given_same_seed():
    c = build_constituency("AHMEDABAD EAST", state="Gujarat")
    r1 = simulate_level1_contact_by_district(c, [0.2, 0.3], n_voters=2000, seed=5)
    r2 = simulate_level1_contact_by_district(c, [0.2, 0.3], n_voters=2000, seed=5)
    assert r1 == r2
