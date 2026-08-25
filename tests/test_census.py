"""Tests for attrition.election.census -- real Census 2011 district
demographics rolled up to the 543 real Lok Sabha constituencies via the
real DataMeet AC-to-PC crosswalk's district field (FINDINGS.md#50).
Offline, no network -- reads the checked-in snapshot produced by
scripts/process_census_data.py from a manually-downloaded real Census
2011 table.
"""

import pytest

from attrition.election import (ALL_PHASE1_CONSTITUENCIES, census_rollup_for_seat,
                                all_census_rollups, census_rollup_coverage)


def test_coverage_numbers_are_real_and_sane():
    cov = census_rollup_coverage()
    assert cov["n_districts_total"] == 640  # real 2011 district count
    assert cov["n_real_seats_total"] == 543
    assert 0 < cov["n_real_seats_with_rollup"] <= 543
    assert cov["n_real_seats_with_rollup"] / cov["n_real_seats_total"] > 0.75


def test_known_seat_rollup_has_expected_shape():
    rollup = census_rollup_for_seat("Gujarat", "AHMEDABAD EAST")
    assert rollup is not None
    assert rollup["n_constituent_districts"] >= 1
    assert len(rollup["constituent_districts"]) == rollup["n_constituent_districts"]
    for d in rollup["constituent_districts"]:
        assert d["population_total"] > 0
        assert d["population_male"] + d["population_female"] == d["population_total"]
        assert d["area_sq_km"] > 0
    assert rollup["mean_population_density_per_sq_km"] > 0


def test_unmatched_seat_returns_none_not_a_crash():
    rollups = all_census_rollups()
    unmatched = [c for c in ALL_PHASE1_CONSTITUENCIES
                if (c.state, c.name) not in rollups]
    assert unmatched, "expected at least one real seat with no matched district"
    state, name = unmatched[0].state, unmatched[0].name
    assert census_rollup_for_seat(state, name) is None


def test_mean_density_is_never_a_disguised_sum():
    """Regression check for the module's central honesty claim: the mean
    density across constituent districts must fall within the real
    min/max of those districts' own densities, never exceed the max (a
    sum masquerading as a mean would blow past it for any seat with more
    than one constituent district)."""
    rollups = all_census_rollups()
    multi_district = [r for r in rollups.values() if r and r["n_constituent_districts"] > 1]
    assert multi_district, "expected at least one real seat spanning multiple districts"
    for rollup in multi_district[:20]:
        densities = [d["population_density_per_sq_km"] for d in rollup["constituent_districts"]]
        assert min(densities) <= rollup["mean_population_density_per_sq_km"] <= max(densities)


def test_all_census_rollups_matches_coverage_count():
    assert len(all_census_rollups()) == census_rollup_coverage()["n_real_seats_with_rollup"]


def test_total_district_population_matches_real_2011_census_figure():
    """The real, published 2011 Census total population figure
    (1,210,854,977) as an end-to-end sanity check that
    process_census_data.py extracted the right rows, not a subset or a
    double-counted one."""
    from attrition.election.census import _load_districts
    total = sum(d["population_total"] for d in _load_districts())
    assert total == 1_210_854_977
