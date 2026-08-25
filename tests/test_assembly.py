"""Tests for attrition.election.assembly -- the real state Assembly
election rollup joined to the 543 real Lok Sabha constituencies via
DataMeet's real AC-to-PC crosswalk (FINDINGS.md#49). Offline, no network
-- reads the checked-in snapshots fetched by scripts/fetch_assembly_data.py
and scripts/fetch_assembly_boundaries.py.
"""

import pytest

from attrition.election import (ALL_PHASE1_CONSTITUENCIES, ASSEMBLY_YEAR_BY_STATE,
                                assembly_rollup_for_seat, all_assembly_rollups,
                                assembly_rollup_coverage)


def test_coverage_numbers_are_real_and_sane():
    cov = assembly_rollup_coverage()
    assert cov["n_ac_rows_total"] > 4000
    assert 0 < cov["n_ac_rows_matched_to_crosswalk"] <= cov["n_ac_rows_total"]
    assert cov["n_real_seats_total"] == 543
    assert 0 < cov["n_real_seats_with_rollup"] <= 543
    # a real, checked coverage rate, not a target -- pinned loosely so a
    # future re-fetch doesn't need constant retuning, but tight enough to
    # catch the join silently breaking (e.g. an alias table regression).
    assert cov["n_real_seats_with_rollup"] / cov["n_real_seats_total"] > 0.8


def test_assembly_year_by_state_reports_a_real_gap_from_2019():
    years = ASSEMBLY_YEAR_BY_STATE()
    assert "Karnataka" in years
    year, gap = years["Karnataka"]
    assert 1960 <= year <= 2026
    assert gap == year - 2019
    # not every state can have gap 0 -- assembly elections aren't
    # national-date, so at least some real spread must exist.
    gaps = {g for _, g in years.values()}
    assert len(gaps) > 1


def test_known_seat_rollup_has_expected_shape():
    rollup = assembly_rollup_for_seat("Gujarat", "AHMEDABAD EAST")
    assert rollup is not None
    assert rollup["n_constituent_acs_matched"] >= 1
    assert 0.0 <= rollup["most_common_party_fraction"] <= 1.0
    assert 0.0 <= rollup["woman_winner_fraction"] <= 1.0
    assert 0.0 <= rollup["incumbent_winner_fraction"] <= 1.0
    assert isinstance(rollup["assembly_years_used"], list) and rollup["assembly_years_used"]


def test_unmatched_seat_returns_none_not_a_crash():
    """Real seats with zero matched constituent ACs must report None, not
    raise or silently default to zero -- a rollup of nothing is not the
    same claim as a rollup showing 0%."""
    rollups = all_assembly_rollups()
    unmatched = [c for c in ALL_PHASE1_CONSTITUENCIES
                if (c.state, c.name) not in rollups]
    assert unmatched, "expected at least one real seat with no matched AC"
    state, name = unmatched[0].state, unmatched[0].name
    assert assembly_rollup_for_seat(state, name) is None


def test_party_diversity_is_at_least_one_when_rollup_exists():
    rollups = all_assembly_rollups()
    for seat, rollup in list(rollups.items())[:50]:
        assert rollup["party_diversity"] >= 1
        assert rollup["n_constituent_acs_matched"] >= rollup["party_diversity"]


def test_all_assembly_rollups_matches_coverage_count():
    assert len(all_assembly_rollups()) == assembly_rollup_coverage()["n_real_seats_with_rollup"]
