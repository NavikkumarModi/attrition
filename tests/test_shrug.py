"""Tests for attrition.election.shrug -- real SHRUG socioeconomic
indicators (demographic, economic, wealth-proxy, infrastructure) rolled
up to the 543 real Lok Sabha constituencies via the real DataMeet AC-to-PC
crosswalk, restricted to post-2008-delimitation rows (FINDINGS.md#57).
Offline, no network -- reads the checked-in snapshot produced by
scripts/process_shrug_data.py from real, manually-downloaded SHRUG CSVs.
"""

import pytest

from attrition.election import (ALL_PHASE1_CONSTITUENCIES, shrug_rollup_for_seat,
                                all_shrug_rollups, shrug_rollup_coverage)


def test_coverage_numbers_are_real_and_sane():
    cov = shrug_rollup_coverage()
    assert cov["n_acs_total"] == 4120  # real 2008-delimitation AC count in this snapshot
    assert cov["n_real_seats_total"] == 543
    assert 0 < cov["n_real_seats_with_rollup"] <= 543
    # stricter than assembly.py/census.py's joins (post-delimitation only,
    # see module docstring) -- still expected to clear a real majority.
    assert cov["n_real_seats_with_rollup"] / cov["n_real_seats_total"] > 0.7


def test_known_seat_rollup_has_expected_shape():
    rollup = shrug_rollup_for_seat("Gujarat", "AHMEDABAD EAST")
    assert rollup is not None
    assert rollup["n_constituent_acs_matched"] >= 1
    assert len(rollup["constituent_acs"]) == rollup["n_constituent_acs_matched"]
    for ac in rollup["constituent_acs"]:
        assert ac["ac08_id"].startswith("2008-")
        assert ac["state"]
        assert ac["ac_name"]


def test_unmatched_seat_returns_none_not_a_crash():
    rollups = all_shrug_rollups()
    unmatched = [c for c in ALL_PHASE1_CONSTITUENCIES
                if (c.state, c.name) not in rollups]
    assert unmatched, "expected at least one real seat with no matched AC"
    state, name = unmatched[0].state, unmatched[0].name
    assert shrug_rollup_for_seat(state, name) is None


def test_all_shrug_rollups_matches_coverage_count():
    assert len(all_shrug_rollups()) == shrug_rollup_coverage()["n_real_seats_with_rollup"]


def test_population_total_is_a_real_sum_not_a_mean():
    """Population counts are real, disjoint per-AC figures -- summed
    across a seat's constituent ACs, unlike census.py's district
    densities (which are means, since districts overlap seats). A
    multi-AC seat's total must be at least as large as its largest single
    constituent AC's own population, never smaller."""
    rollups = all_shrug_rollups()
    multi_ac = [r for r in rollups.values()
               if r and r["n_constituent_acs_matched"] > 1
               and r["population_total"] is not None]
    assert multi_ac, "expected at least one real multi-AC seat with population data"
    for rollup in multi_ac[:20]:
        ac_pops = [ac["population_total"] for ac in rollup["constituent_acs"]
                  if ac.get("population_total") is not None]
        if not ac_pops:
            continue
        assert rollup["population_total"] >= max(ac_pops) - 1e-6


def test_poverty_rate_is_population_weighted_mean_within_real_bounds():
    """A regression check against a disguised sum: any real poverty rate
    the rollup reports must fall within [0, 1] and within the real
    min/max of its constituent ACs' own rural poverty rates."""
    rollups = all_shrug_rollups()
    checked = 0
    for rollup in rollups.values():
        if not rollup or rollup["secc_pov_rate_rural"] is None:
            continue
        rates = [ac["secc_pov_rate_rural"] for ac in rollup["constituent_acs"]
                if ac.get("secc_pov_rate_rural") is not None]
        if not rates:
            continue
        assert 0.0 <= rollup["secc_pov_rate_rural"] <= 1.0
        assert min(rates) - 1e-9 <= rollup["secc_pov_rate_rural"] <= max(rates) + 1e-9
        checked += 1
        if checked >= 20:
            break
    assert checked > 0, "expected at least one real seat with rural SECC data"


def test_rural_and_urban_secc_are_never_conflated():
    """Module docstring's central honesty claim: a constituency with only
    rural (or only urban) SECC data must not have the missing segment
    silently filled from the other."""
    rollups = all_shrug_rollups()
    rural_only = [r for r in rollups.values()
                 if r and r["secc_cons_pc_rural"] is not None
                 and r["secc_cons_pc_urban"] is None]
    assert rural_only, "expected at least one real seat with rural-only SECC data"


def test_pmgsy_connectivity_is_a_real_fraction():
    rollups = all_shrug_rollups()
    checked = 0
    for rollup in rollups.values():
        if not rollup or rollup["pmgsy_pct_villages_connected"] is None:
            continue
        assert 0.0 <= rollup["pmgsy_pct_villages_connected"] <= 1.0
        checked += 1
        if checked >= 20:
            break
    assert checked > 0, "expected at least one real seat with PMGSY data"


def test_pre_delimitation_crosswalk_rows_are_excluded():
    """Regression guard for the module's real, checked design choice: a
    crosswalk row explicitly flagged 'Pre delimitation' must never
    contribute a matched AC, since SHRUG's con08 data assumes 2008
    delimitation boundaries specifically."""
    from attrition.election.shrug import _post_delimitation_crosswalk
    rows = _post_delimitation_crosswalk()
    assert all(r.get("status") != "Pre delimitation" for r in rows)
    assert len(rows) < 4182  # the real, full crosswalk row count
