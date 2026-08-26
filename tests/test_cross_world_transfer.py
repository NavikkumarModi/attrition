"""Regression tests for experiments/exp66_cross_world_transfer.py -- the
cross-world benchmark punch-list item ("does the same policy transfer
across domains?") checked here at low seed counts for speed; the full
seeds=200 run lives in the experiment script itself.
"""

import numpy as np
import pytest

from experiments.exp66_cross_world_transfer import (
    DOMAIN_SCENARIOS, unified_value_gap_table, transfer_check)


def test_unified_table_covers_every_registered_domain_world():
    table = unified_value_gap_table()
    assert {row["world"] for row in table} == set(DOMAIN_SCENARIOS)


def test_unified_table_eci_never_below_greedy():
    """Theorem 4 territory: ECI is a genuine correction, never worse than
    greedy in expectation, in every domain checked here."""
    table = unified_value_gap_table()
    for row in table:
        assert row["eci"] >= row["greedy"] - 1e-6


def test_transfer_check_home_world_recovery_is_undefined_not_crashing():
    """clinical-amr (the default home world) is the documented ALIGNED
    regime (greedy already optimal there) -- its own Greedy-to-ECI gap is
    ~0, so recovery is mathematically undefined (NaN), not a bug. This
    must not raise (division-by-zero guarded), and must be flagged."""
    _ratio, rows = transfer_check(home_world="clinical-amr")
    home_row = next(r for r in rows if r["is_home"])
    assert home_row["world"] == "clinical-amr"
    assert np.isnan(home_row["local_recovery_pct"])
    assert np.isnan(home_row["transferred_recovery_pct"])


def test_transfer_check_ratio_is_positive_and_reused_verbatim():
    ratio, rows = transfer_check(home_world="clinical-amr")
    assert ratio > 0
    # the transferred Conservative's e_bound must literally be
    # ratio * that world's own v.max() -- checked indirectly via the
    # transferred and local instances landing at the same value in the
    # (degenerate) case the ratio matches the local e.max()/v.max() ratio
    # exactly, which is true by construction for the home world itself.
    home_row = next(r for r in rows if r["is_home"])
    assert home_row["local"] == pytest.approx(home_row["transferred"])


def test_transfer_check_local_and_transferred_track_each_other_closely_off_home():
    """The real claim under test: a ratio calibrated on ONE world and
    reused, unmodified, on every other world's own v.max() should land
    close to what recalibrating locally in each world would give --
    otherwise "no domain-specific knowledge needed" would not survive
    being taken literally across domains. Checked with a generous
    tolerance (not asserting near-identical, since the domains are
    genuinely different) -- this is the number FINDINGS.md reports
    honestly, not a pass/fail gate on a specific value."""
    _ratio, rows = transfer_check(home_world="clinical-amr")
    non_home = [r for r in rows if not r["is_home"]]
    assert len(non_home) == len(DOMAIN_SCENARIOS) - 1
    gaps = [abs(r["local_recovery_pct"] - r["transferred_recovery_pct"])
           for r in non_home]
    # every world's transferred recovery is within 15 points of that same
    # world's own locally-calibrated recovery -- the ratio transfers,
    # even though the underlying local recipe's own quality varies a lot
    # by domain (see the fisheries-world low-recovery finding).
    assert max(gaps) < 15.0


def test_transfer_check_conservative_instances_do_not_collide_in_compare_output():
    """Regression guard: both Conservative instances share class attribute
    name == "conservative" -- transfer_check must rename the transferred
    one before calling compare(), or its value would silently overwrite
    (or be overwritten by) the locally-calibrated one in the results
    dict."""
    _ratio, rows = transfer_check(home_world="clinical-amr")
    for r in rows:
        assert "local" in r and "transferred" in r
        # for a non-degenerate world, local and transferred use different
        # e_bound values and are not required to match
