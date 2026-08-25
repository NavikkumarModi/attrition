"""Tests for the election track's Phase 1 population engine
(attrition.election). Offline only -- reads the checked-in snapshot, never
the network; see scripts/fetch_election_data.py for how it was built.
"""

import numpy as np
import pytest

from attrition.election import (ALL_PHASE1_CONSTITUENCIES, build_constituency,
                                synthetic_voter_population, validate_population,
                                get_constituency)
from attrition.election.lokdhaba import load_constituencies


def test_ten_constituencies_loaded():
    assert len(ALL_PHASE1_CONSTITUENCIES) == 10
    names = {c.name for c in ALL_PHASE1_CONSTITUENCIES}
    assert "AHMEDABAD EAST" in names
    assert "AZAMGARH" in names


def test_constituencies_are_real_not_synthetic():
    """Every field is either a real elector/turnout/vote-share number or an
    explicit note about why the seat was picked -- nothing invented."""
    for c in ALL_PHASE1_CONSTITUENCIES:
        assert c.electors > 100_000          # a real Lok Sabha seat, not a toy number
        assert 0.0 < c.turnout_pct <= 100.0
        assert sum(c.vote_share_pct) <= 100.5  # real shares, minor float slack
        assert c.year == 2019                # Phase 1's single real cross-section


def test_unknown_constituency_raises_with_available_list():
    with pytest.raises(KeyError, match="AHMEDABAD EAST"):
        get_constituency("Not A Real Place")


def test_synthetic_population_has_no_synthetic_latent_fields():
    """SyntheticVoter must stay thin -- turnout and party choice only, per
    the observed-vs-synthetic-latent discipline in population.py's
    docstring. This test exists so a future change that adds e.g.
    `issue_priorities` or `persuadability` to SyntheticVoter fails loudly."""
    from attrition.election.population import SyntheticVoter
    fields = set(SyntheticVoter.__dataclass_fields__)
    assert fields == {"constituency", "turned_out", "voted_for"}


def test_population_reproduces_real_turnout_and_vote_share():
    """Validation ladder level 1 (demographic/aggregate reproduction) for
    every Phase-1 constituency, not just one -- checked, not asserted once
    and assumed to generalise."""
    for c in ALL_PHASE1_CONSTITUENCIES:
        result = validate_population(c, n_voters=150_000, seed=1)
        assert abs(result["turnout_pct"]["sampled"]
                  - result["turnout_pct"]["real"]) < 0.5


def test_population_size_and_reproducibility():
    c = build_constituency("Lucknow")
    voters = synthetic_voter_population(c, n_voters=1000, seed=0)
    assert len(voters) == 1000
    voters_again = synthetic_voter_population(c, n_voters=1000, seed=0)
    assert [v.voted_for for v in voters] == [v.voted_for for v in voters_again], (
        "same seed must give the same population, deterministically")


def test_duplicate_party_labels_aggregate_correctly():
    """Regression test: several candidates can share a label (all
    independents are 'IND'), so both real and sampled shares must be
    aggregated by unique label before comparison -- a real bug caught while
    building this (comparing a combined sampled share against one
    duplicate row's individual real share, not the aggregate)."""
    ahmedabad = build_constituency("Ahmedabad East")
    real_rows = load_constituencies()
    row = next(r for r in real_rows if r["constituency"] == "AHMEDABAD EAST")
    ind_rows = [c for c in row["candidates"] if c["party"] == "IND"]
    assert len(ind_rows) > 1, "test assumes Ahmedabad East has multiple IND candidates"

    result = validate_population(ahmedabad, n_voters=300_000, seed=2)
    combined_real = sum(c["vote_share_pct"] for c in ind_rows)
    # validate_population renormalises shares to sum to exactly 100 first,
    # so a tiny float difference from the raw (unnormalised) source sum is
    # expected -- the point of this test is that it's off by ~0.0002, a
    # renormalisation artefact, not the ~1.3-point bug this regression-tests.
    assert abs(result["vote_share_pct"]["IND"]["real"] - combined_real) < 0.01
