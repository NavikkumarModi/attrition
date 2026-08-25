"""Regression test for experiments/exp65_amr_five_policy_reference_table.py
-- the "unified five-policy table" FINDINGS.md#26 recorded as open. Checks
the in-repo half is internally consistent (matches exp62's own sanity
check and its cited price-of-anarchy figure) and that the cited external
numbers are transcribed correctly from FINDINGS.md, not recomputed here
(recomputing them needs a separate Python 3.11 environment with
abx_amr_simulator installed -- out of scope for this offline suite).
"""

import pytest

from experiments.exp65_amr_five_policy_reference_table import (
    in_repo_half, EXTERNAL_SILOED, EXTERNAL_SHARED)


def test_in_repo_half_lambda1_matches_eci_exactly():
    """exp62's own sanity check, restated: lambda_weighted(lam=1, noise=0)
    must equal the eci rule bit-for-bit -- both score v - delta*kappa*(T-t)."""
    repo = in_repo_half()
    assert repo["eci"] == pytest.approx(repo["lambda_weighted(lam=1)"])


def test_in_repo_half_price_of_anarchy_matches_findings_citation():
    """FINDINGS.md#24.1 reports price of anarchy against the planner optimum
    bottoms out at 1.07 for lam=1, noise=0 -- checked here so the table
    can't silently drift from the number the surrounding prose cites."""
    repo = in_repo_half()
    poa_eci = repo["planner"] / repo["eci"]
    assert poa_eci == pytest.approx(1.07, abs=0.01)


def test_in_repo_half_planner_dominates_every_decentralised_rule():
    repo = in_repo_half()
    assert repo["planner"] >= repo["greedy"]
    assert repo["planner"] >= repo["eci"]


def test_external_numbers_are_transcribed_not_fabricated():
    """Not a recomputation -- just guards against a silent transcription
    slip from FINDINGS.md#24.2/24.4's tables into this module's constants."""
    assert EXTERNAL_SILOED["final_amr_a"] == pytest.approx(0.7461)
    assert EXTERNAL_SILOED["final_amr_b"] == pytest.approx(0.2506)
    assert EXTERNAL_SHARED["final_amr_a"] == pytest.approx(0.9691)
    assert EXTERNAL_SHARED["final_amr_b"] == pytest.approx(0.1993)
    # decentralising must move antibiotic A toward the no-stewardship level
    # (99.97%, FINDINGS.md#24.2) and antibiotic B the other way -- the
    # direction FINDINGS.md#24.4 reports, not just the raw numbers.
    assert EXTERNAL_SHARED["final_amr_a"] > EXTERNAL_SILOED["final_amr_a"]
    assert EXTERNAL_SHARED["final_amr_b"] < EXTERNAL_SILOED["final_amr_b"]
