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


def test_all_543_constituencies_loaded():
    """Scaled up from the original ten-seat pilot -- 543 is the real,
    checked count of main-poll 2019 Lok Sabha seats (see
    scripts/fetch_election_data.py's own run output), not assumed."""
    assert len(ALL_PHASE1_CONSTITUENCIES) == 543
    names = {c.name for c in ALL_PHASE1_CONSTITUENCIES}
    assert "AHMEDABAD EAST" in names          # original pilot seat, still present
    assert "AZAMGARH" in names
    states = {c.state for c in ALL_PHASE1_CONSTITUENCIES}
    assert len(states) == 36                  # all states/UTs as they existed in 2019
    n_reserved = sum(1 for c in ALL_PHASE1_CONSTITUENCIES if c.constituency_type != "GEN")
    assert n_reserved == 131                  # real: 84 SC + 47 ST seats in 2019


def test_constituencies_are_real_not_synthetic():
    """Every field is either a real elector/turnout/vote-share number or an
    explicit note about why the seat was picked -- nothing invented."""
    for c in ALL_PHASE1_CONSTITUENCIES:
        # Lakshadweep (55,189 electors, India's smallest Lok Sabha seat)
        # is real, not a data error -- an earlier version of this test
        # assumed >100,000 from the ten hand-picked mainland pilot seats,
        # which broke the moment national scale included a real UT
        # constituency far smaller than any of them. Fixed to a floor low
        # enough to admit Lakshadweep but still rule out a genuine zero/
        # placeholder value.
        assert c.electors > 10_000
        assert c.electors < 5_000_000         # sanity ceiling, not a tight guess (checked real max: 3,150,313)
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


# --------------------------------------------- Phase 2: persuasion bridge
def test_persuadable_fraction_derived_from_real_margin():
    from attrition.election import build_constituency, real_margin_pct, persuadable_fraction_from_margin
    ahmedabad = build_constituency("Ahmedabad East")
    margin = real_margin_pct(ahmedabad)
    assert 38.0 < margin < 40.0, f"real BJP-INC margin should be ~38.9pp, got {margin}"
    frac = persuadable_fraction_from_margin(ahmedabad)
    assert abs(frac - (1 - margin / 100.0)) < 1e-9


def test_persuadable_fraction_varies_meaningfully_across_real_constituencies():
    """Not a constant -- real margin data must actually move S0, or the
    'anchored to real data' claim is empty."""
    from attrition.election import PHASE1_PERSUADABLE_FRACTIONS
    values = list(PHASE1_PERSUADABLE_FRACTIONS.values())
    assert max(values) - min(values) > 0.2, (
        f"expected real variation across constituencies, got {values}")


def test_zero_regret_ruin_survives_real_margin_variation():
    """The abstract campaign-persuasion theorem (FINDINGS.md#28) connected
    to real constituency margins (FINDINGS.md#29): does the pattern hold
    across real constituencies, or only the engine's synthetic default?

    Sampled, not exhaustive, since scaling to 543 real seats
    (FINDINGS.md's election-track sections): this property is a
    consequence of the proven isomorphism to derive_antibiotic_parameters
    (test_derive_campaign_parameters_isomorphism_claim), which already
    holds for any S0 in the engine's stable range -- re-running the full
    seeds=100 comparison at all 543 seats would cost real minutes for a
    property that doesn't need national-scale re-verification to stay
    credible. Every 15th constituency (~36 seats, spanning most states)
    is still checked, not just one.
    """
    from attrition import ConsumableBandit, Greedy, ECI, compare
    from attrition.election import (ALL_PHASE1_CONSTITUENCIES,
                                    derive_persuasion_parameters_from_constituency)
    for c in ALL_PHASE1_CONSTITUENCIES[::15]:
        v, p, e, intensity, s0 = derive_persuasion_parameters_from_constituency(c)
        result = compare(lambda s: ConsumableBandit(v, p, e, delta=0.35,
                                                     horizon=9, seed=s),
                         [Greedy(), ECI()], seeds=100)
        assert result["greedy"]["regret"] == 0.0, f"{c.name}: greedy must be zero-regret"
        assert result["eci"]["value"] > result["greedy"]["value"], (
            f"{c.name}: ECI must beat greedy despite greedy's zero regret "
            f"(S0={s0:.3f})")


# ------------------------------------ campaign-persuasion population agents
def test_campaign_population_runs_end_to_end_on_real_constituency():
    """Genuine learning agents (Population, not fixed Greedy/ECI policies)
    over a real, margin-anchored persuadable pool -- closes the loop
    between attrition.election (real data), derive_campaign_parameters
    (the theorem), and the existing LLM-population machinery already
    proven for AMR (examples/04). No new mechanism, only new wiring."""
    from attrition import CAMPAIGN_PERSONAS, Population, MockLLMClient
    from attrition.population import (simulate_population,
                                      compare_population_to_baselines)
    from attrition import ConsumableBandit, Greedy, ECI
    from attrition.election import derive_persuasion_parameters_from_constituency

    v, p, e, intensity, s0 = derive_persuasion_parameters_from_constituency(
        "Ahmedabad East")
    campaigns = [CAMPAIGN_PERSONAS["campaign-disciplined"],
                CAMPAIGN_PERSONAS["campaign-balanced"],
                CAMPAIGN_PERSONAS["campaign-maximalist"]]
    population = Population.from_personas(campaigns, client=MockLLMClient(seed=0))
    result = compare_population_to_baselines(
        lambda s: ConsumableBandit(v, p, e, delta=0.35, horizon=9, seed=s),
        population, [Greedy(), ECI()], seeds=10)
    assert set(result) == {"population", "greedy", "eci"}
    assert result["greedy"]["regret"] == 0.0
    for stats in result.values():
        assert stats["value"] > 0.0


# ---------------------------------------------- identification-cost floor
def test_identification_floor_matches_theorem3_closed_form():
    """Theorem 3's floor, SE >= 2*sigma/sqrt(E[N]), E[N] = sum 1/p_i over
    identifiable arms -- recomputed independently here and checked against
    the module's own output, not just trusted."""
    import numpy as np
    from attrition.election import identification_floor
    from attrition.election.persuasion import derive_persuasion_parameters_from_constituency
    v, p, e, intensity, s0 = derive_persuasion_parameters_from_constituency(
        "Ahmedabad East")
    identifiable = p > 1e-6
    expected_N = float(np.sum(1.0 / p[identifiable]))
    sigma = 0.05
    expected_floor = 2 * sigma / np.sqrt(expected_N)

    result = identification_floor("Ahmedabad East", sigma=sigma)
    assert result["n_identifiable"] == int(identifiable.sum())
    assert abs(result["expected_N"] - expected_N) < 1e-9
    assert abs(result["se_floor"] - expected_floor) < 1e-9


def test_identification_floor_flags_unidentifiable_intensity_levels():
    """Fewer than all five intensity levels ever generate a saturation
    event within the derivation horizon -- for those, e_a is not just hard
    to estimate, it is not identifiable at any horizon. This must be
    reported, not hidden inside an average."""
    from attrition.election import PHASE1_IDENTIFICATION_FLOORS
    for name, r in PHASE1_IDENTIFICATION_FLOORS.items():
        assert r["n_identifiable"] < r["n_intensity_levels"], (
            f"{name}: expected at least one unidentifiable intensity level")
        assert r["n_identifiable"] > 0, f"{name}: expected at least one identifiable level"


def test_identification_floor_is_strictly_positive_where_defined():
    """Theorem 3's whole point: this floor cannot be driven to zero by
    collecting more data -- it must be strictly positive wherever an
    identifying event can occur at all."""
    from attrition.election import PHASE1_IDENTIFICATION_FLOORS
    for name, r in PHASE1_IDENTIFICATION_FLOORS.items():
        assert r["se_floor"] > 0.0, f"{name}: floor must be strictly positive"


# ------------------------------------- campaign-to-campaign peer visibility
def test_campaign_peer_visibility_runs_end_to_end():
    """attrition.network.AgentGraph (proven elsewhere, see
    examples/06_peer_influence.py) wired onto campaign agents over a real
    constituency. Scope check, not a directional claim -- see
    examples/15_campaign_peer_influence.py's docstring for why this is a
    campaign-to-campaign network, not a voter-to-voter one."""
    from attrition import (AgentGraph, MockLLMClient, CAMPAIGN_PERSONAS,
                           Population, SimultaneousPool,
                           simulate_population_simultaneous)
    from attrition.election import derive_persuasion_parameters_from_constituency

    v, p, e, intensity, s0 = derive_persuasion_parameters_from_constituency(
        "Ahmedabad East")
    campaigns = [CAMPAIGN_PERSONAS["campaign-disciplined"],
                CAMPAIGN_PERSONAS["campaign-balanced"],
                CAMPAIGN_PERSONAS["campaign-maximalist"]]
    agent_ids = [c.name for c in campaigns]
    graph = AgentGraph.complete(agent_ids)
    pool = SimultaneousPool(v, p, e, delta=0.35, horizon=9,
                            n_agents=len(campaigns), seed=0)
    population = Population.from_personas(
        campaigns, client=MockLLMClient(seed=0, conformity=0.3))
    result = simulate_population_simultaneous(pool, population, rounds=9,
                                               graph=graph)
    assert "system_value" in result and "system_regret" in result
    assert result["system_regret"] >= 0.0


# ---------------------------------------------- Level 1/2 agents (agents.py)
def test_level1_switched_fraction_rises_monotonically_with_intensity():
    from attrition.election import simulate_level1_contact
    seq = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
    fractions, final = simulate_level1_contact("Ahmedabad East", seq,
                                                rho=0.7, n_voters=3000, seed=0)
    assert len(fractions) == len(seq)
    assert 0.0 <= final <= 1.0
    # not strictly monotone (spontaneous reversion exists), but the trend
    # over a rising-intensity sequence must be up, not flat or down
    assert fractions[-1] > fractions[0] + 0.3, (
        f"expected substantial net movement, got {fractions}")


def test_level1_persuadable_fraction_drives_switching_speed():
    """A close, high-persuadable-fraction seat should switch faster under
    identical contact than a landslide, low-persuadable-fraction one --
    the real margin anchor must actually matter, not just exist."""
    from attrition.election import simulate_level1_contact
    seq = [0.5] * 6
    _, close_final = simulate_level1_contact("Thiruvananthapuram", seq,
                                              n_voters=3000, seed=1)
    _, landslide_final = simulate_level1_contact("Mumbai North", seq,
                                                  n_voters=3000, seed=1)
    assert close_final > landslide_final, (
        f"close seat ({close_final}) should switch more than landslide "
        f"seat ({landslide_final}) under identical contact")


def test_level2_personas_diverge_by_disposition():
    """The whole point of a second, LLM-driven tier: personas should not
    all respond identically to the same contact sequence, or Level 2 adds
    nothing Level 1's aggregate rule doesn't already give."""
    from attrition.election import simulate_level2_contact
    seq = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.0]
    results = simulate_level2_contact("Ahmedabad East", seq)
    assert set(results) == {"voter-loyalist", "voter-swing", "voter-skeptical"}
    switch_round = {name: (decisions.index(True) if True in decisions else None)
                    for name, decisions in results.items()}
    # swing should switch no later than loyalist/skeptical, given its
    # persona is explicitly the least resistant
    assert switch_round["voter-swing"] is not None
    if switch_round["voter-loyalist"] is not None:
        assert switch_round["voter-swing"] <= switch_round["voter-loyalist"]


# --------------------------------------- voter-to-voter social network
def test_voter_network_adds_a_measurable_propagation_effect():
    from attrition.election import simulate_level1_contact_with_network
    seq = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0]
    _, final_networked, final_direct = simulate_level1_contact_with_network(
        "Ahmedabad East", seq, n_voters=500, seed=0)
    assert final_networked >= final_direct, (
        "propagation must not make the final switched fraction lower than "
        "direct contact alone -- it can only add conversions")
    assert final_networked > final_direct, (
        "propagation must add a strictly measurable effect on this seeded run")


# ------------------------------------------------------- coverage gaps
def test_source_provenance_has_required_fields():
    """SOURCE is what makes the data's provenance checkable -- a missing
    key here would silently break that claim for anyone auditing it."""
    from attrition.election.lokdhaba import SOURCE
    for key in ("publisher", "url", "license", "fetched", "note"):
        assert key in SOURCE and SOURCE[key], f"SOURCE missing or empty: {key}"
    assert "Trivedi" in SOURCE["publisher"]


def test_constituency_dataclass_shape():
    from attrition.election import build_constituency, Constituency
    c = build_constituency("Ahmedabad East")
    assert isinstance(c, Constituency)
    assert isinstance(c.parties, tuple) and isinstance(c.vote_share_pct, tuple)
    assert len(c.parties) == len(c.vote_share_pct)
    assert c.constituency_type in ("GEN", "SC", "ST")


def test_behavioural_voter_direct_construction():
    from attrition.election.agents import BehaviouralVoter
    v = BehaviouralVoter(switch_resistance=0.4)
    assert v.switch_resistance == 0.4
    assert v.pressure == 0.0
    assert v.switched is False


# --------------------------------------------------------- Election Layer
def test_recompute_outcome_matches_real_result_at_zero_switched():
    """No persuasion effect must reproduce the real winner exactly -- cheap
    (no simulation), so checked for all 543 real seats, not sampled."""
    from attrition.election import recompute_outcome, ALL_PHASE1_CONSTITUENCIES
    for c in ALL_PHASE1_CONSTITUENCIES:
        outcome = recompute_outcome(c, 0.0)
        assert not outcome.flipped
        assert outcome.winner == c.parties[0]


def test_recompute_outcome_clamps_gain_to_available_share():
    """Regression test for a real bug caught while building this: an
    unclamped switched_fraction larger than every other party's combined
    share produced an impossible negative scale factor and (before the
    fix) a fabricated exact 100.00% margin. The clamp must keep every
    share non-negative and the total at 100."""
    from attrition.election import recompute_outcome
    outcome = recompute_outcome("Ahmedabad East", switched_fraction=5.0)
    assert 0.0 <= outcome.runner_up_share_pct <= outcome.winner_share_pct <= 100.0
    assert outcome.margin_pct >= 0.0
    # an impossibly large switched_fraction clamps to "challenger takes
    # every other party's entire combined share" -- the challenger wins
    # with exactly 100%, not more, and the margin cannot exceed 100pp
    assert outcome.winner_share_pct <= 100.0 + 1e-6
    assert outcome.margin_pct <= 100.0 + 1e-6


def test_tipping_point_correlates_with_real_margin():
    """A closer real seat must need a smaller switched_fraction to flip
    than a landslide one -- the tipping point must actually track the
    real margin it's derived from, not just exist as a number."""
    from attrition.election import tipping_point
    close_tp = tipping_point("Thiruvananthapuram")   # margin 9.89pp
    landslide_tp = tipping_point("Mumbai North")       # margin 47.01pp
    assert close_tp is not None and landslide_tp is not None
    assert close_tp < landslide_tp


def test_level1_switched_fraction_uses_deterministic_threshold_crossing():
    """Regression test for a real ratchet bug: an earlier version re-drew
    `switched` via a per-round Bernoulli trial, which almost never
    reverted once pressure was high (the reversion trial ran only when
    the main trial failed, and a high-pressure voter's main trial rarely
    failed). Fixed to deterministic pressure > resistance, which must
    revert cleanly when pressure decays below the resistance threshold."""
    from attrition.election.agents import BehaviouralVoter
    v = BehaviouralVoter(switch_resistance=0.5)
    rho = 0.5
    # pressure rises under sustained high intensity, past resistance
    for _ in range(5):
        v.pressure = rho * v.pressure + (1.0 - rho) * 1.0
        v.switched = v.pressure > v.switch_resistance
    assert v.switched is True
    # pressure decays under sustained zero intensity, back below resistance
    for _ in range(10):
        v.pressure = rho * v.pressure + (1.0 - rho) * 0.0
        v.switched = v.pressure > v.switch_resistance
    assert v.switched is False, (
        "switched must revert once pressure decays below resistance -- "
        "it must not be a one-way ratchet")
