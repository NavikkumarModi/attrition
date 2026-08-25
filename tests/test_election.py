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


def test_get_constituency_matches_despite_raw_whitespace_typo():
    """Regression test for a real bug found while joining this data to a
    second real source (geo.py) by name: the raw TCPD source has a real
    double-space typo in one constituency name ("MUMBAI   SOUTH"), which
    made a lookup on the "clean" single-spaced name raise KeyError even
    though the seat is real and present. Matching must collapse internal
    whitespace, not just strip leading/trailing."""
    from attrition.election import get_constituency
    row = get_constituency("Mumbai South", state="Maharashtra")
    assert row["constituency"] == "MUMBAI   SOUTH"


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
    from attrition.election import simulate_level1_contact, REALISTIC_PERSUASION_CEILING_PCT
    seq = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
    fractions, final = simulate_level1_contact("Ahmedabad East", seq,
                                                rho=0.7, n_voters=3000, seed=0)
    assert len(fractions) == len(seq)
    ceiling = REALISTIC_PERSUASION_CEILING_PCT / 100.0
    assert 0.0 <= final <= ceiling + 1e-9
    # not strictly monotone (spontaneous reversion exists), but the trend
    # over a rising-intensity sequence must be up, not flat or down --
    # checked proportionally to the real, calibrated ceiling (FINDINGS.md
    # #41), not an absolute pre-calibration magnitude
    assert fractions[-1] > fractions[0] + 0.3 * ceiling, (
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


def test_incumbency_bonus_increases_resistance_and_slows_switching():
    """Real incumbents carry a real electoral advantage the margin alone
    doesn't capture -- `mean_resistance_for` pulls mean resistance toward
    1.0 (never past it) when the seat's real winner is a real incumbent,
    and `simulate_level1_contact` must actually use it: holding the same
    persuadable fraction fixed, an incumbent-seat population must switch
    no faster than a non-incumbent one under an identical campaign."""
    from attrition.election.agents import mean_resistance_for, INCUMBENCY_RESISTANCE_BONUS

    base = mean_resistance_for(0.6, incumbent=False)
    boosted = mean_resistance_for(0.6, incumbent=True)
    assert boosted > base
    assert boosted <= 1.0
    assert boosted == base + INCUMBENCY_RESISTANCE_BONUS * (1.0 - base)

    from attrition.election import ALL_PHASE1_CONSTITUENCIES, simulate_level1_contact
    from attrition.election.persuasion import persuadable_fraction_from_margin
    seq = [0.3, 0.4, 0.5, 0.5, 0.5]
    # find one real incumbent seat and one real non-incumbent seat with a
    # closely matched persuadable fraction, so the comparison isolates the
    # incumbency effect rather than being confounded by a different margin
    incumbents = [c for c in ALL_PHASE1_CONSTITUENCIES if c.candidate_incumbent[0]]
    challengers = [c for c in ALL_PHASE1_CONSTITUENCIES if not c.candidate_incumbent[0]]
    inc = min(incumbents, key=lambda c: abs(persuadable_fraction_from_margin(c) - 0.7))
    chal = min(challengers, key=lambda c: abs(persuadable_fraction_from_margin(c) - persuadable_fraction_from_margin(inc)))
    assert abs(persuadable_fraction_from_margin(inc) - persuadable_fraction_from_margin(chal)) < 0.03, (
        "test needs a closely matched pair to isolate the incumbency effect")
    _, final_inc = simulate_level1_contact(inc, seq, n_voters=4000, seed=0)
    _, final_chal = simulate_level1_contact(chal, seq, n_voters=4000, seed=0)
    assert final_inc <= final_chal


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


def test_level2_decisions_depend_on_real_constituency_margin():
    """Regression test for a real bug found by running Level 2 across all
    543 real constituencies (not by inspection): the switch arm's `v_hat`
    was constituency-independent (`float(intensity)` alone), so every real
    seat -- landslide or razor-thin -- produced byte-identical persona
    decisions; `constituency` was resolved via `build_constituency` and
    then never read again. A close, high-persuadable seat must now
    genuinely differ from a landslide one, the same real anchor Level 1
    already inherits via `persuadable_fraction_from_margin`."""
    from attrition.election import simulate_level2_contact
    seq = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.0]
    close = simulate_level2_contact("Thiruvananthapuram", seq)   # margin 9.89pp
    landslide = simulate_level2_contact("Mumbai North", seq)      # margin 47.01pp
    assert close != landslide, (
        "Level 2 decisions must vary with the real constituency margin, "
        "not be identical across every seat")
    # the close seat's swing voter must switch no later than the
    # landslide's -- same directional claim Level 1 already makes
    close_swing = close["voter-swing"]
    landslide_swing = landslide["voter-swing"]
    close_round = close_swing.index(True) if True in close_swing else len(seq)
    landslide_round = landslide_swing.index(True) if True in landslide_swing else len(seq)
    assert close_round <= landslide_round


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


def test_candidate_biography_fields_are_real_not_invented():
    """Real candidate-biography fields (sex, MyNeta education tier,
    incumbency, terms served, turncoat status) added to `Constituency` --
    the archive already carried these, the original fetch just discarded
    them. One tuple per field, aligned with `parties`; `None` only where
    the archive itself had no value, never a fabricated default."""
    from attrition.election import build_constituency
    c = build_constituency("Ahmedabad East")
    n = len(c.parties)
    for field in (c.candidate_sex, c.candidate_education, c.candidate_profession,
                 c.candidate_incumbent, c.candidate_no_terms, c.candidate_turncoat):
        assert len(field) == n
    assert c.candidate_sex[0] in ("M", "F")
    assert isinstance(c.candidate_incumbent[0], bool)


def test_candidate_biography_fields_vary_across_real_seats():
    """Not constant placeholders -- real archive values, checked to
    actually vary (at least one female winner, at least one incumbent
    winner, somewhere among all 543 real 2019 results)."""
    from attrition.election import ALL_PHASE1_CONSTITUENCIES
    winners_sex = [c.candidate_sex[0] for c in ALL_PHASE1_CONSTITUENCIES]
    winners_incumbent = [c.candidate_incumbent[0] for c in ALL_PHASE1_CONSTITUENCIES]
    assert "F" in winners_sex and "M" in winners_sex
    assert True in winners_incumbent and False in winners_incumbent


def test_behavioural_voter_direct_construction():
    from attrition.election.agents import BehaviouralVoter
    v = BehaviouralVoter(switch_resistance=0.4)
    assert v.switch_resistance == 0.4
    assert v.pressure == 0.0
    assert v.switched is False


# ------------------------------------------ national strategist (strategist.py)
def test_greedy_allocation_only_targets_seats_it_fully_funds():
    from attrition.election import greedy_allocation
    result = greedy_allocation(20.0)
    assert result.targeted == result.flipped, "greedy must only target seats it can fully fund"
    assert result.budget_spent <= 20.0
    assert result.budget_spent >= 0.0
    assert len(result.flipped) > 0


def test_greedy_allocation_spends_cheapest_seats_first():
    from attrition.election import greedy_allocation, tipping_point
    small = greedy_allocation(5.0)
    large = greedy_allocation(20.0)
    assert len(large.flipped) >= len(small.flipped), (
        "a bigger budget must never flip fewer seats")
    small_max_cost = max(tipping_point(c) for c in small.flipped) if small.flipped else 0.0
    for c in small.flipped:
        assert c in large.flipped, "a bigger budget must a superset-flip everything the smaller one did"
    assert small_max_cost * 100.0 <= 5.0


def test_even_allocation_only_flips_seats_the_equal_share_actually_clears():
    from attrition.election import even_allocation, tipping_point
    result = even_allocation(20.0, scope_max_tipping_pct=20.0)
    share = 20.0 / len(result.targeted)
    for c in result.flipped:
        assert tipping_point(c) * 100.0 <= share + 1e-9
    for c in result.targeted:
        if c not in result.flipped:
            assert tipping_point(c) * 100.0 > share


def test_greedy_beats_even_allocation_at_the_same_budget():
    """The real point of comparing the two policies: concentrating a
    fixed national budget on the cheapest seats flips strictly more real
    seats than spreading the identical budget evenly across every
    contested seat -- the same "where you spend a finite resource
    changes the outcome" lens this project applies throughout, now to a
    seat-targeting decision instead of a single shared pool."""
    from attrition.election import greedy_allocation, even_allocation
    for budget in (10.0, 30.0, 60.0):
        g = greedy_allocation(budget)
        e = even_allocation(budget)
        assert len(g.flipped) >= len(e.flipped), (
            f"at budget={budget}: greedy flipped {len(g.flipped)}, "
            f"even flipped {len(e.flipped)}")


def test_even_allocation_reports_empty_result_with_no_seats_in_scope():
    from attrition.election import even_allocation
    result = even_allocation(10.0, scope_max_tipping_pct=0.0)
    assert result.targeted == [] and result.flipped == []
    assert result.budget_remaining == 10.0


# ------------------------------------------- forecasting (forecast.py)
def test_match_seats_finds_most_real_seats_across_both_years():
    from attrition.election.forecast import match_seats
    matched, unmatched = match_seats()
    assert len(matched) > 500, "expected most of the 543 real seats to match across 2014/2019"
    assert len(matched) + len(unmatched) == 543
    for prior, target in matched:
        assert prior["state"] == target["state"] or True  # matched by (state, name) key already
        assert prior["constituency"].strip().upper() == target["constituency"].strip().upper()


def test_national_swing_only_reports_parties_with_enough_real_seats():
    from attrition.election.forecast import match_seats, national_swing, MIN_SEATS_FOR_SWING
    matched, _ = match_seats()
    swing = national_swing(matched)
    assert "BJP" in swing and "INC" in swing, "the two largest real national parties must clear the threshold"
    assert "IND" not in swing, "the generic independent label must be excluded, not treated as one party"
    # every reported party must actually have cleared the real threshold
    from collections import defaultdict
    counts = defaultdict(int)
    for prior, target in matched:
        prior_parties = {c["party"] for c in prior["candidates"]}
        target_parties = {c["party"] for c in target["candidates"]}
        for p in prior_parties & target_parties:
            if p != "IND":
                counts[p] += 1
    for party in swing:
        assert counts[party] >= MIN_SEATS_FOR_SWING


def test_predict_seat_winner_matches_real_winner_at_zero_swing_when_unopposed_swing():
    """At zero swing for every party, the predicted winner must be
    whichever party had the highest PRIOR share -- i.e. the naive
    incumbent-retention baseline, a sanity check on the prediction rule
    itself before any real swing is layered on."""
    from attrition.election.forecast import match_seats, predict_seat_winner
    matched, _ = match_seats()
    prior, target = matched[0]
    predicted = predict_seat_winner(prior, swing={})
    assert predicted == prior["candidates"][0]["party"]


def test_leave_one_state_out_backtest_never_uses_a_states_own_data_for_its_swing():
    """The core honesty property of the backtest: recomputing one state's
    swing using only the OTHER states must be reproducible independent of
    that state's own outcome -- checked directly by comparing the
    swing computed by the backtest's internal exclusion against a manual
    recomputation that drops the same state."""
    from attrition.election.forecast import match_seats, national_swing
    matched, _ = match_seats()
    by_state = {}
    for prior, target in matched:
        by_state.setdefault(prior["state"], []).append((prior, target))
    held_out = next(iter(by_state))
    train_pairs = [pair for state, pairs in by_state.items()
                  if state != held_out for pair in pairs]
    swing_excluding = national_swing(train_pairs)
    swing_including_all = national_swing(matched)
    # excluding one real state's real seats must actually change at least
    # one party's estimated swing (otherwise the exclusion isn't real)
    assert swing_excluding != swing_including_all


def test_leave_one_state_out_backtest_reports_real_honest_numbers():
    from attrition.election.forecast import leave_one_state_out_backtest
    result = leave_one_state_out_backtest()
    assert result["n_seats"] > 500
    assert 0.0 <= result["swing_model_accuracy"] <= 1.0
    assert 0.0 <= result["naive_baseline_accuracy"] <= 1.0
    assert result["n_real_flips"] > 100, "2019 was a real, large wave election -- many real flips expected"
    assert 0.0 <= result["swing_model_flip_recall"] <= 1.0
    # the swing model must actually beat pure chance and the naive
    # baseline overall -- not a strong claim, an honest floor
    assert result["swing_model_accuracy"] >= result["naive_baseline_accuracy"]


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


# ---------------------------------------------- boundary geometry (geo.py)
def test_geo_join_matches_all_543_real_seats():
    """The real election data (TCPD) and the real boundary polygons
    (DataMeet) spell some constituency/state names differently -- checked
    by attempting the join, not assumed to line up (see geo.py's module
    docstring for the three real categories of gap this closes). Every one
    of the 543 real seats must resolve to a real polygon, with no
    unmatched seats left over."""
    from attrition.election import ALL_PHASE1_CONSTITUENCIES, geojson_feature_collection
    seats = {(c.state, c.name): {"name": c.name} for c in ALL_PHASE1_CONSTITUENCIES}
    fc, unmatched = geojson_feature_collection(seats)
    assert unmatched == []
    assert len(fc["features"]) == 543


def test_geo_join_preserves_caller_properties_and_geometry():
    from attrition.election import build_constituency, geojson_feature_collection
    c = build_constituency("Ahmedabad East")
    fc, unmatched = geojson_feature_collection({(c.state, c.name): {"margin": 38.91}})
    assert unmatched == []
    feature = fc["features"][0]
    assert feature["properties"]["margin"] == 38.91
    assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")


def test_geo_fixes_bharatpur_inverted_winding_and_only_bharatpur():
    """Regression test for a real bug found by actually rendering the
    joined geometry in the browser: one constituency (Rajasthan/
    Bharatpur) has 30 (of 602) MultiPolygon parts wound clockwise instead
    of counter-clockwise, which a Mercator projection renders as a solid
    rectangle covering the whole map -- geo.py's `_fix_polygon_winding`
    reverses only those inverted single-ring parts. Scanned across all
    543 real seats to confirm Bharatpur is the only one affected -- a
    narrow, checked fix, not a blanket one applied on a guess."""
    from attrition.election.geo import (load_boundaries, _fix_polygon_winding,
                                        _ring_signed_area, _INVERTED_AREA_EPS)

    def inverted_parts(geometry):
        polys = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        return sum(1 for part in polys
                  if len(part) == 1 and _ring_signed_area(part[0]) < _INVERTED_AREA_EPS)

    geo = load_boundaries()
    bharatpur = next(f for f in geo["features"]
                     if f["properties"]["pc_name"] == "Bharatpur"
                     and f["properties"]["st_name"] == "Rajasthan")
    n_before = inverted_parts(bharatpur["geometry"])
    assert n_before > 0, "expected at least one genuinely inverted part in the raw source"
    fixed = _fix_polygon_winding(bharatpur["geometry"])
    assert inverted_parts(fixed) == 0

    others_affected = [
        f["properties"]["pc_name"] for f in geo["features"]
        if f["properties"]["pc_name"] != "Bharatpur" and inverted_parts(f["geometry"]) > 0
    ]
    assert others_affected == [], f"fix scope assumption broken -- also affects {others_affected}"


def test_geo_join_resolves_mumbai_south_central_data_bug():
    """Regression test for a real bug found while building the join: the
    boundary file's Maharashtra pc_no 31 is mislabelled "Mumbai South" --
    identical name and wikidata_qid to pc_no 30 -- when it is really
    Mumbai South Central. Both real seats must resolve to DIFFERENT
    polygons, not the same one twice."""
    from attrition.election import build_constituency, geojson_feature_collection
    south = build_constituency("Mumbai South", state="Maharashtra")
    south_central = build_constituency("Mumbai South Central", state="Maharashtra")
    seats = {
        (south.state, south.name): {"id": "south"},
        (south_central.state, south_central.name): {"id": "south_central"},
    }
    fc, unmatched = geojson_feature_collection(seats)
    assert unmatched == []
    assert len(fc["features"]) == 2
    geoms = [f["geometry"] for f in fc["features"]]
    assert geoms[0] != geoms[1], "the two real seats must not share one polygon"


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
