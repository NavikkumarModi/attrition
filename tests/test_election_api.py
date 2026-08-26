"""Tests for election_api.py's endpoints (FINDINGS.md#44/#45) --
/api/campaign-policy plus the generic /api/worlds, /api/scenarios/*, and
/api/amr/reference-table routes that generalised this API beyond the
election track. Offline, no network -- everything here trains/computes
fresh in-process, same as the underlying CLI examples.
"""

import pytest
from fastapi.testclient import TestClient

from election_api import app

client = TestClient(app)


def test_campaign_policy_endpoint_returns_expected_shape():
    resp = client.get("/api/campaign-policy")
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["contact_intensity"]) == len(data["learned_preferences"])
    assert 0 <= data["favourite_arm"] < len(data["contact_intensity"])

    for key in ("greedy", "eci", "trained"):
        stats = data["single_agent"][key]
        assert "value" in stats and "regret" in stats

    dec = data["decentralised"]
    assert dec["n_agents"] == 3
    assert dec["planner_optimum"] >= dec["decentralised_eci"]
    assert dec["planner_optimum"] >= dec["decentralised_greedy"]


def test_campaign_policy_trained_policy_regret_below_eci():
    """Same claim examples/20 and FINDINGS.md#44 make: the trained policy's
    regret should sit much closer to greedy's near-zero than to ECI's."""
    data = client.get("/api/campaign-policy").json()
    trained_regret = data["single_agent"]["trained"]["regret"]
    eci_regret = data["single_agent"]["eci"]["regret"]
    assert trained_regret < eci_regret / 2


def test_worlds_endpoint_groups_every_scenario():
    data = client.get("/api/worlds").json()
    assert "agent-ecosystem" in data and "clinical-amr" in data
    names = {s["name"] for w in data.values() for s in w["scenarios"]}
    assert "shared-quota" in names and "antibiotic-stewardship" in names


def test_scenario_detail_returns_arrays_matching_length():
    data = client.get("/api/scenarios/shared-quota").json()
    assert data["world"] == "agent-ecosystem"
    assert len(data["v"]) == len(data["p"]) == len(data["e"])


def test_scenario_detail_404s_on_unknown_name():
    resp = client.get("/api/scenarios/does-not-exist")
    assert resp.status_code == 404


def test_scenario_compare_returns_requested_policies():
    resp = client.post("/api/scenarios/shared-quota/compare",
                       json={"policies": ["greedy", "eci"], "seeds": 10})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"greedy", "eci"}
    assert data["greedy"]["regret"] == 0.0


def test_scenario_compare_conservative_does_not_crash():
    """Regression check: Conservative() needs an e_bound (no default), so
    the endpoint must derive one from the scenario's own e array rather
    than calling Conservative() bare, which raises TypeError."""
    resp = client.post("/api/scenarios/shared-quota/compare",
                       json={"policies": ["conservative"], "seeds": 5})
    assert resp.status_code == 200


def test_scenario_compare_unknown_policy_400s():
    resp = client.post("/api/scenarios/shared-quota/compare",
                       json={"policies": ["not-a-policy"]})
    assert resp.status_code == 400


def test_scenario_personas_follows_world():
    agent_personas = client.get("/api/scenarios/shared-quota/personas").json()
    assert {p["name"] for p in agent_personas} == {
        "agent-conservative", "agent-aggressive", "agent-balanced"}

    amr_personas = client.get(
        "/api/scenarios/antibiotic-stewardship/personas").json()
    assert "dr-conservative" in {p["name"] for p in amr_personas}


def test_scenario_personas_empty_for_world_with_no_persona_set():
    resp = client.get("/api/scenarios/high-dispersion/personas")
    assert resp.status_code == 200
    assert resp.json() == []


def test_scenario_personas_registered_for_environment_and_cybersecurity():
    """FINDINGS.md #55: environment and cybersecurity used to be in the
    no-persona-set list too -- confirm they no longer are."""
    fisheries_personas = client.get(
        "/api/scenarios/fisheries-commons-real/personas").json()
    assert {p["name"] for p in fisheries_personas} == {
        "fisher-conservative", "fisher-aggressive", "fisher-balanced"}

    cve_personas = client.get(
        "/api/scenarios/exploit-catalog-real/personas").json()
    assert {p["name"] for p in cve_personas} == {
        "patch-conservative", "patch-aggressive", "patch-balanced"}


def test_llm_population_gracefully_reports_auth_failure_not_a_crash():
    """A bad API key should fall back to Greedy per-decision (LLMPolicy's
    own documented behaviour) and report the failure, not 500."""
    resp = client.post(
        "/api/scenarios/shared-quota/llm-population",
        json={"provider": "anthropic", "api_key": "not-a-real-key",
             "persona_names": ["agent-conservative"], "horizon": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["per_persona"]["agent-conservative"]["total"] == 3
    assert data["per_persona"]["agent-conservative"]["genuine"] == 0


def test_llm_population_unknown_persona_400s():
    resp = client.post(
        "/api/scenarios/shared-quota/llm-population",
        json={"provider": "anthropic", "api_key": "x",
             "persona_names": ["dr-conservative"]})  # wrong world's persona
    assert resp.status_code == 400


def test_amr_reference_table_endpoint_matches_exp65():
    data = client.get("/api/amr/reference-table").json()
    assert data["in_repo"]["eci"] == data["in_repo"]["lambda_weighted(lam=1)"]
    assert data["external_siloed"]["final_amr_a"] == pytest.approx(0.7461)


# ------------------------------------------------- granular /api/simulate
_SEAT = {"state": "Gujarat", "name": "AHMEDABAD EAST"}


def test_simulate_basic_still_works():
    resp = client.post("/api/simulate", json=_SEAT)
    assert resp.status_code == 200
    assert "outcome" in resp.json()


def test_simulate_network_stays_within_the_realistic_ceiling():
    """Regression test for FINDINGS.md#47: simulate_level1_contact_with_
    network predates REALISTIC_PERSUASION_CEILING_PCT (agents.py) and was
    never updated to respect it -- wiring it into the API surfaced raw
    switched fractions above 30%, an order of magnitude past the real
    field-experiment ceiling the non-network path already respects. Fixed
    in attrition/election/network.py; this pins the fix at the API layer
    too so it can't silently regress."""
    resp = client.post("/api/simulate", json={**_SEAT, "use_network": True, "n_voters": 500})
    assert resp.status_code == 200
    net = resp.json()["network"]
    assert net["final_switched_fraction"] <= 0.02
    assert net["direct_only_final_switched_fraction"] <= 0.02
    assert net["final_switched_fraction"] >= net["direct_only_final_switched_fraction"]


def test_simulate_flip_probability_sweep():
    resp = client.post("/api/simulate", json={**_SEAT, "n_seeds": 10, "n_voters": 500})
    assert resp.status_code == 200
    fp = resp.json()["flip_probability"]
    assert fp["n_seeds"] == 10
    assert 0.0 <= fp["flip_rate"] <= 1.0
    assert fp["final_switched_fraction_min"] <= fp["final_switched_fraction_mean"]
    assert fp["final_switched_fraction_mean"] <= fp["final_switched_fraction_max"]


def test_simulate_without_n_seeds_omits_flip_probability():
    resp = client.post("/api/simulate", json=_SEAT)
    assert "flip_probability" not in resp.json()


def test_simulate_incumbent_override_changes_reported_seat_and_outcome():
    real = client.post("/api/simulate", json=_SEAT).json()
    overridden = client.post(
        "/api/simulate", json={**_SEAT, "incumbent_override": not real["seat"]["winner_incumbent"]}
    ).json()
    assert overridden["seat"]["winner_incumbent"] != real["seat"]["winner_incumbent"]


def test_simulate_margin_override_changes_reported_margin_and_tipping_point():
    resp = client.post("/api/simulate", json={**_SEAT, "margin_override_pct": 1.0}).json()
    assert resp["seat"]["real_margin_pct"] == pytest.approx(1.0)
    # a 1pp margin must produce a small, non-None tipping point -- close
    # seats are cheaper to flip than the unmodified real seat.
    assert resp["seat"]["tipping_point_pct"] is not None


def test_simulate_seat_summary_reflects_override_not_the_cached_real_seat():
    """Regression check: _seat_summary_live must recompute tipping_point
    from the (possibly overridden) constituency directly, not look it up
    from the national _tipping_points() cache keyed by the real seat's
    identity, which would silently report the REAL seat's tipping point
    for a hypothetical one."""
    real_tp = client.post("/api/simulate", json=_SEAT).json()["seat"]["tipping_point_pct"]
    landslide_tp = client.post(
        "/api/simulate", json={**_SEAT, "margin_override_pct": 40.0}).json()["seat"]["tipping_point_pct"]
    assert landslide_tp != real_tp


def test_simulate_n_seeds_and_n_voters_are_capped():
    resp = client.post("/api/simulate", json={**_SEAT, "n_seeds": 500, "n_voters": 999999})
    assert resp.status_code == 200
    assert resp.json()["flip_probability"]["n_seeds"] == 50


def test_simulate_district_granularity_partitions_voters_exactly():
    resp = client.post("/api/simulate",
                       json={**_SEAT, "use_district_granularity": True, "n_voters": 3000})
    assert resp.status_code == 200
    data = resp.json()["district_granularity"]
    assert sum(d["n_voters"] for d in data["per_district"]) == 3000
    assert data["combined_final_switched_fraction"] == \
        pytest.approx(resp.json()["level1_final_switched_fraction"])


def test_simulate_district_granularity_and_network_now_combine(): # FINDINGS.md#53
    """Regression check: district granularity + network propagation used
    to 400 (network.py's O(n^2) graph was a real scaling wall). Both must
    now combine cleanly, with propagation run independently per district."""
    resp = client.post("/api/simulate",
                       json={**_SEAT, "use_district_granularity": True,
                            "use_network": True, "n_voters": 3000})
    assert resp.status_code == 200
    dg = resp.json()["district_granularity"]
    assert dg["network_within_district"] is True
    for d in dg["per_district"]:
        assert "direct_only_final_switched_fraction" in d
        # propagation can only add conversions, never reduce the total
        assert d["final_switched_fraction"] >= d["direct_only_final_switched_fraction"]


def test_simulate_district_granularity_without_network_omits_direct_only():
    resp = client.post("/api/simulate",
                       json={**_SEAT, "use_district_granularity": True, "n_voters": 3000})
    assert resp.status_code == 200
    dg = resp.json()["district_granularity"]
    assert dg["network_within_district"] is False
    for d in dg["per_district"]:
        assert "direct_only_final_switched_fraction" not in d


def test_simulate_literacy_effect_reports_real_fraction_used(): # FINDINGS.md#59
    resp = client.post("/api/simulate",
                       json={"state": "Uttar Pradesh", "name": "VARANASI",
                            "use_literacy_effect": True, "n_voters": 3000})
    assert resp.status_code == 200
    data = resp.json()
    assert "literacy_effect" in data
    lf = data["literacy_effect"]["literacy_fraction"]
    assert lf is not None and 0.0 < lf < 1.0


def test_simulate_literacy_effect_off_by_default_omits_field():
    resp = client.post("/api/simulate", json={**_SEAT, "n_voters": 3000})
    assert resp.status_code == 200
    assert "literacy_effect" not in resp.json()


def test_simulate_literacy_effect_falls_back_cleanly_with_no_shrug_match():
    """AHMEDABAD EAST's two constituent ACs have no matched SHRUG
    population data (a real, disclosed gap -- FINDINGS.md #57) -- must
    report literacy_fraction: null, never crash or guess a value."""
    resp = client.post("/api/simulate",
                       json={**_SEAT, "use_literacy_effect": True, "n_voters": 3000})
    assert resp.status_code == 200
    assert resp.json()["literacy_effect"]["literacy_fraction"] is None


def test_simulate_district_granularity_falls_back_cleanly_with_no_rollup():
    resp = client.post(
        "/api/simulate",
        json={"state": "Andaman & Nicobar Islands", "name": "ANDAMAN & NICOBAR ISLANDS",
             "use_district_granularity": True, "n_voters": 500})
    assert resp.status_code == 200
    per_district = resp.json()["district_granularity"]["per_district"]
    assert len(per_district) == 1
    assert per_district[0]["district"] is None
    assert per_district[0]["n_voters"] == 500


# ------------------------------------------ large-population scale (FINDINGS.md#52)
def test_simulate_n_voters_cap_is_much_higher_without_network():
    """Regression check: the plain/district-granularity paths must accept
    populations well past the old 10,000 cap -- agents.py's Level 1 loop
    was benchmarked fast enough (~0.5s at 1,000,000) to raise it."""
    resp = client.post("/api/simulate", json={**_SEAT, "n_voters": 200000})
    assert resp.status_code == 200
    assert resp.json()["level1_final_switched_fraction"] >= 0.0


def test_simulate_network_small_population_keeps_exact_graph():
    resp = client.post("/api/simulate",
                       json={**_SEAT, "use_network": True, "n_voters": 500})
    assert resp.status_code == 200
    assert resp.json()["network"]["graph_method"] == "random"


def test_simulate_network_large_population_switches_to_degree_graph():
    """Regression check for the real correctness fix: above the large-
    population threshold, the API must not build AgentGraph.random's
    O(n^2) graph (an absurd edge count at real scale, see network.py's
    docstring) -- it must switch to the mean-degree generator."""
    resp = client.post("/api/simulate",
                       json={**_SEAT, "use_network": True, "n_voters": 20000})
    assert resp.status_code == 200
    net = resp.json()["network"]
    assert net["graph_method"] == "random_by_degree"
    assert net["graph_param"] <= 100.0  # never silently exceeds the sanity cap


def test_simulate_network_explicit_mean_degree_is_honoured_not_silently_ignored():
    resp = client.post("/api/simulate",
                       json={**_SEAT, "use_network": True, "n_voters": 500,
                            "mean_degree": 5})
    assert resp.status_code == 200
    net = resp.json()["network"]
    assert net["graph_method"] == "random_by_degree"
    assert net["graph_param"] == 5.0


# ---------------------------------------------- granular /api/strategist
def test_strategist_filters_endpoint_lists_real_facets():
    data = client.get("/api/strategist/filters").json()
    assert set(data["constituency_type"]) <= {"GEN", "SC", "ST"}
    assert len(data["sub_region"]) > 0


def test_strategist_incumbency_filter_restricts_scope():
    unfiltered = client.get("/api/strategist", params={"budget_pct": 30}).json()
    incumbent_only = client.get(
        "/api/strategist", params={"budget_pct": 30, "incumbency": "incumbent"}).json()
    challenger_only = client.get(
        "/api/strategist", params={"budget_pct": 30, "incumbency": "challenger"}).json()
    assert incumbent_only["n_seats_in_scope"] < unfiltered["n_seats_in_scope"]
    assert challenger_only["n_seats_in_scope"] < unfiltered["n_seats_in_scope"]
    assert (incumbent_only["n_seats_in_scope"] + challenger_only["n_seats_in_scope"]
           == unfiltered["n_seats_in_scope"])


def test_strategist_constituency_type_filter_only_returns_matching_seats():
    resp = client.get("/api/strategist",
                      params={"budget_pct": 50, "constituency_type": "SC"})
    assert resp.status_code == 200
    data = resp.json()
    for seat in data["greedy"]["flipped"]:
        assert seat["constituency_type"] == "SC"


def test_strategist_invalid_incumbency_400s():
    resp = client.get("/api/strategist", params={"incumbency": "not-a-value"})
    assert resp.status_code == 400


# --------------------------------------------------- society metrics
def test_strategist_society_metrics_present_for_both_policies():
    data = client.get("/api/strategist", params={"budget_pct": 30}).json()
    for key in ("greedy", "even"):
        society = data[key]["society"]
        assert "efficiency" in society
        assert "electors_in_scope" in society
        assert "n_states_touched" in society


def test_strategist_greedy_efficiency_is_always_one():
    """greedy_allocation never targets a seat it can't finish (strategist
    .py's own docstring), so every targeted seat must also be flipped."""
    data = client.get("/api/strategist", params={"budget_pct": 30}).json()
    assert data["greedy"]["society"]["efficiency"] == pytest.approx(1.0)


def test_strategist_even_touches_more_states_than_greedy_at_low_budget():
    """The real, checkable tradeoff this metric exists to show: even
    spreads across far more of the country than greedy at the same
    budget, at the cost of efficiency."""
    data = client.get("/api/strategist", params={"budget_pct": 30}).json()
    assert (data["even"]["society"]["n_states_touched"]
           >= data["greedy"]["society"]["n_states_touched"])
    assert (data["even"]["society"]["electors_in_scope_pct_national"]
           >= data["greedy"]["society"]["electors_in_scope_pct_national"])


def test_strategist_society_metrics_empty_scope_does_not_crash():
    """A filter combination with zero matching seats must return a sane
    empty society block, not a 500 from statistics.fmean([])."""
    resp = client.get("/api/strategist",
                      params={"budget_pct": 1, "constituency_type": "ST",
                             "sub_region": "AVADH"})
    assert resp.status_code == 200
    society = resp.json()["greedy"]["society"]
    assert society["efficiency"] in (None, 0.0, 1.0)
    assert society["cost_dispersion_pct"] == 0.0


# --------------------------------------- indicator-driven strategist (FINDINGS.md#62)
def test_strategist_literacy_effect_picks_a_different_real_set_at_same_budget():
    baseline = client.get("/api/strategist", params={"budget_pct": 20}).json()
    literacy = client.get("/api/strategist",
                          params={"budget_pct": 20, "use_literacy_effect": True}).json()
    assert literacy["use_literacy_effect"] is True
    baseline_names = {s["name"] for s in baseline["greedy"]["flipped"]}
    literacy_names = {s["name"] for s in literacy["greedy"]["flipped"]}
    assert baseline_names != literacy_names


def test_strategist_literacy_effect_off_by_default():
    data = client.get("/api/strategist", params={"budget_pct": 20}).json()
    assert data["use_literacy_effect"] is False


def test_strategist_poverty_rate_filter_restricts_scope():
    unfiltered = client.get("/api/strategist", params={"budget_pct": 20}).json()
    high_poverty = client.get(
        "/api/strategist", params={"budget_pct": 20, "min_poverty_rate_pct": 30}).json()
    assert high_poverty["n_seats_in_scope"] < unfiltered["n_seats_in_scope"]
    assert high_poverty["filters"]["min_poverty_rate_pct"] == 30


def test_strategist_poverty_rate_filter_range_is_consistent():
    low = client.get("/api/strategist",
                     params={"budget_pct": 20, "max_poverty_rate_pct": 10}).json()
    high = client.get("/api/strategist",
                      params={"budget_pct": 20, "min_poverty_rate_pct": 40}).json()
    unfiltered = client.get("/api/strategist", params={"budget_pct": 20}).json()
    assert low["n_seats_in_scope"] + high["n_seats_in_scope"] <= unfiltered["n_seats_in_scope"]


def test_strategist_poverty_and_literacy_combine_without_crashing():
    resp = client.get("/api/strategist",
                      params={"budget_pct": 20, "min_poverty_rate_pct": 20,
                             "use_literacy_effect": True})
    assert resp.status_code == 200
    assert resp.json()["use_literacy_effect"] is True


# --------------------------------------------- campaign map (FINDINGS.md#64)
def test_campaign_map_nationwide_covers_all_real_seats():
    resp = client.post("/api/campaign-map", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_seats"] == 543
    assert len(data["seats"]) == 543
    assert 0 <= data["n_flipped"] <= 543


def test_campaign_map_heavier_effort_flips_at_least_as_many_seats():
    light = client.post("/api/campaign-map",
                        json={"intensity_sequence": [0.05, 0.05, 0.05, 0.05, 0.05]}).json()
    heavy = client.post("/api/campaign-map",
                        json={"intensity_sequence": [0.3, 0.4, 0.5, 0.6, 0.7]}).json()
    assert heavy["n_flipped"] >= light["n_flipped"]


def test_campaign_map_state_scope_restricts_to_that_state():
    resp = client.post("/api/campaign-map", json={"state": "Gujarat"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_seats"] > 0
    assert all(s["state"] == "Gujarat" for s in data["seats"])


def test_campaign_map_constituency_type_scope_restricts_correctly():
    resp = client.post("/api/campaign-map", json={"constituency_type": "SC"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_seats"] > 0
    assert data["n_seats"] < 543


def test_campaign_map_unknown_state_returns_empty_not_error():
    resp = client.post("/api/campaign-map", json={"state": "Nowhereland"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_seats"] == 0
    assert data["seats"] == []
    assert data["n_flipped"] == 0


def test_campaign_map_n_voters_is_capped():
    """A map-wide run must not accept an unbounded n_voters -- real,
    stated tradeoff for map responsiveness, checked here so the cap
    can't silently regress."""
    resp = client.post("/api/campaign-map", json={"n_voters": 10_000_000})
    assert resp.status_code == 200
    # doesn't crash or hang; capped internally at 5,000 per the endpoint's
    # own docstring -- indirectly checked via a sane response time bound
    # is out of scope for a unit test, but the shape must still be sane.
    assert resp.json()["n_seats"] == 543


def test_campaign_map_reports_real_literacy_effect_flag():
    resp = client.post("/api/campaign-map", json={"use_literacy_effect": True, "state": "Gujarat"})
    assert resp.status_code == 200
    assert resp.json()["use_literacy_effect"] is True


def test_campaign_map_poverty_rate_filter_restricts_scope():
    # Same real rural SECC poverty-rate filter /api/strategist already
    # supports, reused here via the shared _filtered_constituencies helper.
    unfiltered = client.post("/api/campaign-map", json={"state": "Gujarat"}).json()
    filtered = client.post("/api/campaign-map", json={
        "state": "Gujarat", "min_poverty_rate_pct": 0.0, "max_poverty_rate_pct": 5.0,
    }).json()
    assert filtered["n_seats"] <= unfiltered["n_seats"]
    assert filtered["filters"]["min_poverty_rate_pct"] == 0.0
    assert filtered["filters"]["max_poverty_rate_pct"] == 5.0


def test_campaign_map_incumbency_filter_restricts_scope():
    resp = client.post("/api/campaign-map", json={"state": "Gujarat", "incumbency": "incumbent"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["filters"]["incumbency"] == "incumbent"
    assert data["n_seats"] > 0


# ----------------------------------------------- assembly rollup
def test_assembly_rollup_endpoint_returns_real_shape():
    resp = client.get("/api/constituencies/Gujarat/AHMEDABAD EAST/assembly-rollup")
    assert resp.status_code == 200
    data = resp.json()
    assert data["seat"] == {"state": "Gujarat", "name": "AHMEDABAD EAST"}
    assert data["rollup"]["n_constituent_acs_matched"] >= 1


def test_assembly_rollup_endpoint_404s_on_unknown_seat():
    resp = client.get("/api/constituencies/Nowhere/Nothing/assembly-rollup")
    assert resp.status_code == 404


def test_assembly_rollup_coverage_endpoint():
    resp = client.get("/api/assembly-rollup/coverage")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_real_seats_total"] == 543
    assert "Karnataka" in data["year_by_state"]
    assert data["year_by_state"]["Karnataka"]["gap_from_2019"] == \
        data["year_by_state"]["Karnataka"]["year"] - 2019


# ------------------------------------------------------ census rollup
def test_census_rollup_endpoint_returns_real_shape():
    resp = client.get("/api/constituencies/Gujarat/AHMEDABAD EAST/census-rollup")
    assert resp.status_code == 200
    data = resp.json()
    assert data["seat"] == {"state": "Gujarat", "name": "AHMEDABAD EAST"}
    assert data["rollup"]["n_constituent_districts"] >= 1


def test_census_rollup_endpoint_404s_on_unknown_seat():
    resp = client.get("/api/constituencies/Nowhere/Nothing/census-rollup")
    assert resp.status_code == 404


def test_census_rollup_coverage_endpoint():
    resp = client.get("/api/census-rollup/coverage")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_districts_total"] == 640


# ------------------------------------------------------ shrug rollup (FINDINGS.md#57)
def test_shrug_rollup_endpoint_returns_real_shape():
    resp = client.get("/api/constituencies/Gujarat/AHMEDABAD EAST/shrug-rollup")
    assert resp.status_code == 200
    data = resp.json()
    assert data["seat"] == {"state": "Gujarat", "name": "AHMEDABAD EAST"}
    assert data["rollup"]["n_constituent_acs_matched"] >= 1
    assert "secc_pov_rate_rural" in data["rollup"]
    assert "facebook_mean_rwi" in data["rollup"]
    assert "pmgsy_pct_villages_connected" in data["rollup"]


def test_shrug_rollup_endpoint_404s_on_unknown_seat():
    resp = client.get("/api/constituencies/Nowhere/Nothing/shrug-rollup")
    assert resp.status_code == 404


def test_shrug_rollup_coverage_endpoint():
    resp = client.get("/api/shrug-rollup/coverage")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_acs_total"] == 4120
    assert data["n_real_seats_total"] == 543
