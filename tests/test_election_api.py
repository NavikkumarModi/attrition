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
    resp = client.get("/api/scenarios/fisheries-commons-real/personas")
    assert resp.status_code == 200
    assert resp.json() == []


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
