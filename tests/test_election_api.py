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
