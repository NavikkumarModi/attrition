"""Test for election_api.py's new /api/campaign-policy endpoint (FINDINGS.md#44
/ examples/20_campaign_policy_checkpoint.py, exposed to the frontend). Offline,
no network -- trains fresh in-process, same as the CLI example.
"""

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
