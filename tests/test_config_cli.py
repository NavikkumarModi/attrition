"""Tests for the config-file loader and the CLI built on top of it.

Offline only: config files select `llm_client: "mock"` implicitly (the only
option), so nothing here touches the network.
"""

import json

import pytest

from attrition import cli
from attrition.config import build_from_config, load_config
from attrition.consumable import ConsumableBandit
from attrition.simultaneous import SimultaneousPool


def _inline_config(mode="turn-taking", baselines=None):
    return {
        "domain": {"v": [1.0, 0.8, 0.5], "p": [0.5, 0.5, 0.5], "e": [1.0, 0.5, 0.0]},
        "delta": 0.1, "horizon": 6, "seed": 0, "mode": mode,
        "population": [
            {"persona": "dr-conservative"},
            {"name": "custom", "role": "tester", "risk_tolerance": 0.7},
        ],
        "baselines": baselines or [],
    }


def test_build_from_config_turn_taking():
    env, population, baselines, meta = build_from_config(_inline_config())
    assert isinstance(env, ConsumableBandit)
    assert len(population) == 2
    assert meta["mode"] == "turn-taking"


def test_build_from_config_simultaneous():
    pool, population, baselines, meta = build_from_config(
        _inline_config(mode="simultaneous"))
    assert isinstance(pool, SimultaneousPool)
    assert pool.m == 2


def test_build_from_config_unknown_persona_raises():
    cfg = _inline_config()
    cfg["population"] = [{"persona": "no-such-persona"}]
    with pytest.raises(KeyError):
        build_from_config(cfg)


def test_build_from_config_unknown_baseline_raises():
    cfg = _inline_config(baselines=["not-a-policy"])
    with pytest.raises(KeyError):
        build_from_config(cfg)


def test_load_config_json_roundtrip(tmp_path):
    cfg = _inline_config()
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    loaded = load_config(path)
    assert loaded == cfg


def test_cli_list_domains(capsys):
    rc = cli.main(["list-domains"])
    assert rc == 0
    assert "antibiotic-stewardship" in capsys.readouterr().out


def test_cli_describe(capsys):
    rc = cli.main(["describe", "antibiotic-stewardship"])
    assert rc == 0
    assert "phenomenon" in capsys.readouterr().out


def test_cli_run_turn_taking(tmp_path, capsys):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_inline_config(baselines=["greedy", "eci"])))
    rc = cli.main(["run", str(path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "system_value" in out and "greedy" in out and "eci" in out


def test_cli_run_simultaneous_with_trace(tmp_path, capsys):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_inline_config(mode="simultaneous")))
    trace_path = tmp_path / "trace.sqlite3"
    rc = cli.main(["run", str(path), "--trace", str(trace_path)])
    assert rc == 0
    assert trace_path.exists()
    out = capsys.readouterr().out
    assert "planner (optimum)" in out
