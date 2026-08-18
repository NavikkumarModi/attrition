"""Tests for the scenario registry as an extension point."""

import numpy as np
import pytest

from attrition import SCENARIOS, describe, from_arrays, get_arrays, load


@pytest.fixture(autouse=True)
def _clean_registry():
    """SCENARIOS is a package-wide singleton: anything a test registers must
    be removed afterwards, or it leaks into other tests/modules that iterate
    over every registered scenario (e.g. test_theorems.py's
    test_every_scenario_matches_its_documented_behaviour).
    """
    before = set(SCENARIOS)
    yield
    for name in set(SCENARIOS) - before:
        del SCENARIOS[name]


def test_from_arrays_registers_a_discoverable_domain():
    before = set(SCENARIOS)
    from_arrays(v=[1.0, 0.8, 0.5], p=[0.5, 0.5, 0.5], e=[1.0, 0.5, 0.0],
               name="test-widget-domain", agents=2, delta=0.1, horizon=10)
    assert "test-widget-domain" in SCENARIOS
    assert set(SCENARIOS) - before == {"test-widget-domain"}
    spec = SCENARIOS["test-widget-domain"]
    assert spec["agents"] == 2


def test_from_arrays_domain_loads_and_runs():
    from_arrays(v=[1.0, 0.8, 0.5], p=[0.5, 0.5, 0.5], e=[1.0, 0.5, 0.0],
               name="test-loadable-domain", agents=1, delta=0.1, horizon=5)
    env = load("test-loadable-domain", seed=0)
    obs, info = env.reset(0)
    assert info["arms_alive"] == 3


def test_from_arrays_copies_input_arrays():
    v = np.array([1.0, 0.8, 0.5])
    from_arrays(v=v, p=[0.5, 0.5, 0.5], e=[1.0, 0.5, 0.0],
               name="test-copy-domain", agents=1)
    v[:] = 0.0
    v2, _, _, _ = get_arrays("test-copy-domain")
    assert not np.allclose(v2, 0.0)


def test_get_arrays_matches_existing_scenario_shape():
    v, p, e, kw = get_arrays("antibiotic-stewardship")
    assert len(v) == len(p) == len(e)
    assert "delta" in kw and "horizon" in kw


def test_describe_does_not_error_on_registered_domain(capsys):
    from_arrays(v=[1.0], p=[0.5], e=[0.0], name="test-describe-domain", agents=1)
    describe("test-describe-domain")
    out = capsys.readouterr().out
    assert "test-describe-domain" in out
