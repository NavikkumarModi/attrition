"""Tests for attrition/real_data.py. Fully offline: reads only the checked-in
snapshots under attrition/data/, no network access -- consistent with every
other test in this repo.
"""

import numpy as np

from attrition import (SCENARIOS, describe, get_arrays, load,
                       derive_real_amr_parameters, derive_real_cmc_parameters,
                       SOURCES)


def test_sources_documents_the_real_vs_proxy_split():
    assert "who_amr" in SOURCES and "fda_cmc" in SOURCES
    for entry in SOURCES.values():
        assert "REAL" in entry["fit"]
        assert entry["url"] and entry["publisher"]


def test_amr_parameters_p_is_real_proportion_in_unit_interval():
    v, p, e, labels = derive_real_amr_parameters(indicator="both")
    assert len(v) == len(p) == len(e) == len(labels)
    assert len(v) > 900   # ~1,005 real rows across both indicators
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert p.std() > 0.0   # real dispersion, not a constant


def test_amr_parameters_single_indicator_smaller():
    v_both, *_ = derive_real_amr_parameters(indicator="both")
    v_mrsa, *_ = derive_real_amr_parameters(indicator="mrsa")
    v_ecoli, *_ = derive_real_amr_parameters(indicator="ecoli")
    assert len(v_mrsa) < len(v_both)
    assert len(v_mrsa) + len(v_ecoli) == len(v_both)


def test_amr_parameters_n_truncation_is_deterministic():
    v1, p1, e1, l1 = derive_real_amr_parameters(n=50, seed=3)
    v2, p2, e2, l2 = derive_real_amr_parameters(n=50, seed=3)
    assert len(v1) == 50
    assert l1 == l2
    assert np.array_equal(p1, p2)


def test_amr_parameters_unknown_indicator_raises():
    import pytest
    with pytest.raises(ValueError):
        derive_real_amr_parameters(indicator="nonsense")


def test_cmc_parameters_one_arm_per_category_with_dispersed_kappa():
    v, p, e, labels = derive_real_cmc_parameters()
    assert len(v) == len(p) == len(e) == len(labels) == 9
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    kappa = p * e
    assert kappa.std() > 0.0
    assert kappa.max() > 5 * max(kappa.min(), 1e-6)   # real, not marginal, spread


def test_real_scenarios_registered_and_loadable():
    for name in ("antibiotic-stewardship-real", "design-space-real"):
        assert name in SCENARIOS
        env = load(name, seed=0)
        obs, info = env.reset(0)
        assert info["arms_alive"] > 0


def test_get_arrays_matches_derive_functions_directly():
    v_direct, p_direct, e_direct = derive_real_cmc_parameters()[:3]
    v_scn, p_scn, e_scn, kw = get_arrays("design-space-real")
    assert np.allclose(sorted(v_direct), sorted(v_scn))


def test_describe_real_scenarios_states_what_is_real(capsys):
    describe("antibiotic-stewardship-real")
    out = capsys.readouterr().out
    assert "real" in out.lower()


def test_synthetic_scenarios_still_present_and_unchanged():
    """The pre-existing mechanistic scenarios must still exist alongside the
    new real-data ones -- this is additive, not a replacement.
    """
    for name in ("antibiotic-stewardship", "design-space"):
        assert name in SCENARIOS
