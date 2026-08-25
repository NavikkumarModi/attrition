"""Phase 2 of the election track: connecting the abstract persuasion-commons
theorem (`attrition.engines.derive_campaign_parameters`, FINDINGS.md#28) to
the real, validated Phase-1 population (FINDINGS.md#27) -- deliberately a
partial connection, not a full recalibration, following the same scope
discipline as `docs/amr-stewardship-extension.md#5`'s rho-anchoring: anchor
what is honestly derivable from real data, state plainly what still isn't.

What's anchored: `S0`, the engine's initial persuadable-compartment
fraction. What it's anchored to: a real constituency's **margin** --
`winner_share - runner_up_share`, both real vote shares from
`attrition.election.lokdhaba`. The reasoning, stated as a proxy claim, not
a measured one: a close contest (small margin) means a large share of the
electorate divided its support nearly evenly, which is what "highly
contestable/persuadable" should look like; a landslide (large margin)
means most of the electorate had already settled, which is what "mostly
already-decided" should look like. `persuadable_fraction = 1 - margin/100`,
clipped away from the degenerate 0/1 boundaries the engine is not stable
at.

What's NOT anchored: `R0` (initial saturated fraction), `r_S`/`r_R`
(growth-rate asymmetry), `a_SR`/`a_RS` (cross-competition), `kill_max`
(contact effectiveness ceiling) -- all still the engine's synthetic
defaults. This is the same honesty boundary the AMR robustness leg drew:
one real anchor checked and reported, not a claim that every parameter is
now real.
"""

import numpy as np

from .population import Constituency, build_constituency, ALL_PHASE1_CONSTITUENCIES
from ..engines import derive_arm_parameters

__all__ = ["real_margin_pct", "persuadable_fraction_from_margin",
          "derive_persuasion_parameters_from_constituency",
          "PHASE1_PERSUADABLE_FRACTIONS"]


def real_margin_pct(constituency):
    """Real winner-minus-runner-up vote share, in percentage points --
    straight from the checked-in LokDhaba snapshot, no proxy involved."""
    shares = sorted(constituency.vote_share_pct, reverse=True)
    return float(shares[0] - shares[1])


def persuadable_fraction_from_margin(constituency, floor=0.05, ceiling=0.95):
    """The one real anchor this module adds -- see module docstring. Clipped
    away from 0/1 because LotkaVolterraTumour is degenerate at the
    boundary (an all-persuadable or zero-persuadable starting state removes
    the competitive-release dynamic the engine exists to exhibit)."""
    margin = real_margin_pct(constituency)
    return float(np.clip(1.0 - margin / 100.0, floor, ceiling))


def derive_persuasion_parameters_from_constituency(constituency, intensity=None,
                                                    horizon=20, verbose=False):
    """`derive_campaign_parameters`, with S0 anchored to one real
    constituency's margin instead of the engine's synthetic default --
    everything else (R0, r_S, r_R, a_SR, a_RS, kill_max, dt) stays at
    `derive_campaign_parameters`'s defaults. See module docstring for
    exactly what this does and does not establish.
    """
    if isinstance(constituency, str):
        constituency = build_constituency(constituency)
    s0 = persuadable_fraction_from_margin(constituency)
    intensity = np.linspace(0.15, 1.0, 5) if intensity is None else np.asarray(intensity)
    v, p, e, _ = derive_arm_parameters(
        doses=intensity, engine_kwargs={"dt": 5.0, "S0": s0}, horizon=horizon,
        sensitive_floor=0.02, s0_sd=0.26, verbose=verbose)
    return v, p, e, intensity, s0


PHASE1_PERSUADABLE_FRACTIONS = {
    (c.state, c.name): persuadable_fraction_from_margin(c)
    for c in ALL_PHASE1_CONSTITUENCIES
}
# Keyed by (state, name), not name alone: three real names are ambiguous
# across states at national scale (see lokdhaba.get_constituency's
# docstring) -- a name-only key would silently drop one state's entry in
# favour of the other's, found by checking, not assumed safe once this
# stopped being ten hand-picked, individually-unique seats.
