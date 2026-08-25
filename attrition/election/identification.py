"""The identification-cost result for elections, promised early in this
project's planning and closed out here: how much a campaign gives up to
even know whether contact is helping or hurting.

Not a new theorem. Theorem 3 (`THEORY.md`, "estimation impossibility under
consumption") already proves the externality vector `e` cannot be
estimated to arbitrary precision no matter the horizon, because all
information about `e_i` enters through the reward-level shift at the
moment arm `i` "dies" (here: the moment a contact-intensity level actually
saturates a persuadable segment), and the episode's information budget is
capped by `E[N] = sum_i 1/p_i`, not by how long the campaign runs. This
module evaluates that closed-form floor at the real, heterogeneous `p`
already derived for each Phase-1 constituency
(`attrition.election.persuasion`) -- a new application of an existing,
proven result, the same relationship every other transplant in this
project has to what it transplants.

The general (heterogeneous-`p`) form, not the homogeneous simplification
`THEORY.md` uses for its closed-form headline number:

    E[N] = sum_i 1/p_i           (over arms with p_i > 0 -- see below)
    SE(delta * e_i) >= 2 * sigma / sqrt(E[N])

An intensity level with `p_i = 0` (never observed to saturate a segment
within the derivation's horizon) contributes no term at all -- there is no
floor to compute for it because there is no identifying event, at any
horizon. That is a sharper, not milder, version of Theorem 3's message,
and it is reported explicitly rather than silently excluded.
"""

import numpy as np

from .persuasion import derive_persuasion_parameters_from_constituency
from .population import ALL_PHASE1_CONSTITUENCIES

__all__ = ["identification_floor", "PHASE1_IDENTIFICATION_FLOORS"]


def identification_floor(constituency, sigma=0.05, intensity=None):
    """Theorem 3's closed-form estimation floor for a real constituency's
    derived campaign-persuasion parameters.

    Returns a dict: how many of the intensity levels ever generate an
    identifying (saturation) event at all (`identifiable`/`total`), the
    expected information budget `E[N]` those levels provide, and the
    resulting standard-error floor on estimating `e_a` -- a floor no
    amount of additional campaign time lowers, by Theorem 3. `None` for
    `expected_N`/`se_floor` if no intensity level ever saturates.
    """
    v, p, e, intensity_, s0 = derive_persuasion_parameters_from_constituency(
        constituency, intensity=intensity)
    identifiable = p > 1e-6
    n_identifiable = int(identifiable.sum())
    result = {"s0": s0, "n_intensity_levels": len(p),
             "n_identifiable": n_identifiable,
             "expected_N": None, "se_floor": None}
    if n_identifiable == 0:
        return result
    expected_N = float(np.sum(1.0 / p[identifiable]))
    result["expected_N"] = expected_N
    result["se_floor"] = 2.0 * sigma / np.sqrt(expected_N)
    return result


PHASE1_IDENTIFICATION_FLOORS = {
    (c.state, c.name): identification_floor(c) for c in ALL_PHASE1_CONSTITUENCIES
}
# Passes the Constituency OBJECT, not c.name, and keys by (state, name):
# three real names are ambiguous across states at national scale (see
# lokdhaba.get_constituency's docstring) -- resolving through a name-only
# lookup here crashed the whole package import the first time this ran at
# full scale (AURANGABAD exists in both Maharashtra and Bihar). Passing
# the already-disambiguated object sidesteps the lookup instead of trying
# to make the lookup smarter than it needs to be.
