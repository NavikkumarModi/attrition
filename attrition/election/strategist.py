"""A national campaign-strategist agent: given one finite total contact
budget, decides WHICH real seats to spend it on to flip the most seats --
a resource-ALLOCATION decision layered on top of Level 0/1/2 (which model
what happens to voters within one already-chosen seat), not a new voter
mechanism itself.

This is a genuinely new decision problem, not a re-application of
Theorems 1-6 -- stated plainly so it isn't overclaimed. Those theorems
are about a single depleting resource consumed round-by-round within one
shared pool (a bandit with a consumption externality); this is a
one-shot portfolio choice across 543 INDEPENDENT real seats, each with
its own fixed "price" (the real `tipping_point`, in the same
percent-of-electorate unit throughout this project). It borrows this
project's own recurring lens -- a finite resource, spent where, at what
opportunity cost -- without claiming the bandit machinery transfers.

Two policies, both real, comparable baselines rather than a single
"optimiser" asserted to be correct:

`greedy_allocation` -- spend the budget on the CHEAPEST real seats to
flip first (ascending `tipping_point`), stopping when the budget runs
out. This is the same greedy-by-marginal-value philosophy as
`attrition.policies.Greedy` -- reused, not reinvented, for a seat-
targeting decision instead of an arm-pull decision.

`even_allocation` -- spread the SAME total budget equally across every
seat within `scope_max_tipping_pct` (seats a real strategist would
plausibly contest at all; landslide seats are excluded from the split,
stated as the deliberate scope choice it is, not assumed infinite
strategist attention). A seat only flips if its equal share actually
clears its own real tipping point.

Both take the real, checked `tipping_point` values already computed for
all 543 seats (`outcome.py`) as the cost model -- no new persuasion
magnitude is invented here; this module only decides where an existing,
already-illustrative-not-calibrated unit of effort gets spent.

Optional `use_literacy_effect=True` (FINDINGS.md#62) makes that cost
model indicator-driven, extending the SAME real, literature-checked
mechanism #59 already wired into `simulate_level1_contact` -- not a
second, disconnected one. A seat's real literacy rate (Census 2011, via
SHRUG) shifts its Level-1 mean switch-resistance (`agents.
mean_resistance_for`, Zaller's 1992 Receive-Accept-Sample model: lower
literacy makes a seat more persuadable under intense contact); this
module rescales that seat's `tipping_point` cost by the SAME ratio --
`resistance_with_literacy / resistance_without_literacy` -- so a more
persuadable seat costs proportionally less to flip here too, bounded by
the same `LITERACY_SUSCEPTIBILITY_ADJUSTMENT` cap `mean_resistance_for`
already uses, not a new, separately-invented magnitude. Seats with no
matched SHRUG literacy data keep their unadjusted real `tipping_point`,
never a guessed one.
"""
from .agents import literacy_fraction_for_seat, mean_resistance_for
from .outcome import tipping_point
from .persuasion import persuadable_fraction_from_margin
from .population import ALL_PHASE1_CONSTITUENCIES

__all__ = ["greedy_allocation", "even_allocation", "AllocationResult",
          "literacy_adjusted_cost_multiplier"]


def literacy_adjusted_cost_multiplier(constituency):
    """Real, bounded rescaling factor for one seat's `tipping_point` cost,
    from the ratio of literacy-adjusted to unadjusted mean switch-
    resistance (see module docstring). Returns 1.0 (no adjustment) for a
    seat with no matched real SHRUG literacy data -- never a guessed
    value standing in for a real one."""
    literacy = literacy_fraction_for_seat(constituency.state, constituency.name)
    if literacy is None:
        return 1.0
    persuadable = persuadable_fraction_from_margin(constituency)
    incumbent = constituency.candidate_incumbent[0]
    base = mean_resistance_for(persuadable, incumbent)
    adjusted = mean_resistance_for(persuadable, incumbent, literacy_fraction=literacy)
    if base <= 0:
        return 1.0
    return adjusted / base


class AllocationResult:
    """`targeted`: seats the policy chose to spend budget on.
    `flipped`: the subset of `targeted` that actually clears its own real
    tipping point (for `greedy_allocation` these are identical -- it only
    ever targets seats it fully funds; `even_allocation` can target a seat
    and still fail to flip it if the equal share falls short).
    `budget_spent`/`budget_remaining` in the same percent-of-electorate
    unit as `tipping_point`."""

    __slots__ = ("targeted", "flipped", "budget_spent", "budget_remaining")

    def __init__(self, targeted, flipped, budget_spent, budget_remaining):
        self.targeted = targeted
        self.flipped = flipped
        self.budget_spent = budget_spent
        self.budget_remaining = budget_remaining


def _real_tipping_points(constituencies, use_literacy_effect=False):
    """(seat, effective cost) pairs -- cost is the real `tipping_point`,
    rescaled by `literacy_adjusted_cost_multiplier` when
    `use_literacy_effect` is set (1.0, i.e. no change, for every seat
    otherwise, and for any seat with no matched real SHRUG literacy
    data)."""
    pairs = [(c, tipping_point(c)) for c in constituencies]
    pairs = [(c, tp) for c, tp in pairs if tp is not None]
    if not use_literacy_effect:
        return pairs
    return [(c, tp * literacy_adjusted_cost_multiplier(c)) for c, tp in pairs]


def greedy_allocation(budget_pct, constituencies=None, use_literacy_effect=False):
    """Fund the cheapest real seats to flip first (ascending real
    `tipping_point`, in percent-of-electorate terms) until `budget_pct` is
    exhausted. Every targeted seat is fully funded and therefore flips --
    greedy spends nothing on a seat it can't finish.

    `use_literacy_effect=True` ranks and charges seats by their real
    literacy-adjusted cost (see module docstring) instead of the raw
    `tipping_point` -- a genuinely different, indicator-driven ordering,
    not just a display change: a seat with lower real literacy can now be
    prioritised ahead of a nominally cheaper but higher-literacy one."""
    cs = constituencies if constituencies is not None else ALL_PHASE1_CONSTITUENCIES
    ranked = sorted(_real_tipping_points(cs, use_literacy_effect), key=lambda ct: ct[1])
    remaining = float(budget_pct)
    targeted = []
    for c, tp in ranked:
        cost = tp * 100.0
        if cost <= remaining:
            targeted.append(c)
            remaining -= cost
    return AllocationResult(targeted=targeted, flipped=list(targeted),
                            budget_spent=float(budget_pct) - remaining,
                            budget_remaining=remaining)


def even_allocation(budget_pct, scope_max_tipping_pct=20.0, constituencies=None,
                    use_literacy_effect=False):
    """Spread `budget_pct` equally across every real seat within
    `scope_max_tipping_pct` of its own tipping point (the "seats worth
    contesting at all" scope a real strategist would plausibly use, not
    all 543 -- stated as the deliberate choice it is). A seat only flips
    if its equal share actually clears its own real tipping point; seats
    that don't are still reported as targeted, distinguishing "spent on"
    from "won". `use_literacy_effect=True` scopes and flips against the
    real literacy-adjusted cost, same as `greedy_allocation`."""
    cs = constituencies if constituencies is not None else ALL_PHASE1_CONSTITUENCIES
    in_scope = [(c, tp) for c, tp in _real_tipping_points(cs, use_literacy_effect)
               if tp * 100.0 <= scope_max_tipping_pct]
    if not in_scope:
        return AllocationResult(targeted=[], flipped=[], budget_spent=0.0,
                                budget_remaining=float(budget_pct))
    share = float(budget_pct) / len(in_scope)
    targeted = [c for c, _ in in_scope]
    flipped = [c for c, tp in in_scope if tp * 100.0 <= share]
    return AllocationResult(targeted=targeted, flipped=flipped,
                            budget_spent=float(budget_pct), budget_remaining=0.0)
