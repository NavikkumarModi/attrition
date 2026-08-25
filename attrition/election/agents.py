"""Level 1 and Level 2 of the population engine's three-tier plan
(FINDINGS.md's election-track sections; Level 0 is `population.py`'s
`SyntheticVoter`, sampled once from real aggregates and never updated).

Level 0 never responds to anything -- it is a fixed draw. Levels 1 and 2
exist for the same reason the plan gave them: something has to happen to a
voter (a campaign contacts them) before there is anything to model a
response *to*. Both tiers below respond to the same event -- a campaign
choosing a contact intensity this round -- so they are directly comparable,
not two unrelated mechanisms.

Level 1 -- BehaviouralVoter (no LLM, cheap, the tier meant to scale)
---------------------------------------------------------------------
Each voter accrues persuasion pressure toward the contacting campaign using
the SAME rho-decay update rule already proven for `DegradingPool`
(`attrition.simultaneous`) and reused for the same reason
`derive_campaign_parameters` reuses `LotkaVolterraTumour`: it is the
correct mechanism, relabelled, not a new one invented for this tier.
    pressure(t+1) = rho * pressure(t) + (1 - rho) * intensity(t)
A voter switches stochastically with probability `pressure * (1 -
switch_resistance)`, where `switch_resistance` is drawn per voter from a
Beta distribution centred on the constituency's real, margin-anchored
persuadable fraction (`attrition.election.persuasion`) -- so a voter
population inherits the same real anchor the persuasion theorem does, not
a second, disconnected one.

Level 2 -- CognitiveVoter (LLM-driven, small numbers, the tier meant to be
deep, not wide)
---------------------------------------------------------------------------
A handful of `VOTER_PERSONAS` (`attrition.persona`) reason about the same
contact event via the SAME `LLMPolicy`/`State`/`MockLLMClient` machinery
already proven for campaigns and prescribers -- the decision is framed as
a genuine two-arm choice (stay vs. switch), not a bespoke prompt format,
so `MockLLMClient` (which parses `LLMPolicy`'s specific prompt shape)
works unmodified and no API key is required by default. The `v`/`p`/`e`
values fed to that choice are given an explicit, honest interpretation
below -- stated, not left to be inferred from bandit-domain vocabulary
that doesn't literally apply to a voter's decision.
"""

import numpy as np

from ..llm import MockLLMClient
from ..llm_policy import LLMPolicy
from ..policies import State
from ..persona import VOTER_PERSONAS
from .persuasion import persuadable_fraction_from_margin
from .population import build_constituency

__all__ = ["BehaviouralVoter", "simulate_level1_contact",
          "simulate_level2_contact"]


class BehaviouralVoter:
    """One Level-1 voter: a persuasion-pressure state and a resistance
    draw, nothing else. `pressure` and `switched` are the only mutable
    state -- see module docstring for the update rule."""

    __slots__ = ("switch_resistance", "pressure", "switched")

    def __init__(self, switch_resistance):
        self.switch_resistance = float(switch_resistance)
        self.pressure = 0.0
        self.switched = False


def simulate_level1_contact(constituency, intensity_sequence, rho=0.7,
                            n_voters=2000, seed=0):
    """Run `n_voters` Level-1 voters through a sequence of campaign-contact
    rounds. `intensity_sequence` is contact intensity in [0, 1] per round
    (e.g. one of `derive_campaign_parameters`'s `intensity` levels, chosen
    by a Level-2/campaign-persuasion agent each round -- not generated
    here, this function only consumes the sequence).

    `switch_resistance` per voter is drawn from Beta(a, b) centred on
    `1 - persuadable_fraction_from_margin(constituency)` -- so a
    constituency with a real close margin (high persuadable fraction, low
    average resistance) produces a population that is, on average,
    genuinely easier to move, and a landslide seat produces one that
    isn't, without hand-tuning per constituency.

    Returns the per-round switched fraction (list, one entry per round)
    and the final switched fraction.
    """
    if isinstance(constituency, str):
        constituency = build_constituency(constituency)
    persuadable = persuadable_fraction_from_margin(constituency)
    mean_resistance = 1.0 - persuadable
    rng = np.random.default_rng(seed)
    # Beta(a, b) with mean = mean_resistance, concentration held fixed at 8
    # (moderate spread around the mean, not degenerate) -- a modelling
    # choice, stated as one.
    conc = 8.0
    a = max(mean_resistance * conc, 0.5)
    b = max((1.0 - mean_resistance) * conc, 0.5)
    voters = [BehaviouralVoter(r) for r in rng.beta(a, b, size=n_voters)]

    switched_fraction_by_round = []
    for intensity in intensity_sequence:
        for v in voters:
            v.pressure = rho * v.pressure + (1.0 - rho) * float(intensity)
            # Deterministic threshold-crossing, not a per-round stochastic
            # re-test: `switched` is exactly `pressure > switch_resistance`.
            # An earlier version re-drew `switched` via a Bernoulli trial
            # every round, which created an unintended ratchet -- once
            # pressure was high, the trial kept re-confirming True almost
            # every round, so the sub-1% reversion chance never had a real
            # opportunity to matter. Found by connecting this to
            # `outcome.py` and seeing 88%+ cumulative switching after only
            # 9 rounds, wildly outside real campaign-swing magnitudes --
            # not tuned away, fixed at the mechanism. `pressure` is
            # bounded in [0, 1] by construction (a convex combination of
            # its previous value and `intensity`), so a voter only ever
            # switches if accumulated pressure genuinely exceeds their own
            # resistance, and un-switches cleanly if pressure decays back
            # below it.
            v.switched = v.pressure > v.switch_resistance
        switched_fraction_by_round.append(
            sum(v.switched for v in voters) / n_voters)

    return switched_fraction_by_round, switched_fraction_by_round[-1]


def simulate_level2_contact(constituency, intensity_sequence, personas=None,
                            client=None):
    """The same contact sequence, run through a small set of LLM-driven
    voter personas instead of Level 1's stochastic rule.

    Each round is framed as a real two-arm `State`/`LLMPolicy` choice:
      arm 0 ("stay")   v_hat=0.5 -- a fixed baseline utility for keeping a
                       settled preference, deliberately not zero, so
                       "stay" is a genuine option, not a null one.
      arm 1 ("switch")  v_hat=intensity(t) -- the immediate case for
                       switching scales with how hard the campaign is
                       contacting this round, which is the honest
                       interpretation of what a contact's "value" to the
                       voter's decision even means here.
      p, e            interpreted as the cost of over-committing: e is set
                       to the persona's own risk_tolerance (a stand-in for
                       how much future flexibility a hasty switch costs a
                       more loyalist persona), p is set to `intensity(t)`
                       itself (repeated high-intensity contact is what
                       risks "locking in" a switch). Stated as an explicit
                       reinterpretation, not a literal bandit-arm-death
                       claim about voters -- see module docstring.

    Returns {persona_name: [switch_decision_per_round]} where each entry
    is True (switched this round) or False (stayed).
    """
    if isinstance(constituency, str):
        constituency = build_constituency(constituency)
    personas = list(VOTER_PERSONAS.values()) if personas is None else personas
    client = client if client is not None else MockLLMClient(seed=0)

    horizon = len(intensity_sequence)
    results = {}
    for persona in personas:
        policy = LLMPolicy(persona, client=client)
        decisions = []
        for t, intensity in enumerate(intensity_sequence):
            state = State(
                available=[0, 1], t=t, horizon=horizon,
                v_hat=[0.5, float(intensity)],
                p=[0.0, float(intensity)],
                e=[0.0, float(persona.risk_tolerance)],
                delta=1.0,
            )
            arm = policy.select(state)
            decisions.append(arm == 1)
        results[persona.name] = decisions
    return results
