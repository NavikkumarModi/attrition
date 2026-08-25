"""Election track, Phase 1: a population engine, not a predictor.

Ten real Lok Sabha constituencies (2019 General Election, Trivedi Centre
for Political Data / LokDhaba), deliberately different environments --
see `attrition.election.lokdhaba.SOURCE` for provenance and
`attrition.election.population`'s module docstring for the
observed-vs-synthetic-latent discipline this package enforces.

This is a subpackage, not a `attrition.scenarios.SCENARIOS` entry -- see
`RESEARCH_PROGRAMME.md` and the domain-extension discussion: the population
layer here (millions of records, real data pipelines, no single (v, p, e)
triple) doesn't fit the scenario-registry contract every other domain in
this repo terminates at. Campaign/persuasion agents that draw on this
population and *do* reduce to that contract are Phase 2+ and unbuilt.

No LLM calls anywhere in this package. No campaign, persuasion, or theorem
claim is made here -- this is the population substrate those would be
built on, validated on its own terms first.
"""

from .lokdhaba import SOURCE, load_constituencies, get_constituency
from .population import (SyntheticVoter, Constituency, build_constituency,
                         synthetic_voter_population, validate_population,
                         ALL_PHASE1_CONSTITUENCIES)
from .persuasion import (real_margin_pct, persuadable_fraction_from_margin,
                         derive_persuasion_parameters_from_constituency,
                         PHASE1_PERSUADABLE_FRACTIONS)
from .identification import identification_floor, PHASE1_IDENTIFICATION_FLOORS
from .agents import (BehaviouralVoter, simulate_level1_contact,
                     simulate_level2_contact, REALISTIC_PERSUASION_CEILING_PCT)
from .network import simulate_level1_contact_with_network
from .outcome import Outcome, recompute_outcome, tipping_point
from .geo import load_boundaries, geojson_feature_collection
from .strategist import greedy_allocation, even_allocation, AllocationResult
from .forecast import (match_seats, national_swing, predict_seat_winner,
                       leave_one_state_out_backtest, MIN_SEATS_FOR_SWING)
from .policy_training import GradientBanditPolicy, train_gradient_bandit

__all__ = ["SOURCE", "load_constituencies", "get_constituency",
          "SyntheticVoter", "Constituency", "build_constituency",
          "synthetic_voter_population", "validate_population",
          "ALL_PHASE1_CONSTITUENCIES",
          "real_margin_pct", "persuadable_fraction_from_margin",
          "derive_persuasion_parameters_from_constituency",
          "PHASE1_PERSUADABLE_FRACTIONS",
          "identification_floor", "PHASE1_IDENTIFICATION_FLOORS",
          "BehaviouralVoter", "simulate_level1_contact", "simulate_level2_contact",
          "REALISTIC_PERSUASION_CEILING_PCT",
          "simulate_level1_contact_with_network",
          "Outcome", "recompute_outcome", "tipping_point",
          "load_boundaries", "geojson_feature_collection",
          "greedy_allocation", "even_allocation", "AllocationResult",
          "match_seats", "national_swing", "predict_seat_winner",
          "leave_one_state_out_backtest", "MIN_SEATS_FOR_SWING",
          "GradientBanditPolicy", "train_gradient_bandit"]
