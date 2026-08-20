"""Bandits with consumable action sets.

Actions that destroy the arms they are taken on, and impose a permanent
externality on everything that remains.

Main results
------------
T1  Greedy is optimal iff kappa = p * e is constant across arms.
T2b In pure sequencing the optimal order is ascending in e; values are
    irrelevant.
T3  The externality cannot be estimated to arbitrary precision: the error floor
    2*sigma/sqrt(min(T, sum 1/p_j)) is independent of the horizon.
T4  A zero-regret policy can incur value loss m*delta*E, unbounded in the pool
    size and the externality magnitude. No-regret is not no-harm.

Quick start
-----------
>>> from attrition import ConsumableBandit, Greedy, ECI, compare
>>> f = lambda s: ConsumableBandit.random(n=20, k_spread=1.5, delta=0.03,
...                                       horizon=60, seed=s)
>>> compare(f, [Greedy(), ECI()], seeds=20)          # doctest: +SKIP
"""

from .consumable import ConsumableBandit, run, compare
from .policies import (State, Greedy, ECI, SECI, Conservative, SortByE,
                       Rollout, ThompsonSampling, UCB)
from .domains import (agent_tools, adaptive_therapy, platform_trial,
                      design_space, DOMAIN_NOTES)
from .envs import ConsumableBanditEnv, MultiAgentConsumableEnv
from .scenarios import (SCENARIOS, load, describe, ScenarioRegistry, from_arrays,
                        get_arrays)
from .simultaneous import (SimultaneousPool, planner_value_simultaneous,
                           decentralised_value_simultaneous, price_of_anarchy)
from .commitment import (TerminalCommitment, optimal_commitment_policy,
                         expand_outward_policy, greedy_commitment_policy,
                         edge_first_policy, evaluate_commitment_policy)
from .engines import (LotkaVolterraTumour, derive_arm_parameters,
                      PlatformTrialEngine, derive_trial_parameters,
                      DesignSpaceEngine, derive_design_space_parameters,
                      derive_antibiotic_parameters)
from .llm import LLMClient, MockLLMClient, CallableLLMClient
from .persona import Persona, PHARMA_PERSONAS
from .llm_policy import LLMPolicy
from .population import (Population, simulate_population,
                         simulate_population_simultaneous,
                         compare_population_to_baselines)
from .trace import TraceStore
from .viz import plot_system_value_over_time, plot_burden_over_time
from .config import load_config, build_from_config
from .network import AgentGraph
from .dashboard import render_dashboard
from .real_data import (SOURCES, derive_real_amr_parameters,
                        derive_real_cmc_parameters, derive_real_fisheries_parameters)

__version__ = "0.2.0"
__all__ = ["ConsumableBandit", "run", "compare", "State", "Greedy", "ECI",
           "Conservative", "SortByE", "Rollout", "ThompsonSampling", "UCB", "SECI",
           "agent_tools", "adaptive_therapy", "platform_trial", "design_space",
           "DOMAIN_NOTES",
           "LotkaVolterraTumour", "derive_arm_parameters",
           "PlatformTrialEngine", "derive_trial_parameters",
           "DesignSpaceEngine", "derive_design_space_parameters",
           "derive_antibiotic_parameters",
           "ConsumableBanditEnv", "MultiAgentConsumableEnv",
           "SCENARIOS", "load", "describe", "ScenarioRegistry", "from_arrays",
           "get_arrays",
           "TerminalCommitment", "optimal_commitment_policy",
           "expand_outward_policy",
           "greedy_commitment_policy", "edge_first_policy",
           "evaluate_commitment_policy",
           "SimultaneousPool", "planner_value_simultaneous",
           "decentralised_value_simultaneous", "price_of_anarchy",
           "LLMClient", "MockLLMClient", "CallableLLMClient",
           "Persona", "PHARMA_PERSONAS", "LLMPolicy",
           "Population", "simulate_population",
           "simulate_population_simultaneous",
           "compare_population_to_baselines", "TraceStore",
           "plot_system_value_over_time", "plot_burden_over_time",
           "load_config", "build_from_config",
           "AgentGraph", "render_dashboard",
           "SOURCES", "derive_real_amr_parameters", "derive_real_cmc_parameters",
           "derive_real_fisheries_parameters"]
