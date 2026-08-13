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
from .policies import (State, Greedy, ECI, Conservative, SortByE, Rollout,
                       ThompsonSampling, UCB)
from .domains import (agent_tools, adaptive_therapy, platform_trial,
                      design_space, DOMAIN_NOTES)
from .envs import ConsumableBanditEnv, MultiAgentConsumableEnv
from .scenarios import SCENARIOS, load, describe
from .simultaneous import (SimultaneousPool, planner_value_simultaneous,
                           decentralised_value_simultaneous, price_of_anarchy)
from .commitment import (TerminalCommitment, optimal_commitment_policy,
                         greedy_commitment_policy, edge_first_policy,
                         evaluate_commitment_policy)
from .engines import (LotkaVolterraTumour, derive_arm_parameters,
                      PlatformTrialEngine, derive_trial_parameters,
                      DesignSpaceEngine, derive_design_space_parameters)

__version__ = "0.2.0"
__all__ = ["ConsumableBandit", "run", "compare", "State", "Greedy", "ECI",
           "Conservative", "SortByE", "Rollout", "ThompsonSampling", "UCB",
           "agent_tools", "adaptive_therapy", "platform_trial", "design_space",
           "DOMAIN_NOTES",
           "LotkaVolterraTumour", "derive_arm_parameters",
           "PlatformTrialEngine", "derive_trial_parameters",
           "DesignSpaceEngine", "derive_design_space_parameters",
           "ConsumableBanditEnv", "MultiAgentConsumableEnv",
           "SCENARIOS", "load", "describe",
           "TerminalCommitment", "optimal_commitment_policy",
           "greedy_commitment_policy", "edge_first_policy",
           "evaluate_commitment_policy",
           "SimultaneousPool", "planner_value_simultaneous",
           "decentralised_value_simultaneous", "price_of_anarchy"]
