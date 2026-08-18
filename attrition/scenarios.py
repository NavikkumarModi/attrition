"""Scenario packs: named, reproducible configurations.

Each scenario is a fixed problem instance with a stated purpose, so results are
comparable across papers and implementations. A scenario returns a builder that
produces a fresh environment for a given seed.

    from attrition.scenarios import SCENARIOS, load
    env = load("shared-quota", seed=0)

Every scenario records which phenomenon it is meant to exhibit, and
`expected_behaviour` states what the theory predicts, so a scenario that stops
behaving as documented is a detectable regression rather than a silent drift.

`SCENARIOS` is the extension point: `SCENARIOS.register(...)` or the
convenience `from_arrays(...)` add a new named, discoverable domain -- pharma
or otherwise -- without editing this file. Anything with a consumable-choice
structure (inventory allocation, hiring funnels, credit-limit decisions,
ad-budget pacing, ...) registers the same way the built-in scenarios do.
"""

import numpy as np

from .envs import ConsumableBanditEnv, MultiAgentConsumableEnv
from .engines import (derive_arm_parameters, derive_trial_parameters,
                      derive_design_space_parameters, derive_antibiotic_parameters)

__all__ = ["SCENARIOS", "load", "describe", "ScenarioRegistry", "from_arrays",
           "get_arrays"]


class ScenarioRegistry(dict):
    """Dict of named scenario specs, plus a `.register` convenience method.

    Fully dict-compatible (`SCENARIOS[name]`, `.items()`, `sorted(SCENARIOS)`,
    `in`) -- a drop-in for the plain dict literal it replaces. `.register` is
    the only addition, and is what external code calls to add a new domain.
    """

    def register(self, name, description, phenomenon, expected_behaviour,
                build, agents=1):
        self[name] = {"description": description, "phenomenon": phenomenon,
                     "expected_behaviour": expected_behaviour,
                     "agents": agents, "build": build}
        return self[name]


def _agent_tools():
    roster = [
        ("premium-search",   1.15, 0.15, 0.2),
        ("bulk-search",      1.05, 0.75, 2.4),
        ("premium-codegen",  1.00, 0.10, 0.1),
        ("bulk-scraper",     0.98, 0.85, 2.8),
        ("internal-db",      0.80, 0.05, 0.0),
        ("cached-lookup",    0.62, 0.02, 0.0),
    ]
    return (np.array([r[1] for r in roster]),
            np.array([r[2] for r in roster]),
            np.array([r[3] for r in roster]),
            [r[0] for r in roster])


SCENARIOS = ScenarioRegistry({
    # ------------------------------------------------------ agent ecosystems
    "shared-quota": {
        "description": "Orchestrator routing across API tools that share upstream "
                       "quota. Burning a bulk tool throttles the rest.",
        "phenomenon": "zero-regret ruin",
        "expected_behaviour": "greedy records zero private regret and loses "
                              "~30% of system value",
        "agents": 1,
        "build": lambda: (_agent_tools()[:3], dict(delta=0.06, horizon=40)),
    },
    "shared-quota-competing": {
        "description": "Two orchestrators drawing on the same quota pool. "
                       "Note the burden falls on both, including the agent "
                       "that creates it, so this is not a classical commons.",
        "phenomenon": "free-riding under a shared burden",
        "expected_behaviour": "NO price of anarchy: by Theorem 6 sequential "
                              "agents behave exactly like one learner over m*T "
                              "pulls. The genuine effect is free-riding -- an "
                              "agent charging delta*kappa/m instead of "
                              "delta*kappa loses system value monotonically in m",
        "agents": 2,
        "build": lambda: (_agent_tools()[:3], dict(delta=0.06, horizon=40)),
    },

    # ------------------------------------------------------------- oncology
    "adaptive-therapy": {
        "description": "Dose selection under Lotka-Volterra competition between "
                       "drug-sensitive and drug-resistant tumour compartments.",
        "phenomenon": "competitive release; MTD is exactly greedy",
        "expected_behaviour": "greedy (MTD) loses 27-121% of system value with "
                              "zero private regret; corr(v, kappa) > 0",
        "agents": 1,
        "build": lambda: (derive_arm_parameters(engine_kwargs={"dt": 5.0},
                                                s0_sd=0.26)[:3],
                          dict(delta=0.30, horizon=8)),
    },

    # --------------------------------------------------------- clinical ops
    "platform-trial": {
        "description": "Shared-control platform trial with staggered entry. "
                       "Dropping an arm fragments the control stream.",
        "phenomenon": "value-externality ALIGNMENT -- the negative control",
        "expected_behaviour": "greedy is optimal (0% gap) because corr(v, kappa) "
                              "< 0: promising arms are also the ones never dropped",
        "agents": 1,
        "build": lambda: (derive_trial_parameters(periods=5)[:3],
                          dict(delta=0.4, horizon=8)),
    },

    # --------------------------------------------------------------- process
    "design-space": {
        "description": "CMC process development. An out-of-spec run truncates the "
                       "defensible filing envelope for every later setting.",
        "phenomenon": "zero-regret ruin under regulatory lock-in",
        "expected_behaviour": "greedy loses 17-107% with zero private regret",
        "agents": 1,
        "build": lambda: (derive_design_space_parameters(n_settings=6)[:3],
                          dict(delta=0.6, horizon=8)),
    },

    # ------------------------------------------------------------ pharma pop
    "antibiotic-stewardship": {
        "description": "Several prescribers drawing on one population's "
                       "antimicrobial susceptibility. Choosing broader-"
                       "spectrum coverage clears the current patient's "
                       "infection more reliably but selects for resistance "
                       "that degrades future treatment options for everyone.",
        "phenomenon": "zero-regret ruin under a shared resistance pool",
        "expected_behaviour": "each prescriber's greedy (broadest-coverage) "
                              "choice is privately zero-regret, with kappa "
                              "dispersed across spectrum choices exactly as "
                              "in adaptive-therapy. Turn-taking here "
                              "(Theorem 6) has no genuine price of anarchy; "
                              "see attrition.population and "
                              "examples/04_pharma_population.py for the "
                              "simultaneous-action version where one exists.",
        "agents": 3,
        "build": lambda: (derive_antibiotic_parameters()[:3],
                          dict(delta=0.35, horizon=9)),
    },

    # -------------------------------------------------------------- stress
    "high-dispersion": {
        "description": "Synthetic stress case with large kappa dispersion.",
        "phenomenon": "worst-case separation",
        "expected_behaviour": "greedy value can go negative while the optimum "
                              "stays positive",
        "agents": 1,
        "build": lambda: ((np.array([1.0] + [0.99]*8),
                           np.ones(9),
                           np.array([2.0] + [0.0]*8)),
                          dict(delta=1.0, horizon=10)),
    },
    "aligned-control": {
        "description": "Synthetic negative control: kappa is dispersed but "
                       "negatively correlated with value.",
        "phenomenon": "alignment implies greedy is already optimal",
        "expected_behaviour": "greedy gap is ~0 despite dispersed kappa",
        "agents": 1,
        "build": lambda: ((np.linspace(1.2, 0.4, 6),
                           np.linspace(0.1, 0.9, 6),
                           np.linspace(0.1, 2.0, 6)),
                          dict(delta=0.3, horizon=8)),
    },
})


def from_arrays(v, p, e, name="custom", agents=1, delta=0.05, horizon=50,
                description=None, phenomenon="user-supplied",
                expected_behaviour="not characterised -- user-supplied domain"):
    """Register a scenario directly from arrays.

    The extension point for any consumable-action-set domain outside this
    package's built-in presets -- no new Python function required, just three
    arrays:

        from attrition.scenarios import from_arrays, load
        from_arrays(v=[...], p=[...], e=[...], name="my-domain", agents=3)
        env = load("my-domain", seed=0)

    `v`, `p`, `e` are copied into the registered build closure, so later
    mutating the arrays passed in does not change the registered scenario.
    """
    v = np.asarray(v, float).copy()
    p = np.asarray(p, float).copy()
    e = np.asarray(e, float).copy()
    if description is None:
        description = (f"user-supplied domain ({len(v)} arms, {agents} "
                       f"agent{'s' if agents > 1 else ''})")
    return SCENARIOS.register(
        name, description=description, phenomenon=phenomenon,
        expected_behaviour=expected_behaviour,
        build=lambda: ((v, p, e), dict(delta=delta, horizon=horizon)),
        agents=agents)


def get_arrays(name):
    """Return a scenario's raw `(v, p, e, kw)`, without wrapping it in a Gym
    env. `load()` wraps the same build in `ConsumableBanditEnv`/
    `MultiAgentConsumableEnv` (envs.py); callers that instead want
    `ConsumableBandit`/`SimultaneousPool` -- e.g. `attrition.population`,
    `attrition.config` -- need the arrays directly.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; "
                       f"available: {sorted(SCENARIOS)}")
    spec = SCENARIOS[name]
    (v, p, e), kw = spec["build"]()
    v = np.asarray(v, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1.0)
    e = np.asarray(e, float)
    return v, p, e, dict(kw)


def load(name, seed=0, **overrides):
    """Build a scenario environment. Multi-agent scenarios return the parallel env."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; "
                       f"available: {sorted(SCENARIOS)}")
    spec = SCENARIOS[name]
    (v, p, e), kw = spec["build"]()
    kw = {**kw, **overrides}
    v = np.asarray(v, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1.0)
    e = np.asarray(e, float)
    if spec["agents"] > 1:
        return MultiAgentConsumableEnv(v, p, e, n_agents=spec["agents"],
                                       seed=seed, **kw)
    return ConsumableBanditEnv(v, p, e, seed=seed, **kw)


def describe(name=None):
    """Print scenario documentation."""
    names = sorted(SCENARIOS) if name is None else [name]
    for k in names:
        s = SCENARIOS[k]
        print(f"{k}  ({s['agents']} agent{'s' if s['agents'] > 1 else ''})")
        print(f"  {s['description']}")
        print(f"  phenomenon: {s['phenomenon']}")
        print(f"  expected:   {s['expected_behaviour']}")
        print()
