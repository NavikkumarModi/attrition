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
from .real_data import (derive_real_amr_parameters, derive_real_cmc_parameters,
                        derive_real_fisheries_parameters)

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

# ------------------------------------------------ real-data-grounded (Tier 3)
# Registered via the extension point, not the literal above -- see
# attrition/real_data.py's SOURCES dict for what's actually measured here vs.
# a documented proxy. Unlike every scenario above, at least one of (v, p, e)
# in each of these comes from a real, cited, external dataset, not this repo.
SCENARIOS.register(
    "antibiotic-stewardship-real",
    description=("Real WHO GLASS-fed surveillance data: 1,005 country/year "
                "MRSA and E. coli resistance measurements, each one arm. "
                "p is the real reported resistance proportion; v and e are "
                "documented proxies -- see real_data.SOURCES['who_amr']."),
    phenomenon="negative control, using real measured resistance rates",
    expected_behaviour=("kappa = p*e is real-dispersed (std > 0) but, by "
                        "construction of the v=1-p, e=p proxy, is a "
                        "deterministic monotone function of v -- corr(v, "
                        "kappa) approx -0.96, checked not assumed. That is "
                        "the ALIGNED regime (see platform-trial/"
                        "aligned-control): greedy and ECI coincide here, "
                        "not because real antibiotic resistance behaves "
                        "this way, but as a mechanical consequence of this "
                        "proxy formula -- see derive_real_amr_parameters's "
                        "docstring. n defaults to 200 of the ~1,005 "
                        "available real rows -- a genuinely large-N "
                        "scenario (this repo's previous largest was n=20), "
                        "even though the headline effect doesn't appear at "
                        "these default parameters."),
    agents=1,
    build=lambda: (derive_real_amr_parameters(indicator="both", n=200, seed=0)[:3],
                  dict(delta=0.05, horizon=40)),
)

SCENARIOS.register(
    "design-space-real",
    description=("Real openFDA drug-recall data: 9 manufacturing failure "
                "categories (CGMP, sterility, contamination, ...) covering "
                "15,556 real recall records. p is the real fraction of each "
                "category's recalls classified Class I (most severe); v and "
                "e are documented proxies -- see real_data.SOURCES['fda_cmc']."),
    phenomenon="zero-regret ruin, using real FDA severity classifications",
    expected_behaviour=("kappa is dispersed from ~0 (CGMP, almost never "
                        "Class I) to >1.5 (microbial contamination, 66% "
                        "Class I) -- a real, not tuned, spread; corr(v, "
                        "kappa) approx -0.20, checked not assumed. Unlike "
                        "the synthetic design-space scenario (whose gap is "
                        "tuned to 17-107%), this real spread is mild and "
                        "close to the alignment boundary at n=9, so greedy "
                        "still records zero private regret but the value "
                        "gap to ECI is small (a few percent) and comparable "
                        "to Monte-Carlo noise at moderate seed counts -- "
                        "reported honestly rather than run until a bigger "
                        "gap appears by chance."),
    agents=1,
    build=lambda: (derive_real_cmc_parameters()[:3], dict(delta=0.6, horizon=8)),
)

SCENARIOS.register(
    "fisheries-commons-real",
    description=("Real NOAA commercial landings: 16 U.S. fish and shellfish "
                "species, most with 75 years of real annual data (1950-2024). "
                "p is the real year-over-year landings-decline frequency; v "
                "and e are documented proxies from real price and total-"
                "value data -- see real_data.SOURCES['noaa_fisheries']."),
    phenomenon="zero-regret ruin, in the CONFLICT regime, using real landings data",
    expected_behaviour=("Unlike antibiotic-stewardship-real (mechanically "
                        "ALIGNED by its proxy formula) and design-space-real "
                        "(mild, near the alignment boundary), this one lands "
                        "in the CONFLICT regime from the real data itself: "
                        "corr(v, kappa) = +0.74, checked not assumed -- the "
                        "highest-price, highest-total-value species (lobster, "
                        "sea scallop) also have the highest kappa. At "
                        "delta=0.15, horizon=12, greedy records zero private "
                        "regret while losing ~20% of system value versus ECI "
                        "(checked at 200 seeds); the exact percentage moves "
                        "with delta/horizon, the sign and the mechanism do "
                        "not."),
    agents=1,
    build=lambda: (derive_real_fisheries_parameters()[:3],
                  dict(delta=0.15, horizon=12)),
)


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
