"""Load a population-simulation run from a JSON or YAML config file.

This is what lets a new domain, persona mix, and run be defined without
writing Python -- the `attrition` CLI (`cli.py`) is built entirely on this
module. Config keys map directly onto existing objects/functions:

    domain       a name already registered in attrition.scenarios.SCENARIOS,
                 or an inline {"v": [...], "p": [...], "e": [...]}
    delta, horizon, seed
                 ConsumableBandit/SimultaneousPool parameters; when `domain`
                 is a registered name, its own registered delta/horizon are
                 the default and these override them
    mode         "turn-taking" (default) or "simultaneous"
    population   list of {"persona": "<name in PHARMA_PERSONAS>"} or an
                 inline {"name", "role", "description", "risk_tolerance"}
    baselines    list of policy names to compare against, e.g. ["greedy", "eci"]
    trace_path, run_id
                 optional; if `trace_path` is set the CLI writes every
                 decision to a TraceStore there

A config file can only select `llm_client: "mock"` (the default) -- a real
client is a live Python callable, so wiring one has to happen in Python; see
`attrition/llm.py`'s `CallableLLMClient`.
"""

import json

import numpy as np

from .consumable import ConsumableBandit
from .llm import MockLLMClient
from .persona import Persona, PHARMA_PERSONAS
from .policies import Greedy, ECI, UCB, Conservative
from .population import Population
from .scenarios import get_arrays
from .simultaneous import SimultaneousPool

__all__ = ["load_config", "build_from_config", "BASELINE_POLICIES"]

BASELINE_POLICIES = {"greedy": Greedy, "eci": ECI, "ucb": UCB,
                     "conservative": lambda: Conservative(e_bound=2.0)}


def load_config(path):
    """Read a JSON or YAML config file (by extension) into a plain dict."""
    path = str(path)
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "reading a .yaml config requires pyyaml: "
                "pip install attrition[cli] (or write the config as .json "
                "instead, which needs no extra dependency)") from exc
        with open(path) as f:
            return yaml.safe_load(f)
    with open(path) as f:
        return json.load(f)


def _build_persona(entry):
    if "persona" in entry:
        name = entry["persona"]
        if name not in PHARMA_PERSONAS:
            raise KeyError(f"unknown persona {name!r}; "
                           f"available: {sorted(PHARMA_PERSONAS)}")
        return PHARMA_PERSONAS[name]
    return Persona(name=entry["name"], role=entry.get("role", "agent"),
                  description=entry.get("description", ""),
                  risk_tolerance=float(entry.get("risk_tolerance", 0.5)))


def build_from_config(cfg):
    """Translate a config dict into `(env_or_pool, population, baselines, meta)`.

    `env_or_pool` is a `ConsumableBandit` (mode "turn-taking") or a
    `SimultaneousPool` (mode "simultaneous"), ready for
    `simulate_population`/`simulate_population_simultaneous` respectively.
    """
    domain = cfg["domain"]
    seed = int(cfg.get("seed", 0))
    if isinstance(domain, str):
        v, p, e, kw = get_arrays(domain)
    else:
        v = np.asarray(domain["v"], float)
        p = np.asarray(domain["p"], float)
        e = np.asarray(domain["e"], float)
        kw = {}
    delta = float(cfg.get("delta", kw.get("delta", 0.05)))
    horizon = int(cfg.get("horizon", kw.get("horizon", 50)))

    personas = [_build_persona(entry) for entry in cfg.get("population", [])]
    if not personas:
        raise ValueError("config must define at least one 'population' entry")
    client = MockLLMClient(seed=seed)
    population = Population.from_personas(personas, client=client)

    mode = cfg.get("mode", "turn-taking")
    if mode == "simultaneous":
        env_or_pool = SimultaneousPool(v, p, e, delta=delta, horizon=horizon,
                                       n_agents=len(personas), seed=seed)
    elif mode == "turn-taking":
        env_or_pool = ConsumableBandit(v, p, e, delta=delta, horizon=horizon,
                                       seed=seed)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected "
                         f"'turn-taking' or 'simultaneous'")

    baselines = []
    for name in cfg.get("baselines", []):
        if name not in BASELINE_POLICIES:
            raise KeyError(f"unknown baseline {name!r}; "
                           f"available: {sorted(BASELINE_POLICIES)}")
        baselines.append(BASELINE_POLICIES[name]())

    domain_name = domain if isinstance(domain, str) else "inline"
    meta = {"mode": mode, "domain": domain_name, "delta": delta,
            "horizon": horizon, "seed": seed,
            "trace_path": cfg.get("trace_path"),
            "run_id": cfg.get("run_id", domain_name)}
    return env_or_pool, population, baselines, meta
