"""Command-line entry point: ``attrition run/list-domains/describe``.

Lets a domain, persona mix, and population run be defined in a JSON/YAML
config file and executed with no Python -- see `attrition/config.py` for the
config schema. Registered as `[project.scripts]` in `pyproject.toml`, so after
`pip install -e .` the `attrition` command is on PATH; it also works
unregistered as ``python -m attrition.cli ...``.
"""

import argparse
import sys

from .config import build_from_config, load_config
from .consumable import run as run_episode
from .population import simulate_population, simulate_population_simultaneous
from .scenarios import describe as describe_scenarios
from .simultaneous import decentralised_value_simultaneous, planner_value_simultaneous
from .trace import TraceStore

__all__ = ["main", "build_parser"]


def _cmd_list_domains(args):
    describe_scenarios()
    return 0


def _cmd_describe(args):
    describe_scenarios(args.domain)
    return 0


def _cmd_run(args):
    cfg = load_config(args.config)
    env_or_pool, population, baselines, meta = build_from_config(cfg)
    trace_path = args.trace or meta["trace_path"]
    store = TraceStore(trace_path) if trace_path else None
    run_id = meta["run_id"]

    print(f"domain={meta['domain']}  mode={meta['mode']}  "
          f"delta={meta['delta']}  horizon={meta['horizon']}  "
          f"agents={len(population)}")

    if meta["mode"] == "turn-taking":
        result = simulate_population(env_or_pool, population,
                                     trace_store=store, run_id=run_id)
        print(f"  {'population':>20}  system_value={result['system_value']:.3f}  "
              f"system_regret={result['system_regret']:.3f}")
        for pol in baselines:
            r = run_episode(env_or_pool, pol, seed=meta["seed"])
            print(f"  {pol.name:>20}  value={r['value']:.3f}  "
                  f"regret={r['regret']:.3f}")
    else:
        rounds = meta["horizon"]
        result = simulate_population_simultaneous(
            env_or_pool, population, rounds=rounds,
            max_workers=args.max_workers, trace_store=store, run_id=run_id)
        print(f"  {'population':>20}  system_value={result['system_value']:.3f}  "
              f"system_regret={result['system_regret']:.3f}")
        v, p, e, m = env_or_pool.v, env_or_pool.p, env_or_pool.e, env_or_pool.m
        planner = planner_value_simultaneous(v, p, e, meta["delta"], rounds, m)
        print(f"  {'planner (optimum)':>20}  value={planner:.3f}")
        for rule in ("greedy", "eci"):
            dec = decentralised_value_simultaneous(v, p, e, meta["delta"], rounds,
                                                    m, rule=rule)
            print(f"  {'decentralised ' + rule:>20}  value={dec:.3f}")
        if baselines:
            print("  (baselines: config only compares against classical "
                  "policies in mode='turn-taking'; simultaneous mode "
                  "compares against the planner/decentralised references "
                  "above instead)")

    if store is not None:
        print(f"trace written to {trace_path} (run_id={run_id!r})")
        store.close()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="attrition", description="ATTRITION population simulator CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a population simulation from a config file")
    p_run.add_argument("config", help="path to a .json or .yaml config file")
    p_run.add_argument("--trace", default=None,
                       help="path to write a TraceStore (overrides trace_path "
                            "in the config)")
    p_run.add_argument("--max-workers", type=int, default=None,
                       help="concurrent agent decisions per round "
                            "(mode='simultaneous' only)")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list-domains",
                            help="list every registered scenario/domain")
    p_list.set_defaults(func=_cmd_list_domains)

    p_desc = sub.add_parser("describe", help="describe one registered domain")
    p_desc.add_argument("domain")
    p_desc.set_defaults(func=_cmd_describe)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
