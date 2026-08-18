"""The no-Python path: everything examples/04_pharma_population.py builds by
hand -- domain, personas, mode, baselines -- defined instead in
examples/antibiotic-stewardship.json and run through attrition.config.

Equivalent from a shell, with no Python at all (after `pip install -e .`):

    attrition run examples/antibiotic-stewardship.json --trace /tmp/run.sqlite3

Run:  python examples/05_config_driven_run.py
"""
from attrition import TraceStore, build_from_config, load_config
from attrition.population import simulate_population_simultaneous


def main():
    print(__doc__)
    cfg = load_config("examples/antibiotic-stewardship.json")
    pool, population, baselines, meta = build_from_config(cfg)
    print(f"Loaded config: domain={meta['domain']}  mode={meta['mode']}  "
          f"agents={len(population)}")

    store = TraceStore("/tmp/attrition_config_demo.sqlite3")
    result = simulate_population_simultaneous(
        pool, population, rounds=meta["horizon"], trace_store=store,
        run_id=meta["run_id"])
    print(f"system_value={result['system_value']:.3f}  "
          f"system_regret={result['system_regret']:.3f}")

    rows = store.read(meta["run_id"])
    print(f"{len(rows)} trace rows persisted to {store.path} "
          f"(query with TraceStore.read/.to_dataframe)")
    store.close()


if __name__ == "__main__":
    main()
