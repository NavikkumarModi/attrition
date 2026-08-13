"""ATTRITION quickstart.

Five minutes: build an environment, watch a zero-regret policy destroy value,
watch a corrected one not.

Run:  python examples/01_quickstart.py
"""
from attrition import ConsumableBandit, Greedy, ECI, compare, run


def main():
    print(__doc__)

    env_factory = lambda seed: ConsumableBandit.random(
        n=20, k_spread=1.5, delta=0.03, horizon=60, seed=seed)

    print("Comparing greedy against the externality-corrected index, 25 seeds:\n")
    results = compare(env_factory, [Greedy(), ECI()], seeds=25)
    for name, stats in results.items():
        print(f"  {name:>8}  value={stats['value']:7.3f}  "
              f"regret={stats['regret']:7.3f}")

    print("\nThe policy with WORSE regret has BETTER value. That's the whole point:")
    print("regret is measured against a benchmark the greedy policy itself has")
    print("already degraded.\n")

    print("Watching a single session step by step (first 8 steps):\n")
    trace = run(env_factory(0), Greedy(), log=True)
    for row in trace["log"][:8]:
        print(f"  t={row['t']:2d}  arm={row['arm']:2d}  "
              f"value={row['value']:6.3f}  regret={row['regret']:6.3f}  "
              f"alive={row['alive']:2d}")


if __name__ == "__main__":
    main()
