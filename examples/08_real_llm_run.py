"""Run the pharma population against a REAL Anthropic model instead of
MockLLMClient -- the one validation this repo's LLM-population layer has
never had. Every result in the README, the dashboard, and examples/04-07
comes from a hand-written offline heuristic; this script is the first one
that asks an actual model what it would do.

Requires a real, separately-billed Anthropic API key -- a claude.ai Pro
subscription does NOT include API access. Get one at console.anthropic.com,
then set it as an environment variable (never hardcode it, never paste it
into a chat or commit it):

    export ANTHROPIC_API_KEY=sk-ant-...

This costs real money. Defaults are kept small (5 rounds x 3 seeds = 15
calls, on Haiku 4.5, the cheapest current model) so a first run costs a
fraction of a cent. The script prints the exact call count and asks for
confirmation before spending anything. Scale up with --agents/--rounds/
--seeds/--model once you've seen it work.

Install:  pip install "attrition[llm]"
Run:      python examples/08_real_llm_run.py
"""

import argparse
import os
import sys

from attrition import (CallableLLMClient, ConsumableBandit, ECI, Greedy,
                       MockLLMClient, PHARMA_PERSONAS, Population,
                       compare_population_to_baselines,
                       derive_antibiotic_parameters)


def _build_anthropic_client(model):
    try:
        import anthropic
    except ImportError:
        print("The 'anthropic' package isn't installed.\n"
              "  pip install \"attrition[llm]\"", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set.\n"
              "Get a key at console.anthropic.com (a claude.ai Pro "
              "subscription does not include API access -- API billing is "
              "separate), then:\n"
              "  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        sys.exit(1)

    sdk_client = anthropic.Anthropic(api_key=api_key)

    def call(system, user):
        msg = sdk_client.messages.create(
            model=model, max_tokens=64, system=system,
            messages=[{"role": "user", "content": user}])
        return msg.content[0].text

    return CallableLLMClient(call)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Anthropic model id (default: Haiku 4.5, cheapest)")
    parser.add_argument("--agents", type=int, default=2, choices=range(1, 5),
                        help="number of prescriber personas, 1-4 (default: 2)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="shared pull budget per seed (default: 5)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="seeds for the comparison (default: 3)")
    parser.add_argument("--yes", action="store_true",
                        help="skip the cost confirmation prompt")
    args = parser.parse_args()

    persona_names = ["dr-conservative", "dr-balanced", "dr-aggressive",
                     "pharmacist-formulary"][:args.agents]
    personas = [PHARMA_PERSONAS[n] for n in persona_names]
    v, p, e, _ = derive_antibiotic_parameters()

    # Turn-taking shares one pull budget (`rounds`) across all agents, so the
    # call count is bounded by rounds*seeds, not agents*rounds*seeds.
    n_calls = args.rounds * args.seeds
    print(f"This will make up to {n_calls} real API calls to {args.model} "
          f"({args.rounds} shared pulls x {args.seeds} seeds). The Greedy/ECI "
          f"baselines below make zero calls -- they're classical policies, "
          f"not LLM-driven.")
    if not args.yes:
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            print("Aborted -- no API calls made.")
            return

    real_client = _build_anthropic_client(args.model)
    real_population = Population.from_personas(personas, client=real_client)
    mock_population = Population.from_personas(personas, client=MockLLMClient(seed=0))
    env = lambda seed: ConsumableBandit(v, p, e, delta=0.35, horizon=args.rounds,
                                        seed=seed)

    print(f"\n{args.model} (real):")
    real_result = compare_population_to_baselines(env, real_population,
                                                   [Greedy(), ECI()], seeds=args.seeds)
    for name, stats in real_result.items():
        print(f"  {name:>12}  value={stats['value']:7.3f}  "
              f"regret={stats['regret']:7.3f}")

    print("\nMockLLMClient (offline heuristic, for comparison):")
    mock_result = compare_population_to_baselines(env, mock_population,
                                                   [Greedy(), ECI()], seeds=args.seeds)
    for name, stats in mock_result.items():
        print(f"  {name:>12}  value={stats['value']:7.3f}  "
              f"regret={stats['regret']:7.3f}")

    print("\nThis is a first, small-scale look -- report exactly what happened "
          "here, not what the mock predicted. Re-run with more --seeds before "
          "drawing any real conclusion; 3 seeds is not enough to trust a "
          "direction, only enough to confirm the wiring works.")


if __name__ == "__main__":
    main()
