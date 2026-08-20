"""Run the pharma population against a REAL model via Groq's free API tier
-- the one validation this repo's LLM-population layer has never had.
Every result in the README, the dashboard, and examples/04-07 comes from a
hand-written offline heuristic; this script (like examples/08's Anthropic
version) is one that asks an actual model what it would do.

Groq's free tier needs no credit card: get a key at console.groq.com, then
set it as an environment variable (never hardcode it, never paste it into a
chat or commit it):

    export GROQ_API_KEY=gsk_...

Free-tier rate limits are real but generous for a small run like this one's
defaults (5 shared pulls x 3 seeds = 15 calls). If a call gets rate-limited,
LLMPolicy's built-in fallback to Greedy means the run still completes --
you'll just see fewer real model decisions than requested, which the script
reports honestly rather than silently.

Install:  pip install "attrition[groq]"
Run:      python examples/09_real_llm_run_groq.py
"""

import argparse
import os
import sys

from attrition import (CallableLLMClient, ConsumableBandit, ECI, Greedy,
                       MockLLMClient, PHARMA_PERSONAS, Population,
                       compare_population_to_baselines,
                       derive_antibiotic_parameters)
from attrition.llm_policy import _CHOICE


def _build_groq_client(model):
    try:
        import groq
    except ImportError:
        print("The 'groq' package isn't installed.\n"
              "  pip install \"attrition[groq]\"", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is not set.\n"
              "Get a free key at console.groq.com (no credit card needed), "
              "then:\n"
              "  export GROQ_API_KEY=gsk_...", file=sys.stderr)
        sys.exit(1)

    sdk_client = groq.Groq(api_key=api_key)

    def call(system, user):
        kwargs = {}
        if "gpt-oss" in model:
            # gpt-oss models spend an uncapped number of hidden reasoning
            # tokens before the visible answer -- at the default effort,
            # they silently truncated to an EMPTY response (no exception,
            # so LLMPolicy's fallback swallowed it) on most calls at this
            # prompt's length. reasoning_effort='low' keeps the visible
            # answer inside max_tokens reliably; confirmed empirically
            # (0/80 empty responses with this, vs 47/80 without).
            kwargs["reasoning_effort"] = "low"
        completion = sdk_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                     {"role": "user", "content": user}],
            max_tokens=300,
            **kwargs,
        )
        return completion.choices[0].message.content

    return CallableLLMClient(call)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="openai/gpt-oss-20b",
                        help="Groq model id (default: openai/gpt-oss-20b). "
                             "Groq's free-tier catalog changes over time -- "
                             "list current ids with groq.Groq().models.list()")
    parser.add_argument("--agents", type=int, default=2, choices=range(1, 5),
                        help="number of prescriber personas, 1-4 (default: 2)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="shared pull budget per seed (default: 5)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="seeds for the comparison (default: 3)")
    args = parser.parse_args()

    persona_names = ["dr-conservative", "dr-balanced", "dr-aggressive",
                     "pharmacist-formulary"][:args.agents]
    personas = [PHARMA_PERSONAS[n] for n in persona_names]
    v, p, e, _ = derive_antibiotic_parameters()

    n_calls = args.rounds * args.seeds
    print(f"Model: {args.model} (Groq free tier). Up to {n_calls} real calls "
          f"({args.rounds} shared pulls x {args.seeds} seeds). Greedy/ECI "
          f"below make zero calls -- they're classical policies, not "
          f"LLM-driven. Any rate-limited call falls back to Greedy "
          f"automatically (see LLMPolicy).\n")

    real_client = _build_groq_client(args.model)
    real_population = Population.from_personas(personas, client=real_client)
    mock_population = Population.from_personas(personas, client=MockLLMClient(seed=0))
    env = lambda seed: ConsumableBandit(v, p, e, delta=0.35, horizon=args.rounds,
                                        seed=seed)

    print(f"{args.model} (real):")
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

    # .log accumulates across the whole seed sweep for one representative
    # agent (compare_population_to_baselines reuses policy instances across
    # seeds). A row only counts as a genuine decision if the response both
    # exists (no exception) AND actually contained a parseable "CHOICE: N"
    # -- an empty or off-format response is a silent Greedy fallback too,
    # and doesn't raise, so checking `response is not None` alone previously
    # missed this: 47/80 calls on this same model were empty strings that
    # counted as "responded" but were fallbacks in disguise.
    log = real_population.members[persona_names[0]].log
    genuine = sum(1 for row in log
                 if row["response"] and _CHOICE.search(row["response"]))
    exceptions = sum(1 for row in log if row["response"] is None)
    unparseable = len(log) - genuine - exceptions
    print(f"\n{persona_names[0]}'s log (across all {args.seeds} seeds): "
          f"{genuine}/{len(log)} calls produced a genuine parseable decision "
          f"-- report this number, not just the aggregate value/regret "
          f"above, since a low ratio means most of this run was Greedy "
          f"fallback in disguise, not the real model.")
    if exceptions:
        errors = {}
        for row in log:
            if row.get("error"):
                errors[row["error"]] = errors.get(row["error"], 0) + 1
        print(f"{exceptions} raised an exception:")
        for msg, count in errors.items():
            print(f"  x{count}  {msg}")
    if unparseable:
        print(f"{unparseable} returned a response that didn't parse (e.g. "
              f"empty -- often hidden reasoning tokens exhausting max_tokens "
              f"before the visible answer).")
    print("This is a first, small-scale look -- report exactly what "
          "happened here, not what the mock predicted. Re-run with more "
          "--seeds before drawing any real conclusion.")


if __name__ == "__main__":
    main()
