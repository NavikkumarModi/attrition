"""Distill a Level 2 (LLM-driven) policy into a fast DistilledPolicy that
scales past the small N every real-LLM run in this repo has been limited
to (see FINDINGS.md #52/#53 for the Level 0/1 "million agent" work this
extends -- Level 2 has never been run past a handful of personas, because
a real LLM call per agent per round does not scale, and was never claimed
to). This does not remove that constraint; it shows a small, fixed number
of calls against an LLM-driven policy can train a policy that runs past
it, at zero further LLM cost.

No API key required: this uses MockLLMClient, the same deterministic
offline stand-in every other example in this repo defaults to, standing
in for a real, expensive LLM-driven `LLMPolicy`. To use a real model,
build a real client the same way examples/08_real_llm_run.py does and
pass it to `LLMPolicy(persona, client=real_client)` in place of the
Mock -- everything downstream (collection, training, evaluation) is
unchanged, since distillation only ever sees `policy.select(state)`, not
which kind of client backs it.

Run:  python examples/21_distill_llm_policy.py
"""

import numpy as np

from attrition import (State, LLMPolicy, MockLLMClient, VOTER_PERSONAS,
                       Greedy, ECI, collect_decision_traces,
                       train_distilled_policy, evaluate_agreement,
                       compare, ConsumableBandit, derive_campaign_parameters)


def _campaign_states(v, p, e, n, horizon=5, seed=0):
    """Draw `n` real campaign-shaped states -- same (v, p, e) arrays every
    campaign scenario in this repo uses, at random rounds within a
    horizon -- the same states an LLM-driven voter policy would actually
    be asked to decide from during a real run."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        t = int(rng.integers(0, horizon))
        states.append(State(available=np.arange(len(v)), t=t, horizon=horizon,
                            v_hat=v, p=p, e=e, delta=0.3, rng=rng))
    return states


def main():
    print(__doc__)
    v, p, e, _ = derive_campaign_parameters()
    persona = VOTER_PERSONAS["voter-swing"]

    # The "expensive" source policy -- MockLLMClient here so this example
    # runs offline and free; a real client costs one real API call per
    # `.select()` call, exactly like every call `collect_decision_traces`
    # below makes.
    source = LLMPolicy(persona, client=MockLLMClient(seed=0))

    n_train, n_test = 150, 60
    print(f"Collecting {n_train} training + {n_test} held-out decision "
          f"traces from the source policy ({n_train + n_test} total calls "
          f"-- the real, billable cost if this were a real LLM).")
    train_states = _campaign_states(v, p, e, n_train, seed=1)
    test_states = _campaign_states(v, p, e, n_test, seed=2)
    train_traces = collect_decision_traces(source, train_states)
    test_traces = collect_decision_traces(source, test_states)

    print("\nTraining a DistilledPolicy (numpy-only softmax regression, "
          "no LLM calls) to approximate the source's decisions...")
    policy = train_distilled_policy(train_traces, epochs=300, lr=0.5, seed=0)
    train_agreement = policy.training_curve[-1]
    held_out_agreement = evaluate_agreement(policy, test_traces)
    print(f"  training-set agreement:  {train_agreement:6.1%}")
    print(f"  held-out agreement:      {held_out_agreement:6.1%}  "
          f"(the real generalisation figure -- report this one, not the "
          f"training-set number)")

    print("\nNow running the DistilledPolicy at a scale no real-LLM run in "
          "this repo has attempted -- 5,000 seeded episodes, zero further "
          "LLM calls -- alongside the classical Greedy/ECI baselines for "
          "context:\n")
    env_factory = lambda s: ConsumableBandit(v, p, e, delta=0.3, horizon=5, seed=s)
    result = compare(env_factory, [Greedy(), ECI(), policy], seeds=5000)
    for name, stats in result.items():
        print(f"  {name:>16}  value={stats['value']:7.3f}  "
              f"regret={stats['regret']:7.3f}")

    print("\nThis is a distillation-quality check, not a claim that the "
          "distilled policy IS the source policy -- held-out agreement is "
          "the number that matters, and it is reported above, not hidden "
          "behind the value/regret table.")


if __name__ == "__main__":
    main()
