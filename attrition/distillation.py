"""Distillation: approximate an expensive policy's decisions with a fast,
learned one -- so a policy that can only be run a small number of times
(most concretely: a real-LLM-backed `LLMPolicy`, which costs a real,
billable API call per decision) can still inform a population-scale run,
at the one-time cost of a fixed number of calls against it, not one call
per agent per round.

This closes a real, previously-open gap, not a hypothetical one: every
"million agent" scale claim this repo has made (`FINDINGS.md` #52, #53)
is Level 0/1 -- statistical or behavioural voter models, never LLM-driven.
Level 2 (`attrition.election.agents.simulate_level2_contact`,
`attrition.llm_policy.LLMPolicy`) has never been run past a handful of
personas, because a real LLM call per agent per round does not scale, and
was never claimed to. This module does not remove that constraint -- it
lets a small number of real calls train a policy that scales past it,
which is a different, checkable claim.

`collect_decision_traces` runs a source policy against a list of states
once each, recording only what it actually chose -- no proxy for *why*.
`train_distilled_policy` fits a `DistilledPolicy` -- a linear per-arm
scoring function, `w . features(state, arm)`, chosen to agree with the
traces via multinomial-logistic (softmax) regression trained by plain
gradient descent -- the same no-RL-framework, numpy-only design as
`attrition.election.policy_training.train_gradient_bandit`, supervised on
another policy's choices instead of on environment reward. Not a neural
net, not a black box: the learned `weights` are one number per real,
already-observable feature (`v_hat`, `p`, `e`, `delta`, round progress),
inspectable directly.

Faithfulness, not correctness, is what is being learned here. A
`DistilledPolicy` can only ever be as good, or as biased, as the source
policy's own decisions on the states it was shown -- it is not claimed to
BE the source policy, only fit to approximate it. `evaluate_agreement`
reports how well, and should always be checked against held-out states
(never the states used for training) before trusting a `DistilledPolicy`
at scale.
"""

import numpy as np

from .policies import Policy

__all__ = ["DecisionTrace", "collect_decision_traces", "DistilledPolicy",
           "train_distilled_policy", "evaluate_agreement"]


def _arm_features(state, arm):
    """Real, already-observable per-arm signal any policy in this repo
    could see -- v_hat, p, e, delta, and round progress -- plus a bias
    term. No invented feature, nothing beyond what `State` already
    carries."""
    return np.array([
        state.v_hat[arm],
        state.p[arm],
        state.e[arm],
        state.delta,
        state.t / state.horizon if state.horizon else 0.0,
        1.0,
    ])


class DecisionTrace:
    """One recorded decision: the state a policy was shown, and the arm
    it actually chose. Nothing about *why* -- distillation only ever
    supervises on the observable choice, never on an LLM's stated
    reasoning."""

    def __init__(self, state, chosen_arm):
        self.state = state
        self.chosen_arm = int(chosen_arm)


def collect_decision_traces(policy, states):
    """Run `policy.select(state)` once per state in `states` (any policy
    satisfying the standard `Policy.select(state) -> arm_index` contract
    -- most usefully an expensive one, e.g. an `LLMPolicy` backed by a
    real LLM client) and record each decision as a `DecisionTrace`.

    Exactly one call to `policy.select` per state -- no retries, no
    resampling. The cost of collection is exactly `len(states)` policy
    calls, stated up front since that is the real, billable cost when
    `policy` is LLM-backed.
    """
    return [DecisionTrace(state, policy.select(state)) for state in states]


class DistilledPolicy(Policy):
    """A fast, numpy-only linear scoring policy fit by
    `train_distilled_policy` to approximate another, typically far more
    expensive, policy's decisions. Same `.select(state) -> arm_index`
    interface as every policy in `attrition.policies` -- drops into
    `run`, `compare`, `Population`, and `SimultaneousPool` unchanged, at
    the cost of one dot product per available arm instead of, say, one
    real LLM call.
    """
    name = "distilled-policy"

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def scores(self, state):
        return np.array([_arm_features(state, a) @ self.weights
                         for a in state.available])

    def select(self, state):
        return self._argmax(state, self.scores(state))


def train_distilled_policy(traces, epochs=200, lr=0.5, seed=0):
    """Fit a `DistilledPolicy` to agree with `traces` (from
    `collect_decision_traces`) via multinomial-logistic (softmax)
    regression, trained by plain gradient descent -- same no-RL-
    framework, numpy-only design as
    `attrition.election.policy_training.train_gradient_bandit`,
    supervised on another policy's choices instead of on environment
    reward.

    Requires every trace's `state` to expose the standard feature
    contract (`v_hat`, `p`, `e`, `delta`, `t`, `horizon`, `available`)
    -- true of every `State` in `attrition.policies`, real or synthetic,
    so training states can be drawn from any world (election, AMR,
    agent-ecosystem, ...) without special-casing.

    Returns a `DistilledPolicy` with a `.training_curve` attribute: the
    per-epoch fraction of `traces` the fitted policy would itself choose
    (training-set agreement, not held-out) -- report this, and a
    held-out figure from `evaluate_agreement`, before trusting a
    `DistilledPolicy` at scale. This function does not silently claim
    convergence.
    """
    if not traces:
        raise ValueError("train_distilled_policy requires at least one trace")

    rng = np.random.default_rng(seed)
    n_features = _arm_features(traces[0].state,
                               int(traces[0].state.available[0])).shape[0]
    w = rng.normal(scale=0.01, size=n_features)
    training_curve = []

    for _epoch in range(epochs):
        grad = np.zeros_like(w)
        correct = 0
        for tr in traces:
            state = tr.state
            available = state.available
            feats = np.array([_arm_features(state, a) for a in available])
            logits = feats @ w
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            target_idx = int(np.where(available == tr.chosen_arm)[0][0])
            onehot = np.zeros_like(probs)
            onehot[target_idx] = 1.0
            grad += feats.T @ (probs - onehot)
            if int(np.argmax(probs)) == target_idx:
                correct += 1
        grad /= len(traces)
        w -= lr * grad
        training_curve.append(correct / len(traces))

    policy = DistilledPolicy(w)
    policy.training_curve = np.asarray(training_curve)
    return policy


def evaluate_agreement(policy, traces):
    """Fraction of `traces` on which `policy.select(trace.state) ==
    trace.chosen_arm` -- the real, held-out check `train_distilled_policy`'s
    docstring calls for. Pass traces NOT used for training to get a
    genuine generalisation figure, not a restatement of training
    accuracy."""
    if not traces:
        return 0.0
    agree = sum(int(policy.select(tr.state) == tr.chosen_arm) for tr in traces)
    return agree / len(traces)
