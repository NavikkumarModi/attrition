"""Pluggable LLM client interface for LLM-driven policies.

`LLMPolicy` (see `llm_policy.py`) only needs an object with a
`.complete(system, user) -> str` method. Two implementations are provided:

    MockLLMClient       deterministic, offline, no network -- the default
                         everywhere, so examples and tests need no API key.
    CallableLLMClient    adapts any `fn(system, user) -> str`, e.g. a closure
                         wrapping the Anthropic or OpenAI SDK. This module
                         never imports those SDKs itself, so the package keeps
                         its zero-hard-dependency property.

Wiring a real backend:

    from attrition.llm import CallableLLMClient
    import anthropic
    _client = anthropic.Anthropic()

    def call(system, user):
        msg = _client.messages.create(
            model="claude-sonnet-5", max_tokens=32, system=system,
            messages=[{"role": "user", "content": user}])
        return msg.content[0].text

    llm = CallableLLMClient(call)
"""

import hashlib
import re

import numpy as np

__all__ = ["LLMClient", "MockLLMClient", "CallableLLMClient"]


class LLMClient:
    """Interface: anything with this method works as an LLM backend."""

    def complete(self, system, user):
        raise NotImplementedError


class CallableLLMClient(LLMClient):
    """Adapts a plain ``fn(system, user) -> str`` callable to `LLMClient`."""

    def __init__(self, fn):
        self.fn = fn

    def complete(self, system, user):
        return self.fn(system, user)


_ARM_LINE = re.compile(r"arm (\d+): value~([\-0-9.]+) charge~([\-0-9.]+)")
_RISK = re.compile(r"risk_tolerance: ([0-9.]+)")
_PEER_ARM = re.compile(r"peer_majority_arm: (\d+)")
_PEER_SHARE = re.compile(r"peer_majority_share: ([0-9.]+)")


class MockLLMClient(LLMClient):
    """Deterministic offline stand-in for a real chat model.

    Reads the same structured prompt `LLMPolicy` sends a real LLM (see
    `llm_policy.py`'s `_render_prompt`) and picks an arm with a persona-
    weighted heuristic: the persona's `risk_tolerance` (0 = fully
    externality-averse, 1 = fully myopic) linearly blends a value-only score
    with the arm's externality charge (`delta * p * e * remaining`, the
    quantity ECI subtracts), plus small seeded noise so two agents that share
    a persona still diverge slightly.

    This stands in for behaviour, not for language -- it exists so the rest
    of the stack (prompting, parsing, fallback, population loop) is exercised
    end to end with no network access and no API key.

    The response is a pure function of `(seed, system, user)`, hashed to seed
    the noise term: no mutable call-order state, so the same inputs always
    give the same output regardless of how many other calls happened first
    or whether calls run concurrently. This matters once `population.py`
    starts dispatching agent decisions through a thread pool (see
    `max_workers` on `simulate_population_simultaneous`) -- a call-counter
    based seed would both race under concurrency and make results depend on
    scheduling order.

    `conformity` (default 0.0, i.e. off) adds `conformity * peer_majority_share`
    to whichever arm most of the agent's visible peers chose last round, when
    the prompt carries that information (see `llm_policy.py`). This is a
    deliberate, documented modeling choice for this offline stand-in -- a
    simple herding rule to exercise the peer-visibility plumbing end to end
    -- not a claim about how a real language model would respond to seeing
    its peers' choices.
    """

    def __init__(self, seed=0, conformity=0.0):
        self._seed = int(seed)
        self.conformity = float(conformity)

    def complete(self, system, user):
        digest = hashlib.sha256(f"{self._seed}\n{system}\n{user}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        rows = _ARM_LINE.findall(user)
        if not rows:
            return "CHOICE: 0"
        m = _RISK.search(system)
        risk = float(m.group(1)) if m else 0.5
        peer_arm_m, peer_share_m = _PEER_ARM.search(user), _PEER_SHARE.search(user)
        peer_arm = int(peer_arm_m.group(1)) if peer_arm_m else None
        peer_share = float(peer_share_m.group(1)) if peer_share_m else 0.0
        best_arm, best_score = None, -np.inf
        for arm_s, value_s, charge_s in rows:
            arm, value, charge = int(arm_s), float(value_s), float(charge_s)
            score = value - (1.0 - risk) * charge + rng.normal(0, 0.01)
            if self.conformity and arm == peer_arm:
                score += self.conformity * peer_share
            if score > best_score:
                best_arm, best_score = arm, score
        return f"CHOICE: {best_arm}"
