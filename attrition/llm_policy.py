"""LLM-driven policy: an agent that chooses arms by prompting a language model.

Implements the same `Policy.select(state) -> arm_index` contract as every
other policy in `policies.py`, so it drops into `run()`, `compare()`,
`MultiAgentConsumableEnv`, and `SimultaneousPool` unchanged. The prompt states
each available arm's estimated value and its externality charge
(`delta * p * e * remaining`, the quantity ECI subtracts) and the persona
under which the agent is acting; the model's job is to name an arm.

A malformed or out-of-range response never crashes a run: it falls back to
`Greedy`, so a population of LLM agents behaves like a population of Greedy
agents in the worst case, which is itself a meaningful (and cheap-to-produce)
baseline to compare against.
"""

import re

from .llm import MockLLMClient
from .policies import Policy, Greedy

__all__ = ["LLMPolicy"]

_CHOICE = re.compile(r"CHOICE:\s*(\d+)")


class LLMPolicy(Policy):
    """Prompts an LLM client for an arm choice under a given persona.

    Parameters
    ----------
    persona : Persona
    client : LLMClient, optional      defaults to `MockLLMClient()`
    history_window : int              number of past decisions shown in the
                                       prompt, for continuity across rounds
    """

    def __init__(self, persona, client=None, history_window=5):
        self.persona = persona
        self.client = client if client is not None else MockLLMClient()
        self.history_window = int(history_window)
        self.name = f"llm:{persona.name}"
        self.log = []
        self._fallback = Greedy()

    def _render_prompt(self, state):
        system = (
            f"You are acting as: {self.persona.role} ({self.persona.name}).\n"
            f"{self.persona.description}\n"
            f"risk_tolerance: {self.persona.risk_tolerance}\n"
            "Respond with exactly one line: CHOICE: <arm index>"
        )
        lines = [f"Round {state.t}/{state.horizon}. Available arms:"]
        for a in state.available:
            charge = state.delta * state.p[a] * state.e[a] * state.remaining
            lines.append(f"  arm {int(a)}: value~{state.v_hat[a]:.3f} "
                         f"charge~{charge:.3f}")
        if self.log:
            recent = self.log[-self.history_window:]
            lines.append("Your recent choices:")
            for row in recent:
                lines.append(f"  t={row['t']} arm={row['arm']}")
        lines.append("Choose the arm index to pull.")
        return system, "\n".join(lines)

    def select(self, state):
        system, user = self._render_prompt(state)
        try:
            response = self.client.complete(system, user)
            m = _CHOICE.search(response)
            arm = int(m.group(1)) if m else None
        except Exception:
            arm, response = None, None
        if arm is None or arm not in set(int(a) for a in state.available):
            arm = self._fallback.select(state)
        self.log.append({"t": state.t, "arm": arm, "response": response})
        return arm

    def scores(self, state):
        raise NotImplementedError("LLMPolicy has no closed-form score; use select()")
