"""A genuinely trained -- not hand-derived -- campaign-contact policy.

FINDINGS.md#31 records this as the one item on the election track with no
independently-trained external checkpoint to transplant, unlike AMR
(FINDINGS.md#24.4, which decentralises a real PPO checkpoint trained by
`abx_amr_simulator`, an external project). No equivalent external campaign
policy exists, so the only honest way to close this item is to train one
from scratch, in this repo's own engine.

`train_gradient_bandit` does that with the gradient-bandit method (Sutton &
Barto, *Reinforcement Learning: An Introduction*, 2nd ed., section 2.8): a
softmax preference vector over arms, updated by a REINFORCE-style step after
every pull, using only the realised per-pull reward the environment
returns. No access to `(p, e)` -- the same information Greedy and Thompson
Sampling see, less than ECI or Conservative see. No new dependency: this is
plain numpy, not a call out to an RL framework, matching `envs.py`'s stated
design (native Gym/PettingZoo-shaped interfaces, no RL-framework
dependency in this package).

What this predicts, and why it is not a surprise result: the per-pull
reward `ConsumableBandit.step` returns already has the externality baked in
as a common additive burden term, identical across every available arm in a
given round (`realised = v[arm] - burden`). A learner maximising that
signal therefore has no gradient information distinguishing "this arm is
cheap to the pool" from "this arm is expensive to the pool" -- only
`v[arm]` varies within a round. So a policy trained to maximise its own
realised reward is expected to converge toward Greedy, not because it was
built to imitate Greedy, but because Greedy is what naive reward
maximisation on this environment's own signal actually is. That is the
same point Theorem 4 makes about hand-derived policies, now shown to
emerge from a learned one too -- see `train_gradient_bandit`'s docstring
for the checked prediction and `examples/20_campaign_policy_checkpoint.py`
for the full training-and-evaluation run.
"""

import numpy as np

from ..policies import Policy

__all__ = ["GradientBanditPolicy", "train_gradient_bandit"]


class GradientBanditPolicy(Policy):
    """A policy whose per-arm preferences were learned by
    `train_gradient_bandit`, not hand-derived. Same `.select(state)`
    interface as every policy in `attrition.policies` -- drops into `run`,
    `compare`, `Population`, and `SimultaneousPool` unchanged.
    """
    name = "trained-campaign-policy"

    def __init__(self, preferences):
        self.h = np.asarray(preferences, dtype=float)

    def probs(self, available):
        h = self.h[available]
        h = h - h.max()
        p = np.exp(h)
        return p / p.sum()

    def select(self, state):
        available = state.available
        p = self.probs(available)
        return int(state.rng.choice(available, p=p))


def train_gradient_bandit(env_factory, episodes=4000, alpha=0.1, seed=0):
    """Train a `GradientBanditPolicy` via the gradient-bandit update, one
    step per pull, seeing only the realised reward the environment returns
    each pull -- no `(p, e)`, no simulator rollouts, no closed-form index.

    `env_factory(seed)` must return a fresh env exposing the
    `ConsumableBandit` contract (`.n`, `.reset(seed)`, `.state()`,
    `.step(arm) -> (obs, realised, regret, destroyed)`, `.done()`) --
    the same contract `run`/`compare` train and evaluate every policy
    against, so a trained policy is directly comparable to Greedy/ECI/
    Conservative on identical seeded episodes.

    Preferences persist across episodes (this call trains one policy over
    `episodes` full horizons of the *same* underlying `(v, p, e)`
    parameterisation, re-randomised only in the noise/destruction draws --
    a fixed-scenario policy, the same setting `ECI`/`Conservative` are
    analysed in, not a policy meant to generalise across scenarios).
    """
    rng = np.random.default_rng(seed)
    n = env_factory(seed).n
    h = np.zeros(n)
    baseline, baseline_n = 0.0, 0
    episode_returns = []

    for ep in range(episodes):
        env = env_factory(seed + ep)
        env.reset(seed + ep)
        ep_return = 0.0
        while not env.done():
            available = env.state().available
            hh = h[available] - h[available].max()
            p = np.exp(hh)
            p /= p.sum()
            choice_idx = rng.choice(len(available), p=p)
            arm = int(available[choice_idx])

            _, realised, _regret, _destroyed = env.step(arm)
            ep_return += realised

            baseline_n += 1
            baseline += (realised - baseline) / baseline_n
            advantage = realised - baseline

            grad = -p
            grad[choice_idx] += 1.0
            h[available] += alpha * advantage * grad

        episode_returns.append(ep_return)

    policy = GradientBanditPolicy(h)
    policy.training_curve = np.asarray(episode_returns)
    return policy
