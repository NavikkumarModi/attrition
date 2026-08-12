"""Gym- and PettingZoo-compatible environment interfaces.

The interfaces are implemented natively rather than by importing gymnasium or
pettingzoo, so the package has no RL-framework dependency and works standalone.
The method signatures match, so these drop into either ecosystem:

    Gym:        reset() -> (obs, info);  step(a) -> (obs, reward, terminated,
                truncated, info)
    PettingZoo: reset() -> (obs_dict, info_dict);  step(action_dict) ->
                (obs, rewards, terminations, truncations, infos)

Observations expose what a real deployment could see: which arms are available,
their features, accumulated burden, and elapsed time. They do NOT expose the true
externality coefficients, since Theorem 3 establishes those cannot be reliably
estimated from experience -- an agent that reads them off the observation would be
solving a different problem from the one the theory describes.

Set `reveal_externality=True` to lift that restriction for oracle experiments.
"""

import numpy as np

__all__ = ["ConsumableBanditEnv", "MultiAgentConsumableEnv"]


class ConsumableBanditEnv:
    """Single-agent Gym-style environment over a consumable action set.

    Parameters
    ----------
    v, p, e : array-like        arm values, destruction probabilities, externalities
    delta : float               externality scale
    horizon : int
    sigma : float               reward noise
    reveal_externality : bool   include true (p, e) in the observation
    seed : int
    """

    metadata = {"name": "consumable-bandit-v0"}

    def __init__(self, v, p, e, delta=0.05, horizon=100, sigma=0.1,
                 reveal_externality=False, seed=0):
        self.v = np.asarray(v, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.e = np.asarray(e, dtype=float)
        self.n = len(self.v)
        self.delta, self.horizon, self.sigma = float(delta), int(horizon), float(sigma)
        self.reveal = bool(reveal_externality)
        self._seed = seed
        self.reset(seed)

    # ---- spaces (described rather than typed, to avoid a gym dependency) ----
    @property
    def action_space(self):
        return {"type": "Discrete", "n": self.n,
                "note": "invalid actions on destroyed arms raise ValueError"}

    @property
    def observation_space(self):
        d = 3 + (2 if self.reveal else 0)
        return {"type": "Box", "shape": (self.n, d),
                "columns": (["available", "value_estimate", "pulls"]
                            + (["p", "e"] if self.reveal else []))}

    # ---- core API ----------------------------------------------------------
    def reset(self, seed=None):
        self.rng = np.random.default_rng(self._seed if seed is None else seed)
        self.alive = np.ones(self.n, dtype=bool)
        self.t = 0
        self.counts = np.zeros(self.n)
        self._sums = np.zeros(self.n)
        self.total_value = 0.0
        self.total_regret = 0.0
        return self._obs(), self._info()

    @property
    def burden(self):
        return self.delta * float(self.e[~self.alive].sum())

    def _obs(self):
        est = np.where(self.counts > 0, self._sums / np.maximum(self.counts, 1),
                       float(self.v.max()))
        cols = [self.alive.astype(float), est, self.counts.astype(float)]
        if self.reveal:
            cols += [self.p, self.e]
        return np.stack(cols, axis=1)

    def _info(self):
        return {"t": self.t, "burden": self.burden,
                "arms_alive": int(self.alive.sum()),
                "total_value": self.total_value,
                "total_regret": self.total_regret}

    def step(self, action):
        a = int(action)
        if not self.alive[a]:
            raise ValueError(f"arm {a} has been destroyed")
        b = self.burden
        realised = float(self.v[a] - b)
        obs_reward = realised + self.rng.normal(0, self.sigma)

        idx = np.flatnonzero(self.alive)
        best = float(np.max(self.v[idx] - b))
        self.total_regret += best - realised
        self.total_value += realised

        if self.rng.random() < self.p[a]:
            self.alive[a] = False
        self.counts[a] += 1
        self._sums[a] += obs_reward
        self.t += 1

        terminated = not self.alive.any()
        truncated = self.t >= self.horizon
        info = self._info()
        info["realised_value"] = realised
        return self._obs(), obs_reward, terminated, truncated, info

    def valid_actions(self):
        return np.flatnonzero(self.alive)


class MultiAgentConsumableEnv:
    """PettingZoo-style parallel environment: several agents, one shared pool.

    Every agent acts each step. Destruction by any agent raises the burden for all,
    which makes restraint a public good and produces the price of anarchy the
    single-agent setting cannot express.

    Agents act in a fixed order within a step, so an agent later in the order sees
    the pool as it stands after earlier agents have acted.
    """

    metadata = {"name": "consumable-bandit-multi-v0"}

    def __init__(self, v, p, e, n_agents=2, delta=0.05, horizon=100, sigma=0.1,
                 reveal_externality=False, seed=0):
        self.base = ConsumableBanditEnv(v, p, e, delta=delta, horizon=horizon,
                                        sigma=sigma,
                                        reveal_externality=reveal_externality,
                                        seed=seed)
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.reset(seed)

    @property
    def agents(self):
        return list(self.possible_agents)

    def action_space(self, agent):
        return self.base.action_space

    def observation_space(self, agent):
        return self.base.observation_space

    def reset(self, seed=None):
        obs, info = self.base.reset(seed)
        o = {a: obs for a in self.possible_agents}
        return o, {a: info for a in self.possible_agents}

    def valid_actions(self):
        return self.base.valid_actions()

    def step(self, actions):
        """actions: dict agent -> arm index. Returns PettingZoo-style tuples."""
        rewards, infos = {}, {}
        terminated = truncated = False
        obs = None
        for agent in self.possible_agents:
            if not self.base.alive.any():
                rewards[agent] = 0.0
                infos[agent] = self.base._info()
                continue
            a = int(actions[agent])
            if not self.base.alive[a]:               # re-target if already gone
                a = int(self.base.valid_actions()[0])
            obs, r, term, trunc, info = self.base.step(a)
            rewards[agent] = r
            infos[agent] = info
            terminated |= term
            truncated |= trunc
        if obs is None:
            obs = self.base._obs()
        o = {a: obs for a in self.possible_agents}
        return (o, rewards,
                {a: terminated for a in self.possible_agents},
                {a: truncated for a in self.possible_agents},
                infos)

    @property
    def system_value(self):
        """Total value collected by all agents -- the quantity that matters."""
        return self.base.total_value
