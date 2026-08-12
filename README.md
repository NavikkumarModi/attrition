# Bandits with Consumable Action Sets

**No-regret is not no-harm.**

A policy can achieve *exactly zero regret* while destroying unbounded value — and
its own metric will report flawless performance the entire time.

This repository contains the theory, the proofs, the falsifications, and a small
library for studying bandit problems where **taking an action destroys it**, and
each destroyed action imposes a permanent cost on everything that remains.

---

## The one-table version

Five policies, same environments, 25 seeds:

| policy | value | regret | verdict |
|---|---|---|---|
| greedy | 14.449 | **0.000** | perfect metric, worst outcome |
| thompson | 15.171 | 1.341 | |
| ucb | 13.257 | 6.136 | |
| conservative | 17.089 | 8.160 | |
| **eci** | **21.214** | **10.069** | worst metric, best outcome |

The regret column and the value column are ordered **backwards** relative to each
other. Greedy never once fails to pull the best available arm — and ends 47%
below the policy that "looks" worst.

---

## Why this happens

Regret is measured against the best arm *currently available*. When your own
actions remove arms, the benchmark degrades in lockstep with the damage you
cause. Every choice greedy makes is optimal **given the environment it has
already ruined**.

```
regret(greedy) = 0            always, by construction
V_opt - V_greedy = m · δE     unbounded in pool size m and externality δE
```

---

## Install

```bash
git clone https://github.com/<user>/consumable-bandits
cd consumable-bandits
pip install -e .
```

Requires Python 3.9+ and NumPy.

## Quick start

```python
from evolving_bandits import ConsumableBandit, Greedy, ECI, Conservative, compare

env = lambda seed: ConsumableBandit.random(
    n=20, k_spread=1.5, delta=0.03, horizon=60, seed=seed)

results = compare(env, [Greedy(), Conservative(e_bound=2.0), ECI()], seeds=25)

for name, stats in results.items():
    print(f"{name:>14}  value {stats['value']:7.3f}  regret {stats['regret']:7.3f}")
```

Inspect a single session step by step:

```python
from evolving_bandits import run
result = run(env(0), Greedy(), log=True)
for row in result["log"][:10]:
    print(row)          # {'t':.., 'arm':.., 'value':.., 'regret':.., 'alive':..}
```

---

## The model

Arm `a` has value `v_a`, destruction probability `p_a`, and externality
coefficient `e_a`. Pulling `a`:

- yields `v_a − B(S)`, where `B(S) = δ · Σ_{dead i} e_i` is accumulated burden;
- destroys `a` with probability `p_a`, permanently adding `δ e_a` to `B`.

The governing quantity is the **expected marginal externality**

```
κ_a = p_a · e_a
```

## Results

| | statement | status |
|---|---|---|
| **T1** | greedy is optimal **iff** `κ_a` is constant across arms | proven |
| **T2** | ECI index `v_a − δ κ_a (T−t)` captures 52–99% of the gap | empirical |
| **T2b** | pure sequencing: optimal order is **ascending in `e`**; values are irrelevant | proven |
| **T3** | `SE(δê) ≥ 2σ/√min(T, Σ 1/p_j)` — **independent of horizon** | proven |
| **T4** | zero-regret policy loses `m·δE`, unbounded | proven |

### T1 — what actually matters is the product

Neither `p` nor `e` alone. Holding `p·e` constant while varying both factors
widely leaves greedy **exactly** optimal (0/10 instances broken); varying the
product breaks it in 10/10.

### T3 — the externality cannot be learned

Estimating `e_i` requires comparing rewards before and after arm `i` dies.
Destruction is irreversible, so **each arm supplies exactly one such
transition**. The sample budget is capped by the pool, not the clock:

| `p` range | rounds of data | corr(ê, e) |
|---|---|---|
| [0.90, 1.00] | 8.5 | **+0.118** |
| [0.25, 0.50] | 23.0 | +0.261 |
| [0.04, 0.10] | 60.0 | **+0.828** |

A 16× increase in horizon changes RMSE by *zero* at `p = 0.9`. The data-generating
process is the consumption itself.

> **Specify, don't estimate.** `Conservative`, which assumes a uniform upper bound
> and learns nothing, beats every learned estimator tested.

### T4 — the proof

`m+1` arms: one "hot" (value 1, externality `E`) and `m` "safe" (value `1−ε`,
externality 0), with `p = 1`.

Greedy takes hot first and pays `δE` on all `m` remaining pulls. The optimum
defers hot to last and pays it zero times.

```
V_opt − V_greedy = m · δE          exact, verified 15/15 against DP
```

At `m = 12`, `δE = 2`: optimum `+12.88`, greedy `−11.12`. Regret: `0.0000`.

---

## Policies

| class | knowledge required | notes |
|---|---|---|
| `Greedy` | `v` | zero regret, unbounded loss |
| `ThompsonSampling`, `UCB` | — | baselines; both consumption-blind |
| `ECI` | `v`, `p`, `e` | externality-corrected index |
| `Conservative` | `v`, `p`, one bound | **recommended default** — no per-arm knowledge |
| `SortByE` | `e` | exactly optimal in pure sequencing |
| `Rollout` | base policy + simulator | 99.5–99.9% of optimum |

## Agent tool ecosystems

An orchestrator routing across API-backed tools is an instance: heavy use trips a
rate limit or revokes a key (destruction), and tools sharing upstream
infrastructure degrade together (externality). See
`experiments/exp17_agent_ecosystem.py` for a session trace where the greedy
router's dashboard shows a flawless zero-regret record while it burns the
ecosystem to a worse outcome with fewer tools remaining.

---

## Reproducing

```bash
python -m pytest tests/          # 13 tests, one per theorem
python experiments/exp21_theorem4_construction.py    # T4, exact
python experiments/exp22_theorem3_proof.py           # T3, three parts
python experiments/exp24_remaining_theory.py         # T1 necessity, T2b
```

`experiments/` contains all 24 experiments in order, including the ones that
failed. `FINDINGS.md` is the lab notebook; `THEORY.md` holds statements, proofs,
and proof routes.

## What was falsified

Recorded rather than deleted, because the negative results shaped the theory:

- **IPIC**, a preservation heuristic — lost to plain greedy in every regime; its
  premise (valuable arms are worth preserving) is false without externalities.
- **The endogenous/exogenous unification claim** — rotting bandits already
  handle both (Seznec et al. 2020).
- **The effective-horizon index refinement** — made the index worse.
- **Horizon-scaling of the T4 gap** — the gap is flat in `T`, linear in pool size.
- **ECI exactness in pure sequencing** — brute force found better orderings.
- **Per-arm and feature-based learning of `e`** — the feature route's capture
  *falls* as sharing increases, because low dispersion means nothing to capture.

## Related work

- **Rotting bandits** (Levine et al. 2017; Seznec et al. 2019, 2020) — rewards
  decay with pulls or time. Decay is gradual, not removal, and the regime is known.
- **Mortal bandits** (Chakrabarti et al. 2008) — arms expire, but exogenously.
- **Blocking bandits** — unavailability is temporary.
- **Bandits with knapsacks** (Badanidiyuru et al.) — closest neighbour;
  consumption against a **global exogenous** budget, without cross-arm reward coupling.
- **Positive externalities** (Shah, Johari & Scheinkman, NeurIPS 2018) —
  externalities act on **arrivals** and are positive; they also show UCB incurs
  linear regret, a parallel failure of a standard algorithm under externality
  structure.

## Citation

```bibtex
@misc{consumable-bandits,
  title  = {No-Regret is Not No-Harm: Bandits with Consumable Action Sets},
  author = {Modi, Navikkumar},
  year   = {2026},
  note   = {https://github.com/<user>/consumable-bandits}
}
```

## License

MIT
