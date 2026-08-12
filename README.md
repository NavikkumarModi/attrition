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

## Domains

Four instantiations ship with the library, each a running parameterisation rather
than an illustration. Fit quality is stated honestly.

| domain | arm | destruction (`p`) | externality (`e`) | fit |
|---|---|---|---|---|
| `agent_tools` | API-backed tool | call trips a rate limit or revokes a key | shared upstream infrastructure | strong |
| `adaptive_therapy` | dose level | dose exhausts drug-sensitive cells | released resistant clone degrades all later treatment | strong |
| `platform_trial` | treatment arm | dropped for futility on self-generated evidence | loss of concurrent control | moderate |
| `design_space` | process parameter setting | run consumes material or goes out of spec | narrows the defensible filing envelope | partial |

```python
from evolving_bandits import adaptive_therapy, Greedy, ECI, run, DOMAIN_NOTES

env = adaptive_therapy(seed=0)
print(DOMAIN_NOTES["adaptive_therapy"])
print(run(env, Greedy())["regret"])     # 0.0 -- and it loses by 49%
```

Greedy records **cumulative regret of exactly 0.0000 in all four domains**, and is
beaten in all four — by 30–49% where `κ` dispersion is largest.

### Mechanistic engines

Three domains now derive `(v, p, e)` from dynamics instead of hand-setting them
(`engines.py`): Lotka–Volterra tumour competition, a shared-control platform trial
where dropping an arm fragments the control stream, and a process-development model
where an out-of-spec run truncates the filing envelope.

**A prediction that holds both ways.** Greedy loses 26.8–121.1% (tumour) and
17.3–107% (design space) — but is **exactly optimal** in the platform trial. That
is not a failure; the theory predicts it:

| engine | std(κ) | corr(v, κ) | greedy |
|---|---|---|---|
| tumour dynamics | 0.1705 | **+0.614** | fails |
| platform trial | 0.0371 | **−0.938** | optimal |
| design space | 0.2954 | **+0.789** | fails |

Dispersion of `κ` is necessary but not sufficient — it must *conflict with the value
ordering*. A higher dose controls more tumour **and** exhausts the sensitive
compartment; an aggressive process setting yields more **and** truncates the
envelope. But promising trial arms are precisely those that won't be dropped for
futility, so they never fragment the control — value and safety are aligned, and
greedy is already right.

The failure mode belongs to domains where **ambition and damage travel together**.

**Adaptive therapy is the sharpest case.** Maximum tolerated dose — administer the
highest tolerable dose — *is* the greedy policy. Dose intensity raises immediate
tumour control, the chance of exhausting the drug-sensitive population, and the
damage done when that happens, all together.

This domain is now **mechanistically grounded**: `(v, p, e)` are derived from
Lotka–Volterra competition dynamics rather than set by hand.

```python
from evolving_bandits import derive_arm_parameters
v, p, e, doses = derive_arm_parameters(engine_kwargs={"dt": 5.0}, s0_sd=0.26)
```

On those derived parameters MTD records private regret of **exactly 0.000000** and
loses 5.2% / 17.4% / 36.9% of system value at δ = 0.05 / 0.15 / 0.30.

Running the dynamics with no bandit layer at all, and measuring time to progression
(the clinical endpoint — final burden cannot distinguish protocols, since the
resistant clone always reaches carrying capacity eventually):

| protocol | time to progression | vs MTD |
|---|---|---|
| MTD (always max dose) | 26 | — |
| adaptive, back off at S<0.40 | 38 | +46% |
| adaptive, back off at S<0.50 | **44** | **+69%** |

Competitive release is reproduced, not assumed. Still a two-compartment reduced
model with no spatial structure or pharmacokinetics — no clinical conclusion should
be drawn from it.

See `experiments/exp25_domains.py` for all four, and
`experiments/exp17_agent_ecosystem.py` for a session trace where the greedy
router's dashboard shows a flawless zero-regret record while it burns the
ecosystem to a worse outcome with fewer tools remaining.

---

## Terminal commitment

A distinct problem in the same family: a finite budget of irreversible experiments,
then a **one-shot declaration** that fixes the feasible set for good. Motivated by
CMC design-space filing under ICH Q8 — operating inside the declared envelope needs
no approval, outside it triggers a variation.

```python
from evolving_bandits import (evaluate_commitment_policy, edge_first_policy,
                              optimal_commitment_policy)
evaluate_commitment_policy(optimal_commitment_policy, budget=2)
```

| budget | policy | operating value | envelope width |
|---|---|---|---|
| 2 | greedy (best yield first) | 36.908 | **1.00** |
| 2 | edge-first (chase width) | 36.908 | **1.00** |
| 2 | **expand outward** | **40.664 (+10.2%)** | **2.92** |
| 6 | edge-first | 36.908 (**−18.1%**) | **1.00** |

**Two failure modes, both absent from the sequential setting.**

*Greedy buys yield it cannot claim.* An envelope is an interval containing the
nominal point, so demonstrating a distant high-yield setting is worthless unless
everything between is demonstrated too. At budget 2 greedy ends with width 1.00.

*Ambition narrows the envelope it was trying to widen.* Edge-first targets the
outermost settings to maximise claimable width — and each failure blocks that
setting permanently. Its width is 1.00 at **every** budget, and it gets *worse* as
budget grows (−10.8% → −18.1%). **Spending more to widen the envelope makes it
narrower.**

Note the advantage of the correct policy *shrinks* with budget, the opposite of the
sequential setting. Terminal commitment is about scarcity of evidence at the moment
of an irreversible decision; given enough evidence the problem dissolves. It is
hardest exactly when experiments are expensive.

## Environments and scenarios

Gym- and PettingZoo-compatible interfaces, implemented natively so there is no
RL-framework dependency:

```python
from evolving_bandits import load, describe

describe()                              # list all scenarios and what they show

env = load("shared-quota", seed=0)      # Gym API
obs, info = env.reset(0)
obs, reward, terminated, truncated, info = env.step(action)

menv = load("shared-quota-competing")   # PettingZoo parallel API
obs, rewards, terms, truncs, infos = menv.step({a: 0 for a in menv.agents})
```

**Observations deliberately hide the true externality coefficients.** Theorem 3
says they cannot be reliably estimated from experience, so an agent that read them
off the observation would be solving a different problem. Pass
`reveal_externality=True` for oracle experiments.

### Scenario packs

Each scenario documents the phenomenon it should exhibit, and the test suite
asserts it — so a scenario that drifts is a detectable regression.

| scenario | std(κ) | corr(v,κ) | greedy loss | greedy regret |
|---|---|---|---|---|
| `high-dispersion` | 0.6285 | +1.000 | **176.5%** | 0.00000000 |
| `design-space` | 0.2954 | +0.789 | 99.3% | 0.00000000 |
| `adaptive-therapy` | 0.1705 | +0.614 | 36.9% | 0.00000000 |
| `shared-quota` | 0.9948 | +0.315 | 3.7% | 0.00000000 |
| `platform-trial` | 0.0371 | −0.938 | **0.0%** | 0.00000000 |
| `aligned-control` | 0.6299 | −0.971 | **0.0%** | 0.00000000 |

The last two are negative controls: κ is dispersed but *aligned* with value, so
greedy is already optimal and the theory says so. `shared-quota-competing` adds a
second agent for price-of-anarchy experiments.

## Paper

Two builds from the same body text:

| file | format | use |
|---|---|---|
| `paper/no-regret-is-not-no-harm.pdf` | plain `article` | reading, internal review |
| `paper/arxiv-no-regret-is-not-no-harm.pdf` | `arxiv.sty` | arXiv preprint submission |

```bash
cd paper && make                       # figures + both PDFs
make arxiv-submission.tar.gz           # upload bundle for arXiv
```

arXiv imposes no mandatory template; `arxiv.sty` is the community convention for
preprints and is based on the NeurIPS layout. For a venue submission, swap in that
venue's style file (e.g. `icml2026.sty`) — the body text needs no changes.

## Reproducing

```bash
python -m pytest tests/          # 21 tests, one per theorem
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
