# ATTRITION × antimicrobial stewardship: an external validation

**Status:** empirical extension, kept deliberately separate from the core
paper (`paper/no-regret-is-not-no-harm.pdf`) — see [Scope](#scope) below.
**Code:** `attrition.simultaneous` (`DegradingPool`, `decentralised_value_simultaneous`
with `rule="lambda_weighted"`), `experiments/exp62_amr_lambda_sweep.py`,
`experiments/exp63_partial_recovery.py`, `scripts/amr_decentralize_checkpoint.py`.
**Underlying findings log:** `FINDINGS.md`, sections 24–25 (this document is
the readable summary of that log, not a replacement for it — the log has the
full derivation trail, including a corrected error, kept visible rather than
edited away).

## Summary

ATTRITION's central theorem is that a policy can look zero-regret on its own
metric while causing unbounded harm to a shared, consumable resource.
That result was proved on a synthetic bandit model. This extension asks
whether the same divergence appears in a domain ATTRITION did not build,
using an independently developed, independently motivated simulator as the
external test rather than a domain chosen to flatter the theorem.

`abx_amr_simulator` (Lee & Blumberg, 2026, [arXiv:2603.11369](https://arxiv.org/abs/2603.11369),
[github.com/jl56923/abx_amr_simulator](https://github.com/jl56923/abx_amr_simulator))
frames antibiotic prescribing exactly the way ATTRITION frames a consumable
action: immediate clinical benefit against a resistance externality that
degrades future treatment options. It fixes that externality with a single
dial on a single policy — `R_overall = (1-λ)·individual + λ·community` — and
explicitly lists multi-agent decentralisation and non-permanent resistance
dynamics as unbuilt future work in its own source.

Three results, run in increasing order of how much of their actual system
each one touches:

1. **Transplant** (§1 below) — their reward dial, ported into genuinely
   multi-agent, simultaneous-action play using this repo's own machinery.
2. **Reproduction** (§2) — their own environment, their own training
   pipeline, run end to end, no modifications.
3. **Decentralised checkpoint** (§3) — the literal trained policy from (2),
   run as several non-coordinating copies sharing one resistance pool
   *inside their own environment*, via a seam their own code already
   reserves for this.

A fourth result (§4) relaxes the model's own permanence assumption and asks
whether the finding survives — honestly, including where it changes sign
rather than just shrinking.

## 0. The variable mapping

Stated once, explicitly, rather than left implicit in code comments — this
is what makes §1–§4 an instance of ATTRITION's model, not a loose analogy:

| ATTRITION | AMR |
|---|---|
| `arm` | antibiotic / prescribing option |
| `v_a` | immediate treatment benefit |
| `p_a` | probability / strength of resistance impact |
| `e_a` | resistance externality |
| `κ_a = p_a·e_a` | expected future harm |
| consumption | resistance accumulation / loss of efficacy |
| surviving pool | effective treatment options |
| private reward | immediate patient-level benefit |
| system value | aggregate treatment + future efficacy |

## 1. Transplant: their dial, under real decentralisation

Their model has exactly one decision-maker, so there is nothing for it to
coordinate. `attrition.simultaneous.decentralised_value_simultaneous`'s new
`rule="lambda_weighted"` scores each arm as `v_a - λ·δ·κ̂_a·(T-t)`, an exact
generalisation of the repo's existing `greedy` (`λ=0`) and `eci` (`λ=1`,
noise-free — verified bit-for-bit equal in `tests/test_theorems.py`) rules,
with an `obs_noise` knob mirroring their partial-observability parameter.
Run against `m=3` simultaneously-acting prescribers sharing one antibiotic
pool (`experiments/exp62_amr_lambda_sweep.py`, same arms/budget as the
existing pharma population demo):

| λ | noise | system value | reported value | gap | PoA vs. planner |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0 | 8.00 | 13.72 | 5.71 | 1.61 |
| 0.50 | 0.0 | 11.40 | 13.00 | 1.61 | 1.13 |
| 1.00 | 0.0 | 12.08 | 12.72 | 0.64 | 1.07 |
| 1.00 | 2.0 | 9.46 | 11.67 | 2.20 | 1.36 |

Reward-shaping genuinely narrows the gap between what each prescriber's own
score reports and what actually happens to the pool — by 89% at full
weighting with perfect information. It does not close it: price of anarchy
bottoms out at 1.07, not 1.00, and the floor rises more than 3× under
realistic observation noise. The residual is exactly what this repo's
Theorem 6 predicts — a coordination cost (collision, blindness) that a
single-agent reward dial cannot remove, because their model has no second
agent to collide with.

## 2. Reproduction: does their own dial work, on its own terms?

Cloned their repository at the commit current as of 2026-08-24, installed
under Python 3.11 (their pinned `torch==2.2.2` has no wheel for 3.13),
trained their default PPO configuration (2 antibiotics, 5 patients/step, 500
steps/episode, 25 training episodes — their own tutorial default) at both
dial extremes, no other changes:

| λ | final AMR, antibiotic A | final AMR, antibiotic B | patient clinical benefit |
|---:|---:|---:|---:|
| 0.0 | 99.97% | 99.33% | −1854.7 |
| 1.0 | 74.6% | 25.1% | −1850.1 |

Their claim holds up under real training: the community-weighted dial cuts
antibiotic B's resistance by 74 points at statistically zero clinical cost.
**This is a genuine, independent confirmation of their result, not a
takedown of it** — the divergence in §1 is not a rebuttal, it is the next
question their single-agent environment cannot ask.

Exact reproduction commands are in `FINDINGS.md#24.2`.

## 3. The literal checkpoint, decentralised

`ABXAMREnv.__init__` accepts a pre-built `amr_dynamics_instances` dict —
present in their source, documented in their own code as reserved for a
"future multi-agent refactor," unused by their training or CLI tooling. That
seam is enough to run three independent copies of the trained λ=1.0
checkpoint from §2, each with its own patient generator and reward
calculator, sharing one injected resistance-pool object instead of each
getting its own (`scripts/amr_decentralize_checkpoint.py`). No adapter
between action spaces was needed — each copy sees exactly the observation
shape it was trained on.

| condition | total reward (3 agents) | final AMR, A | final AMR, B |
|---|---:|---:|---:|
| siloed (private pools) | −1109.72 | 74.6% | 25.1% |
| shared (real externality) | −1151.15 | **96.9%** | 19.9% |

The siloed row exactly reproduces §2's single-agent result — a consistency
check that the harness is wired correctly, not a coincidence. Under sharing,
antibiotic A's resistance climbs from 74.6% to 96.9%, nearly back to the
*no-stewardship* level from §2 (99.97%). A policy that was, alone, buying
real stewardship at close to zero clinical cost loses most of that benefit
the moment it is decentralised, because it was never trained to know other
copies of itself exist.

Antibiotic B moves the other way (25.1% → 19.9%) — plausibly reallocation
pressure through the crossresistance matrix between the two antibiotics.
Reported as observed; not dug into further here.

**Caveat, stated rather than smoothed over:** within a round the three
environments are stepped sequentially, so a small amount of within-round
information leaks between them. This is not bit-for-bit `SimultaneousPool`'s
BLINDNESS assumption (burden visible only from the *next* round).
Decentralised and non-coordinating either way, but the distinction is real.

## 4. Partial recovery: the divergence's sign, not just its size

Calling antimicrobial resistance "irreversible arm consumption" overclaims —
real resistance accumulates and decays, it does not binary-flip the way a
destroyed arm does in ATTRITION's base model. `DegradingPool`
(`attrition.simultaneous`) replaces permanent destruction with a continuous,
partially-recoverable state, `s_a(t+1) = ρ·s_a(t) + (1-ρ)·e_a·hit_a(t)`, and
asks directly whether §1's divergence survives when consumption is no longer
permanent.

**It does not survive unchanged — it survives with the sign flipped,
depending on the scoring rule** (`experiments/exp63_partial_recovery.py`,
same setup as §1):

```
λ=0.0 (blind score)              λ=1.0 (eci-style pricing)
 ρ    system  reported  gap       ρ    system  reported   gap
0.00   11.64    15.38  3.74      0.00   13.40    13.11  -0.29
0.50   12.11    15.38  3.28      0.50   13.50    13.11  -0.39
0.99   15.22    15.38  0.17      0.99   13.93    13.11  -0.82
```

At `λ=0` the gap stays positive but shrinks toward 0 as `ρ→1` — at this
horizon, high-persistence harm has not had time to build to its ceiling. At
`λ=1` the gap is **negative at every ρ tested**: ECI-style pricing charges an
arm's full externality the instant it is hit, as if permanent, which
*overcharges* relative to what partial recovery actually costs — a real,
quantifiable cost of applying permanent-loss-calibrated pricing in a domain
where loss is not actually permanent.

Two things checked rather than assumed: `DegradingPool` does **not** reduce
to `SimultaneousPool` as `ρ→1` (a long-horizon run at `ρ=0.999` clears more
than 2× the value of the permanent-destruction model at the same horizon —
they formalise different notions of irreversibility, value-degradation vs.
option-removal, and are not nested). And an earlier informal write-up of
this extension had `ρ`'s persistence direction backwards; corrected in
`DegradingPool`'s docstring rather than silently fixed.

## 5. Is §4's ρ sweep arbitrary, or does it bracket their real system?

§4 sweeps ρ ∈ {0, 0.25, 0.5, 0.75, 0.95, 0.99} without reference to any real
antibiotic. Their `AMR_LeakyBalloon` dynamics give a principled, if partial,
anchor: `leak` is exactly a per-step decay rate on accrued pressure, the
same role `1-ρ` plays in `DegradingPool`. Reading `leak` directly from their
default environment config (`environment/default.yaml`):

| antibiotic | their `leak` | implied ρ = 1 − leak |
|---|---:|---:|
| A | 0.20 | 0.80 |
| B | 0.10 | 0.90 |

Both fall inside the interior of the range already swept in §4 (between the
`ρ=0.75` and `ρ=0.95` rows), not outside it — the sign-flip and floor
findings were already checked at ρ values bracketing their real system's
implied persistence, not only at the extremes. This is a partial robustness
check, not a full one: it validates ρ against their real decay rate, not
`v_a`, `p_a`, or `e_a` against their reward/dose formulas, which is left
open below.

## 6. Sequential vs. simultaneous, for this rule specifically

Not run as a new experiment. `attrition`'s existing Theorem 6 (see
`THEORY.md`, `tests/test_theorems.py`'s
`test_sequential_agents_equal_single_learner`) is a structural equivalence —
under turn-taking with full information, `m` agents over `T` rounds behave
exactly like one learner over `m·T` pulls, independent of which per-round
scoring rule they use. It was proved generally, not re-derived per rule, so
it applies to `lambda_weighted` without needing a dedicated turn-taking run.
What §1–§4 study — collision and blindness under *simultaneous* action — is
precisely the regime that equivalence does not cover, which is why it is
the regime worth studying. Noted here as a reasoned closure, not an
executed experiment, so the distinction is visible.

## Limitations, collected

- No exact planner exists for `DegradingPool` — its state is continuous, not
  a finite alive/dead subset the existing dynamic-programming construction
  can enumerate. §4's comparisons are decentralised-only.
- §3's sequential-within-round stepping is not identical to
  `SimultaneousPool`'s blindness assumption (see caveat in §3).
- §1–§4 use `derive_antibiotic_parameters()` — this repo's own synthetic
  antibiotic-competition engine — for `v_a`, `p_a`, `e_a`; §2 and §3 use
  their environment's real dynamics directly for everything. §5 anchors
  `ρ` against their real `leak` parameter, but `v_a`, `p_a`, and `e_a` are
  not yet re-derived from their reward/dose formulas — the two halves
  share a validated `ρ`, not a single shared parameterisation throughout.
- No single table has all five policies (private/greedy, ECI,
  λ-weighted, their trained RL baseline, planner) compared side by side in
  one environment — the RL baseline only appears in §2/§3, in their
  environment; the other four only appear together in §1/§4, in this
  repo's engine.
- All results here are single-configuration point estimates (one horizon,
  one budget, one patient count), not a sweep across their full config
  space. Robustness to their other environment settings beyond `leak`
  (§5) is untested.

## Scope

This extension is deliberately **not** folded into the JMLR submission as a
new theoretical contribution — consistent with `RESEARCH_PROGRAMME.md`'s own
preference for keeping a result that "can be the thesis" published
separately, rather than diluting either piece's scope. The core paper's
claim stands on the synthetic model and its eleven proven theorems; this
document is the citable record of an independent empirical test of whether
that claim generalises.

## Reproducing this

```bash
# Sections 1 and 4 -- inside this repo, no extra dependencies
python experiments/exp62_amr_lambda_sweep.py
python experiments/exp63_partial_recovery.py

# Sections 2 and 3 -- separate environment, their package
git clone --depth 1 https://github.com/jl56923/abx_amr_simulator.git
python3.11 -m venv venv && venv/bin/pip install -e abx_amr_simulator
# ... see FINDINGS.md#24.2 for the full config-scaffold and training
# commands, and scripts/amr_decentralize_checkpoint.py's header for
# exact setup of section 3.
```
