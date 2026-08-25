# Findings — build session 1

## 1. Competing-risks check (spec action item 2)

**Status: the risk was real, but the framing survives — improved.**

Tsiatis (1975) is the classical non-identifiability result: without an untestable
independence assumption, crude survival probabilities do not identify net survival
probabilities. Crowder (1991) strengthened it. So "latent competing causes are
non-identifiable" is 50-year-old statistics, not a new theorem.

**But the structures differ in a way that matters.** In classical competing risks
the analyst is a *passive observer*: cause of failure is typically observed, and
the non-identifiability concerns the latent marginal distributions. In our setting:

- the cause is *not* observed, and
- one of the two competing risks fires **only when the learner acts**.

The learner therefore controls the exposure of arms to the endogenous risk. It can
manufacture the exogenous variation needed for identification simply by *declining
to pull* — observing departures among unpulled arms isolates `mu`, which then
identifies `p`.

**Reframing:** T1 should not be stated as a non-identifiability theorem (that is
Tsiatis). It should be stated as an **active identification** result:

> Identification of the departure mechanism requires deliberate abstention.
> Characterise the minimum abstention schedule that identifies `(p, mu)`, and
> the regret price paid for it.

That is a bandit result, not a survival-analysis result, and it has no obvious
prior art. It is also strictly more interesting: it converts a known impossibility
into an explicit exploration-cost tradeoff.

---

## 2. Experiment 01 — does irreversibility break departure-blind LinUCB?

Sweep `p` from 0 to 1, no arrivals, no exogenous churn, 60 initial arms.

**Confound found immediately.** Cumulative regret *falls* as `p` rises, because a
higher `p` exhausts the arm pool sooner and the episode ends earlier. Cumulative
regret is not comparable across `p`. Switched to **regret per pull**.

## 3. Experiment 02 — corrected metric, three regimes

### Block A — no arrivals (pure sequencing)

| p | LinUCB regret/pull | IPIC regret/pull |
|---|---|---|
| 0.00 | 0.0186 | 0.0186 |
| 0.10 | 0.0185 | 0.0202 |
| 0.50 | 0.0525 | 0.0725 |
| 1.00 | 0.0841 | 0.2331 |

**Result 1 (positive):** irreversibility genuinely degrades departure-blind
learning — ~4.5x worse per-pull regret at `p=1` versus `p=0`. The problem is real.

### Block B — with arrivals

| p | LinUCB regret/pull |
|---|---|
| 0.00 | 0.0229 |
| 0.50 | 0.0088 |
| 1.00 | 0.0079 |

**Result 2 (the important one):** with arrivals, the effect *reverses* — measured
regret **improves** as irreversibility increases. Destroying pulled arms keeps the
pool churning toward fresh draws, and the myopic benchmark's own value drops with it.

The sign of the headline effect depends on the benchmark. This is not a modelling
detail; it decides whether the paper's central claim is true or false.

### Block C — mixed regime
Same reversal as Block B. Exogenous churn does not rescue it.

---

## 4. Result 3 (negative) — IPIC does not work

The information-per-irreversible-cost heuristic loses to plain LinUCB in **every**
regime tested, catastrophically at `p=1` (0.2331 vs 0.0841 in Block A).

Two distinct reasons, both instructive:

1. **In the pure-sequencing regime the intuition is simply wrong.** With a finite
   pool and no arrivals, every arm is consumed regardless of policy. The problem is
   *ordering*, and the optimal ordering is greedy-descending. Preserving a good arm
   buys nothing because it will be spent anyway. The correct behaviour under
   irreversibility here is to be *more* greedy, not less.

2. **The benchmark cannot reward preservation.** The myopic best-available oracle
   does not plan around departures, so a policy that sacrifices present reward for
   future option value is penalised by construction. Preservation can only be shown
   to pay against a forward-looking benchmark.

---

## 5. What this changes

**The benchmark is now the critical path, not a modelling preliminary.** Both the
sign of the main effect and the evaluability of the proposed algorithm depend on it.
No further algorithm design should happen until the Whittle-index oracle (spec §2.3)
is implemented, or indexability is shown to fail and the α-regret fallback adopted.

**Revised immediate order of work:**

1. Implement the Whittle-index (or Lagrangian-relaxation) oracle. Re-run Blocks A–C
   against it. Confirm whether Result 2's reversal is a benchmark artefact.
2. Restate T1 as active identification with abstention (§1 above).
3. Only then revisit IPIC — and restrict it to the regime where arms are reusable
   *and* the horizon is long enough for option value to matter.

**Honest read:** one session of simulation has already falsified the proposed
algorithm and destabilised the headline claim. That is the testbed doing its job.
Better here than in Phase 4.

---

# Findings — build session 2 (oracle)

## 6. The oracle is greedy. Exactly.

Rather than approximate with a Whittle index, I computed the **exact** optimal
policy by dynamic programming over subsets of available arms
(`attrition/oracle.py`), for 12 random instances spanning
`n ∈ [4,7]`, `T ∈ [3,9]`, `p ∈ {0.2,0.5,0.8,1.0}`, `mu ∈ {0,0.05}`.

**Result: greedy achieves the optimum in every instance.** All gaps sit within
Monte-Carlo error (several are negative, i.e. noise), and the optimal first
action is the highest-value arm in all 12 cases.

**Why — and this is a proof sketch, not a coincidence.** Endogenous death is
geometric, hence memoryless, and rewards are stationary. An arm's *total expected
yield* is therefore independent of when harvesting begins: pulling it now versus
later gives the same expected total. Delay buys nothing and costs the per-round
gap. Exogenous churn strengthens the argument, since a preserved arm may die
while being saved. **There is no option value in preservation.**

### Two consequences

**(a) The benchmark objection dissolves.** The reviewer's PSPACE-hardness concern
about the oracle does not apply: the optimal policy is greedy and exactly
computable. No Whittle relaxation, no indexability condition, no α-approximation.
Spec §2.3 should be rewritten to state and prove greedy-optimality instead.

**(b) IPIC is dead, and provably so.** Its premise — that a valuable arm is worth
preserving — is false in this model. The exp02 failure was not a tuning problem.

---

## 7. The real finding: regret is the wrong objective

exp03 reports per-pull regret *and* per-pull absolute reward across `p`, with
arrivals (`lam=1`):

| p | regret/pull | reward/pull | oracle/pull |
|---|---|---|---|
| 0.00 | 0.0229 | 0.9287 | 0.9515 |
| 0.10 | 0.0125 | 0.7403 | 0.7528 |
| 0.50 | 0.0088 | 0.3991 | 0.4079 |
| 1.00 | 0.0079 | 0.0635 | 0.0714 |

As irreversibility rises, **regret improves by 2.9x while attainable value
collapses by 14.6x.** The learner tracks its optimum ever more closely, and the
optimum is falling off a cliff.

This is not an artefact — greedy is genuinely optimal, so regret against it is
legitimate. It is a property of regret itself: regret is measured against the
optimum *of the instance the learner has already degraded*. Under a consumable
action set, a policy can be near-optimal in regret while destroying almost all
attainable value.

### This reframes the project

The contribution is not an algorithm that minimises regret under irreversibility.
It is the observation that **regret minimisation and value preservation come apart
when actions consume the action set**, plus a proposal for what to measure instead.

Candidate objectives to develop:
- **Competitive ratio against the unconsumed optimum** — compare against the best
  achievable had the action set never degraded, rather than against the degrading
  optimum.
- **Value-preservation metric** — explicitly score the terminal state of the action
  set, not just cumulative reward.
- **Terminal commitment (T5)** — unchanged and now relatively more important, since
  it is the one sub-problem where a genuine planning tradeoff survives.

This is a "name the phenomenon" contribution of the kind that travels further than
a tighter bound: an agent optimising standard regret can burn irreplaceable
resources and its own metric will report success.

## 8. Revised next actions

1. Prove greedy-optimality formally (interchange argument + memorylessness).
   Replace spec §2.3.
2. Formalise the regret/value-preservation divergence — this is now the lead result.
3. Restate T1 as active identification with abstention (session 1, §1).
4. Drop IPIC. Redirect algorithm design toward the new objective, not regret.

---

# Findings — build session 3 (scenario battery)

## 9. Where greedy-optimality holds, and where it breaks

Exact DP against greedy across five scenarios, 8 random instances each
(`experiments/exp04_scenario_battery.py`):

| Scenario | Greedy suboptimal in | Gap |
|---|---|---|
| S1 heterogeneous irreversibility (`p_a` varies) | 0/8 | 0.000 |
| S2 discounting (γ=0.8) | 0/8 | 0.000 |
| S3 rotting (decay in arms consumed) | 0/8 | 0.000 |
| S4 commons degradation (all rewards scaled by survivors) | 0/8 | 0.000 |
| **S5 competitive release (asymmetric externality)** | **8/8** | **12–23%** |

Greedy survives everything except S5, and in S5 it fails in *every* instance by a
large margin.

### The distinguishing condition

S4 and S5 both couple arms. Only S5 breaks greedy. The difference is **asymmetry**:

- In S4 the degradation depends only on *how many* arms have died. With homogeneous
  `p`, that trajectory is the same whichever arm is pulled — the externality is
  independent of the action, hence irrelevant to the choice.
- In S5, consuming a *sensitive* arm imposes a burden while consuming the resistant
  arm does not. The action now determines the degradation path.

**Conjecture (to prove).** Greedy is optimal iff the departure-and-degradation
process is action-symmetric. Asymmetric externality is necessary and sufficient
for planning to have value.

This is a sharper theorem than "greedy is optimal", and it is falsifiable. It also
explains session 2 cleanly: the base model has no externality at all, so greedy
was optimal by construction, and IPIC was doomed for a structural reason rather
than a tuning one.

### Domain validation

S5 is the competitive-release structure from adaptive oncology. The model
reproduces from first principles the known clinical result that maximum-tolerated
dose is beaten by a policy that preserves the sensitive population. That is a
useful sanity check: the framework recovers a real, independently-established
phenomenon rather than a simulation artefact.

---

## 10. Learning in the coupled regime — the externality charge works

Given that the 12–23% gap exists, can a learner capture it? Two heuristics failed
first (`exp05`): a constant charge was too weak, a horizon-scaled charge too
strong — it refused sensitive arms early and then burned them late, accumulating
burden exactly when it still had to be paid.

Sweeping the charge coefficient (`exp06`) settles it:

| δ (coupling strength) | κ=0 (blind) | best κ | gain |
|---|---|---|---|
| 0.05 | 17.64 | 18.35 | **+4.0%** |
| 0.10 | 8.46 | 9.84 | **+16.2%** |
| 0.20 | −9.94 | −7.23 | **+27.2%** |

**Result: gains scale with coupling strength, and the charge saturates for κ ≳ 5.**

Saturation means the winning policy is a corner solution: never consume the
externality-causing class at all. That is the adaptive-therapy insight in miniature,
but it also flags a limitation of this instantiation — with a single resistant arm,
abstention is trivially available. The next model refinement should use **graded**
externalities across many arms rather than a binary sensitive/resistant split, so
the optimal policy is interior and the charge magnitude actually matters.

---

## 11. Consolidated position after three sessions

**Established:**
1. Greedy is exactly optimal in the independent-arm model (12/12 exact-DP instances).
2. It survives heterogeneous `p`, discounting, rotting, and symmetric commons decay.
3. It breaks — 8/8, by 12–23% — under asymmetric externality.
4. Regret and value diverge sharply under consumption: regret improved 2.9× while
   attainable value fell 14.6×.
5. An externality charge captures much of the gap, scaling with coupling strength.

**Dropped:** IPIC as originally motivated; the Whittle/α-regret machinery (not
needed — the oracle is greedy where greedy holds, and exact DP is tractable at the
scales that matter for theory).

**Now the lead candidates, in order:**
- **C1** Action-symmetry characterisation: greedy optimal ⟺ symmetric externality.
- **C2** Regret/value divergence under consumable action sets, and a replacement
  objective (competitive ratio against the unconsumed optimum).
- **C3** Active identification with abstention (session 1, §1).

C1 and C2 are complementary: C1 says when planning matters, C2 says the standard
metric will not tell you that it does.

**Next:** graded-externality model, then formalise C1.

---

# Findings — build session 4 (graded externality, C1 confirmed)

## 12. C1 sharpened and confirmed: the invariant is `p_a · e_a`

Generalised S5 to graded externalities: each arm carries a coefficient `e_a`, and
its death adds `δ e_a` permanent burden. The refined conjecture:

> greedy optimal ⟺ the expected marginal externality `κ_a = p_a · e_a`
> is constant across arms

Five structural cases, exact DP, 10 instances each (`exp07`):

| case | std(κ) | mean gap | broken |
|---|---|---|---|
| A  `p` varies, `e = 0` | 0.0000 | 0.000% | 0/10 |
| B  `p` constant, `e` constant | 0.0000 | 0.000% | 0/10 |
| C  `p` constant, `e` varies | 0.2541 | 6.279% | 10/10 |
| D  `p` varies, `e` constant | 0.2099 | 5.377% | 10/10 |
| **E  both vary, product held constant** | **0.0000** | **0.000%** | **0/10** |

**Case E is the decisive test.** Both factors vary widely; only their product is
pinned. Greedy is exactly optimal. Neither `p` nor `e` alone is the invariant —
the product is. Case D was a new prediction and it broke greedy as predicted.

Dispersion sweep confirms monotonicity: gap runs 0.000% → 0.523% → 2.481% →
4.467% → 8.210% → 19.076% as std(κ) goes 0 → 0.427.

## 13. The corrected index

The characterisation hands over the correction directly. Pulling `a` creates
expected permanent burden `δ κ_a` charged against every remaining round, so

```
I(a, t) = v_a − δ · κ_a · (T − t)
```

Against exact DP (`exp08`), this captures **52% → 92%** of the optimality gap,
with capture improving as coupling strengthens — best exactly where greedy is
worst. Residual gap 0.9–3.7%, attributable to `(T−t)` overstating the effective
remaining horizon.

## 14. Where the three threads converge

Theorem 1 says `κ_a = p_a e_a` is the quantity that determines optimal play.
Session 1 established that `p_a` is **not identifiable** without deliberate
abstention from pulling. Therefore:

> the quantity that governs optimal behaviour is precisely the one that cannot
> be estimated without paying an explicit exploration cost.

That is the tightest framing the project has had, and it unifies C1 (structure),
C3 (identification), and the learning problem into a single statement.

Proof sketches and open technical questions: see `THEORY.md`.

---

# Findings — build session 5 (learning)

## 15. Index refinement: the simple charge is already near-optimal

Attempted to close exp08's residual gap by replacing `(T−t)` with an effective
horizon `H(S,t) = min(T−t, Σ_{i∈S} 1/p_i)`. First run showed no difference —
because with those parameters `Σ 1/p_i > T−t`, so the min never bound. Retested
in a binding regime (n=5, T=16, p∈[0.7,1.0]):

| std(κ) | greedy gap | index `(T−t)` | index `H` |
|---|---|---|---|
| 0.096 | 2.90% | **0.94%** | 2.42% |
| 0.323 | 11.22% | **0.57%** | 4.05% |
| 0.793 | 23.17% | **0.23%** | 2.27% |

The refinement is **worse**, and the simple index reaches **99.0% capture** in the
high-consumption regime. Hypothesis for why: `(T−t)` over-charges relative to the
true burden horizon, and that over-charge happens to also proxy the option-loss
term the index otherwise ignores — it is doing double duty. Worth investigating,
but the practical conclusion is that the simple index needs no repair.

---

## 16. The main result: irreversibility destroys the data needed to price it

Online learning version (`exp10`): estimate `v`, `p`, `e` from observed rewards
and death events, then act on the index. Rewards are linear in the unknowns, so
least squares should work.

**It does not.** The learned policy captured ~0% of the gap, while the oracle
index captured up to 47%:

| spread | blind | learned | oracle index | gap to oracle |
|---|---|---|---|---|
| 0.00 | 4.323 | 4.323 | 4.600 | 6.0% |
| 0.60 | 4.548 | 4.561 | 5.927 | 23.1% |
| 1.00 | 4.503 | 4.503 | 6.606 | **31.8%** |

Diagnosis (`exp11`) shows the cause is structural, not a bug. To estimate `e_i`
you must compare reward levels before and after arm `i` dies. Death is
irreversible, so **each arm supplies exactly one such transition**. Estimate
quality then depends entirely on how many rounds are spent in each burden state,
and fast consumption empties the pool before any averaging can occur:

| p range | rounds of data | rounds per burden state | corr(ê, e) |
|---|---|---|---|
| [0.90, 1.00] | 8.5 | 1.1 | **+0.118** |
| [0.50, 1.00] | 11.0 | 1.4 | +0.132 |
| [0.25, 0.50] | 23.0 | 2.9 | +0.261 |
| [0.10, 0.25] | 44.7 | 6.7 | +0.531 |
| [0.04, 0.10] | 60.0 | 21.5 | **+0.828** |

Monotone. At full irreversibility the externality is essentially unlearnable.

### The tension this exposes

Theorem 1 says `κ_a = p_a · e_a` governs optimal play. But `p_a` enters twice,
in opposite directions:

- **higher `p_a` ⇒ larger `κ_a`** ⇒ the externality matters more;
- **higher `p_a` ⇒ fewer rounds per burden state** ⇒ the externality is harder
  to estimate.

> **The parameters that determine optimal behaviour under irreversibility become
> less estimable exactly as they become more important.**

This is not a sample-complexity inconvenience that more data fixes. The data is
generated by the same consumption that creates the harm, so the budget is
structurally capped. It is the sharpest statement the project has produced and
is the natural headline claim.

**Practical reading for agent systems:** an agent operating in a regime of
irreversible actions cannot learn the true cost of those actions from experience,
because acquiring each observation destroys the resource being priced. Prior
knowledge of `e` — specification, not learning — is the only route.

---

# Findings — build session 6 (prior art, and a policy that beats the index)

## 17. Externality prior-art check

Two neighbours found. Neither covers this model, but both are mandatory citations.

**Bandit Learning with Positive Externalities** (Shah, Johari, Scheinkman,
NeurIPS 2018). Externalities live in the *arrival process*: users satisfied by an
arm attract similar users, so preferences self-reinforce. Notably they prove
**UCB incurs linear regret** under this structure and develop Balanced
Exploration with matching lower bounds.

*Difference:* their externality is **positive**, acts on **arrivals**, and does
not consume arms. Ours is **negative**, acts on **rewards**, and is triggered by
**irreversible arm destruction**. The parallel worth drawing explicitly: in both
settings a standard departure-blind algorithm fails badly — evidence that
externality structure, of either sign, breaks classical guarantees.

**Bandits with Knapsacks** (Badanidiyuru, Kleinberg, Slivkins; Li–Sun–Ye and
others). Playing an arm consumes resources; the process halts when a budget is
exhausted. Structurally the closest neighbour.

*Difference:* BwK budgets are **global and exogenous**, and consumption does not
alter other arms' reward distributions. Here consumption **permanently degrades
every future reward** through the burden term. That is an externality, not a
budget, and it is what makes `κ = p·e` rather than a consumption rate the
governing quantity.

**Verdict:** the model is not covered, but positioning must be explicit against
both. Related work is materially thicker than v0.2 assumed — this is the third
prior-art correction of the project.

---

## 18. Rollout: a policy that beats the index

exp13 showed no closed form exists for the option-loss term the index omits.
Standard remedy: one step of **policy improvement**. Treat the index rule as base
policy `π₀` and act greedily with respect to *its* value function. The marginal
term `V_{π₀}(S) − V_{π₀}(S∖{a})` is exactly the option value the index was
missing, so it is captured implicitly rather than modelled.

Gap to exact optimum (`exp14`, n=6, T=8):

| std(κ) | greedy | index | rollout(greedy) | **rollout(index)** |
|---|---|---|---|---|
| 0.180 | 7.109% | 3.042% | 0.398% | **0.186%** |
| 0.353 | 10.846% | 1.483% | 0.208% | **0.137%** |
| 0.572 | 20.047% | 2.612% | 0.843% | **0.358%** |
| 0.594 | 16.607% | 1.602% | 0.512% | **0.156%** |

**Rollout(index) reaches 99.5–99.9% of optimal.** Rollout over plain greedy is
nearly as good (0.2–0.8%), which is itself notable: one improvement step from any
sensible base recovers most of the gap.

## 19. It survives at scale

Exact policy evaluation memoises over subsets and is exponential, so at scale the
base policy is evaluated by simulation: for each of the top-`m` candidates by
index, run `K` forward episodes under the base policy. Cost is `m·K·T` per
decision — polynomial.

At **n = 40, T = 40** (exact DP infeasible), 12 seeds:

| spread | greedy | index | rollout | index vs greedy | rollout vs index |
|---|---|---|---|---|---|
| 0.30 | 29.841 | 30.511 | 30.520 | +2.2% | +0.0% |
| 0.80 | 29.737 | 32.082 | 32.508 | +7.9% | +1.3% |
| 1.50 | 28.292 | 32.961 | 34.040 | **+16.5%** | **+3.3%** |

Both effects survive and both scale with coupling strength. The index does the
heavy lifting (+16.5% over greedy); rollout adds a further +3.3% on top, for
roughly **+20% total** at strong coupling.

**Practical reading:** use the index when compute is tight — it is a closed-form
scalar per arm. Add rollout when the externality is strong and decisions are
expensive enough to justify `m·K·T` simulation per step.

---

# Findings — build session 7 (agent instantiation)

## 20. Zero-regret ruin in an agent tool ecosystem

The abstract model instantiated where the AI community will recognise it
(`exp17`). An orchestrator routes tasks across API-backed tools. Each tool has
quality `v_a` and fragility `p_a` — heavy use trips a rate limit, exhausts a
quota, or gets a key revoked, permanently removing it for the session. Tools
share upstream infrastructure, so burning one degrades capacity for the rest by
`δ e_a`, where `e_a` measures how much shared infrastructure the tool sits on.

The mapping is not contrived: shared rate limits across a provider's API surface,
noisy-neighbour effects on shared tenancy, and reputation throttling of an IP
range all have exactly this structure.

| coupling δ | V* | greedy | index | greedy loss | **greedy regret** |
|---|---|---|---|---|---|
| 0.00 | 27.60 | 27.60 | 27.60 | 0.00 | **0.0000** |
| 0.01 | 27.46 | 26.69 | 27.15 | 0.77 | **0.0000** |
| 0.03 | 27.28 | 24.87 | 27.06 | 2.41 | **0.0000** |
| 0.06 | 27.07 | 22.13 | 26.99 | 4.94 | **0.0000** |
| 0.10 | 26.83 | 18.49 | 26.72 | **8.34** | **0.0000** |

At δ=0.10 the greedy router destroys **31% of attainable value** while reporting
zero regret. The index router recovers to within 0.4% of optimal.

## 21. The session trace — the artefact to lead with

Identical seed, identical tools, two routers. This is the whole argument in
fourteen lines.

**Greedy router** (regret column = what an ops dashboard displays):

| step | tool called | realised | regret | tools left |
|---|---|---|---|---|
| 0 | premium-search | 1.150 | **0.0000** | 6 |
| 1 | bulk-search | 1.038 | **0.0000** | 5 |
| 2 | premium-codegen | 0.844 | **0.0000** | 4 |
| 5 | bulk-scraper | 0.818 | **0.0000** | 3 |
| 6 | internal-db | 0.470 | **0.0000** | 2 |
| 7–13 | internal-db | 0.470 | **0.0000** | 2 |

**Index router**, same seed:

| step | tool called | realised | regret | tools left |
|---|---|---|---|---|
| 0 | premium-search | 1.150 | 0.0000 | 6 |
| 1–4 | premium-codegen | 0.988 | 0.0500 | 5 |
| 5–13 | internal-db | 0.782 | **0.2500** | 4 |

### Read the two tables against each other

The greedy router reaches `internal-db` at 0.470 per call with **two tools left**
and a flawless regret record. The index router is sitting at 0.782 per call —
**66% higher** — with **four tools left**, while *reporting regret of 0.25 per
step*.

**The policy that looks worse on the dashboard is the one that is actually
winning.** Greedy's zero-regret record is not evidence of good behaviour; it is
an artefact of measuring against a benchmark that degrades in lockstep with the
damage being done. Every call greedy makes is optimal *given the ecosystem it has
already ruined*.

The index router's visible "regret" is the price of preservation, and it is
recorded as a defect by the standard metric.

### Why this is the artefact to lead with

It requires no bandit theory to read. An engineer looks at two logs, sees the
green one arriving at a worse outcome with fewer resources remaining, and the
point lands immediately. It also maps onto operational experience people already
have — burning a shared quota, tripping a provider-wide limit, losing an API key
mid-run.

**Paper structure this suggests:** open with the trace, then the theory
(T1 → T3 → T4), then the domains. Lead with the thing that makes people feel it.

# Findings — build session 8 (real data, live demos, and agent count at scale)

## 22. Both multi-agent claims hold from m=6 up through m=20,000

Every prior multi-agent check in this repo — tests, experiments, examples —
topped out at m=6 agents, because the exact-planner PoA baseline
(`planner_value_simultaneous`) is exponential in m and simply can't go
further. The simulation machinery itself (`SimultaneousPool.step`,
`simulate_population_simultaneous`) has no such ceiling; it had just never
been asked to run past it. `exp61_population_scale.py` did, on real data
(`antibiotic-stewardship-real`, 200 real WHO-derived arms), not synthetic.

**Theorem 6 (sequential equivalence) holds exactly at m=200** — 33x past
anything previously tested. m greedy agents turn-taking over a shared
`m·T`-pull budget match a single learner over the same pulls to float
precision, at every m checked (2, 5, 10, 25, 50, 100, 200), and stay fast
(11ms at m=200, not a scaling concern).

**Simultaneous-action price of anarchy saturates to an EXACT step
function, and the mechanism is provable, not just empirical.** First pass
(pushed to m=190) looked like it might just be "delta is small here" —
per-agent value drops only ~0.4% from m=2 to m=190 (4.9250 → 4.9041). But
pushing further (m=500 through m=20,000 — 100x the pool size) found the
value pins to *exactly* 4.9041 the whole way, and a fine sweep (m=50 to
120 in steps of 5) found why: it's a single step at m=85, not a curve that
merely looks flat at low resolution. The real mechanism is in
`SimultaneousPool.step()` itself, not this scenario's `delta`: every
colliding agent's reward is computed from the same pre-round burden
(collision count never changes any individual's reward for that round),
and the destruction check `if self.alive[a] and rng.random() < p[a]`
short-circuits once the arm dies, so once m is large enough that some pull
destroys the round's contested arm, every pull after that consumes zero
further randomness and changes nothing else about the trajectory. Below
that threshold the arm can survive the round instead, sending the rest of
the run somewhere genuinely different. This predicts an exact step for
*any* scenario, not a delta-dependent curve — the delta-based explanation
in the first pass was a plausible-looking guess that didn't survive
pushing the scale further, corrected here rather than left standing.

Added two permanent regression tests: `test_theorem6_holds_at_m50` (modest
scale, fast enough for the normal suite) and
`test_simultaneous_price_of_anarchy_saturates_past_collision_threshold`
(checks m=50 vs m=500 land on the exact same per-agent value, not pinning
the threshold itself, which is scenario-specific). The full sweeps —
m=200, m=20,000, and the fine-grained step search — stay in
`experiments/exp61_population_scale.py`, which now also prints the
mechanism check directly (`saturation_mechanism_check`).

## 23. The first trustworthy real-LLM validation result — and it matches neither prior guess

Closing a thread that had been open since the first attempt at this: every
earlier try at running a real model (not `MockLLMClient`) against the pharma
scenario was contaminated in one way or another — silent empty-response
fallback (hidden reasoning tokens exhausting the budget with no exception
raised), then daily quota exhaustion mid-run. Both looked like real findings
at the time and were reported as such, then walked back once the
contamination was found. This time: two independent runs (20 seeds, then 50
seeds, 600 total real calls to `openai/gpt-oss-20b` via Groq's free tier),
both 100% genuine parseable responses, zero fallback, confirmed by the
diagnostic built specifically to catch this.

```
              20 seeds          50 seeds
              value   regret    value   regret
greedy        2.606   0.000     2.753   0.000
real model    3.127   0.595     3.352   0.616
eci           3.841   0.491     3.833   0.494
mock          3.606   0.453     3.809   0.402
```

The pattern replicates across both samples, which is what makes it
trustworthy rather than a one-off: the real model's value sits genuinely
between Greedy and ECI (42-55% of the way from Greedy to ECI across the two
runs) — real evidence it isn't simply Greedy-blind to the externality. But
its regret is *higher than ECI's own regret* in both runs, and its value
falls short of what `MockLLMClient` predicts (which tracks ECI closely) in
both runs. So the two hypotheses floated during the contaminated attempts —
"real models behave like Greedy" and "the mock is a reasonable stand-in" —
are both wrong, in a consistent, replicated way, not just once.

The honest reading: the model is doing something distinctly its own, not
approximating either classical policy. It picks up real value gains from
apparent externality-awareness, but inconsistently — round-to-round variation
that a deterministic index policy (ECI) doesn't have, costing it regret ECI
never pays for the same or better value. That inconsistency is plausibly the
real mechanism (an LLM's choice isn't a fixed formula the way ECI's is), but
that is not yet verified directly — a natural next step would be inspecting
the actual per-round transcripts to check whether disagreement-with-ECI
rounds are where the regret concentrates, rather than asserting the
mechanism from the aggregate numbers alone.

Two runs at modest seed count is real progress, not a large-sample claim.
The next honest step, if this matters more precisely, is more seeds at the
same clean settings, not a different model or scenario chosen because it
might look more dramatic.

## 24. AMR domain extension — the divergence transplants, and their own dial checks out

*(Readable summary of this section and §25: `docs/amr-stewardship-extension.md`.)*

Two separate results, run together because they answer the same question from
opposite ends: does the regret-harm divergence survive contact with a domain
ATTRITION didn't build, and is the nearest independent competitor's proposed
fix (`abx_amr_simulator`, Lee & Blumberg 2026, arXiv:2603.11369) actually a
fix, or does it just relocate the problem?

### 24.1 In-repo: their reward dial, transplanted into genuine multi-agent play

`abx_amr_simulator` internalises the resistance externality with one number on
one policy: `R_overall = (1-lambda)*individual + lambda*community`. Their
model has exactly one decision-maker, so there is nothing to coordinate.
`attrition.simultaneous.decentralised_value_simultaneous` now has
`rule="lambda_weighted"` (exact transplant of their dial, `lam=1,
obs_noise=0` reduces to the existing `eci` rule bit-for-bit — verified in
`tests/test_theorems.py`), run against `m=3` simultaneously-acting prescribers
sharing `derive_antibiotic_parameters()`'s pool (`experiments/
exp62_amr_lambda_sweep.py`).

Result: the gap between what each prescriber's own score reports and the
true damage to the pool falls from 5.71 (lam=0) to 0.64 (lam=1) at zero
observation noise -- reward-shaping genuinely helps, by a lot -- but price of
anarchy against the planner optimum bottoms out at 1.07, not 1.00. Add
observation noise (their own partial-observability knob, transplanted) and
the lam=1 floor rises to 2.20 at noise = 2*std(kappa), more than 3x the
noise-free floor. The residual is exactly what Theorem 6 predicts: a
coordination cost a single-agent reward dial cannot remove, because their
environment has no second agent to collide with.

### 24.2 External: reproduced their own environment, their dial checks out on its own terms

Cloned `jl56923/abx_amr_simulator` (BSD-3-Clause, commit as of 2026-08-24),
installed under Python 3.11 (their pinned `torch==2.2.2` has no wheel for
3.13), trained their default PPO config (2 antibiotics, 5 patients/step, 500
steps/episode, 25 training episodes -- their own tutorial default, ~18s
wall-clock each) at both dial extremes, no other changes:

| lambda | mean training reward (reported) | final AMR level, antibiotic A | final AMR level, antibiotic B | patient clinical benefit only |
|---|---:|---:|---:|---:|
| 0.0 | -370.95 | 0.9997 | 0.9933 | -1854.7 |
| 1.0 | -244.07 | 0.7461 | 0.2506 | -1850.1 |

Reproduction commands (from a clean checkout):

```bash
git clone --depth 1 https://github.com/jl56923/abx_amr_simulator.git
python3.11 -m venv venv && venv/bin/pip install -e abx_amr_simulator
venv/bin/python -c "from abx_amr_simulator.utils import setup_config_folders_with_defaults; \
  from pathlib import Path; setup_config_folders_with_defaults(Path('experiments'))"
venv/bin/python -m abx_amr_simulator.training.train \
  --umbrella-config "$(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml" \
  -p training.run_name=lambda0
venv/bin/python -m abx_amr_simulator.training.train \
  --umbrella-config "$(pwd)/experiments/configs/umbrella_configs/base_experiment.yaml" \
  -p training.run_name=lambda1 -p reward_calculator.lambda_weight=1.0
```

Final AMR levels and clinical benefit are read from each run's
`results/<run_name>_*/eval_logs/eval_0004_step_12500.npz` (`actual_amr_levels`
and `individual_rewards`).

At lambda=0 (pure private reward) both antibiotics saturate to near-total
resistance by episode's end. At lambda=1, antibiotic B's resistance is cut
from 99.3% to 25.1% -- and patient clinical benefit is statistically
unchanged (-1850.1 vs -1854.7, well inside the run-to-run noise a single
seed carries). Their claimed mechanism is real: in a single centralised
policy, the dial buys real stewardship at close to zero clinical cost. This
is a genuine, independently-produced confirmation of their result, not a
takedown of it -- worth stating plainly, since 24.1 is not a rebuttal of
24.2, it is the next question 24.2's own environment cannot ask.

### 24.3 What 24.1 and 24.2 don't, by themselves, establish

24.1 and 24.2 are run on two different environments (this repo's antibiotic-
competition engine vs. theirs) -- independently-supported halves of one
argument, not yet a single unified head-to-head run. The strongest version
would take the *literal trained checkpoint from 24.2* and embed it as
several decentralised agents inside their own environment, not this repo's
abstraction of it. That turned out to be tractable without an adapter --
see 24.4.

### 24.4 The literal checkpoint, decentralised, inside their own environment

`ABXAMREnv.__init__` already accepts a pre-built `amr_dynamics_instances`
dict, flagged in its own docstring as "OWNERSHIP (for future multi-agent
refactor)" -- unused by their training/CLI code, but a real seam. That's
enough to run `m=3` independent copies of the actual trained lambda=1.0
checkpoint from 24.2, each with its own patient generator and reward
calculator, sharing one injected pair of `AMR_LeakyBalloon` objects instead
of each getting its own (`scripts/amr_decentralize_checkpoint.py`, run in a
separate environment against the cloned package -- not part of this
package's own dependency tree, see that script's header for exact setup).
No adapter between action spaces was needed: each decentralised copy sees
the same 5-patient, MultiDiscrete observation/action shape it was trained
on, so `model.predict(obs)` works unmodified. Only the AMR dynamics object
is shared.

Two conditions, same checkpoint, 12 episodes each, only one thing differing
(a private balloon pair per prescriber vs. one shared pair):

| condition | total reward (3 agents) | final AMR, antibiotic A | final AMR, antibiotic B |
|---|---:|---:|---:|
| siloed (no externality) | -1109.72 | 0.7461 | 0.2506 |
| shared (real externality) | -1151.15 | 0.9691 | 0.1993 |

The siloed row exactly reproduces 24.2's single-agent lambda=1.0 result
(0.7461 / 0.2506), which is itself a useful consistency check that the
harness is wired correctly. Under sharing, antibiotic A's resistance climbs
from 74.6% back to 96.9% -- most of the way to the *no-stewardship* (lambda=0)
level of 99.97% from 24.2 -- using a policy that was, by 24.2's own numbers,
buying real stewardship at close to zero clinical cost when run alone. The
policy was never trained to know other copies of itself exist; decentralising
it erases most of what its own community-reward term bought. Antibiotic B
moves the other way (0.2506 -> 0.1993), plausibly reallocation pressure
through the crossresistance matrix, not dug into further here -- reported as
observed, not explained away.

One caveat stated plainly, not smoothed over: within a round the three
environments are stepped sequentially (`env0.step()`, then `env1`, then
`env2`), so a little within-round information leaks between them -- this is
not bit-for-bit `SimultaneousPool`'s BLINDNESS assumption (burden visible
only from the *next* round). Decentralised and non-coordinating either way,
but the distinction is real. This closes the gap 24.3 flagged: 24.1's
transplanted-rule finding, 24.2's confirmation their dial works alone, and
24.4's demonstration that the same real policy loses most of that benefit
once decentralised, are now three converging, mutually load-bearing results
rather than two independent halves and an open question.

## 25. Partial recovery breaks the divergence's sign, not just its size

The plan for the AMR extension (see the domain-extension proposal) flagged
that calling AMR "irreversible arm consumption" overclaims: real resistance
accumulates and decays, it does not binary-flip. `DegradingPool`
(`attrition.simultaneous`) generalises `SimultaneousPool`'s permanent
destruction into a continuous, partially-recoverable state, `s_a(t+1) =
rho*s_a(t) + (1-rho)*e_a*hit_a(t)`, and `experiments/
exp63_partial_recovery.py` asks the planned question directly: does the
regret-harm divergence from exp62 survive when consumption is no longer
permanent?

**It does not survive unchanged -- it survives with the sign flipped,
depending on the scoring rule.** At `lam=0` (greedy, blind to burden) the
gap stays positive at every `rho` but shrinks toward 0 as `rho -> 1`, because
at this horizon (T=9) high-persistence harm has not had time to build to its
ceiling. At `lam=1` (eci-style pricing, which charges an arm's full
externality the instant it is hit, as ECI's `delta*p*e*(T-t)` term assumes
permanence) the gap is **negative at every rho tested** -- the policy
*overcharges* relative to realised harm, because its pricing formula was
built for a permanence assumption partial recovery breaks. Concretely (T=9,
delta=0.35, m=3, `derive_antibiotic_parameters()`):

```
lam=0.0                          lam=1.0
 rho  system  reported  gap       rho  system  reported   gap
0.00   11.640   15.383  3.743    0.00   13.399   13.114  -0.285
0.50   12.106   15.383  3.277    0.50   13.505   13.114  -0.392
0.99   15.219   15.383  0.165    0.99   13.932   13.114  -0.819
```

Two things worth stating plainly rather than smoothing over:

- The exp62 headline ("reward-shaping narrows a positive gap to a positive
  floor") does not transfer here as a template. Partial recovery is a
  qualitatively different regime, not exp62 with smaller numbers, and it was
  checked rather than assumed to pattern-match.
- **`DegradingPool` does NOT reduce to `SimultaneousPool` as rho -> 1**,
  checked empirically (a long-horizon run at rho=0.999 clears more than 2x
  the value of the permanent-destruction model at the same horizon, not a
  converging one -- see `test_degrading_pool_does_not_reduce_to_
  simultaneous_pool_at_rho_one`). The two formalise different notions of
  irreversibility: `SimultaneousPool` permanently removes an arm from the
  *choice set*; `DegradingPool` permanently prices an arm at its full
  externality once saturated, but the arm stays selectable throughout. Real
  antimicrobial resistance is closer to the second -- a resistant antibiotic
  can still be prescribed, it is just costly to. An earlier informal write-up
  of this extension also had rho's persistence direction backwards (labelled
  rho=0 "permanent"); corrected in `DegradingPool`'s docstring rather than
  silently fixed.

No exact planner exists for `DegradingPool` (state is continuous, not a
finite alive/dead subset the existing DP construction can enumerate), so
these are decentralised-only comparisons against each other, not against a
system optimum -- a real, stated limitation, not one that undermines the
sign-flip finding, which does not depend on a planner ceiling.

## 26. Closing the gaps an audit found in sections 24-25

A direct audit against everything agreed across the AMR discussion turns
turned up four items planned but not delivered. Two are closed here; two
are closed by explaining why no further experiment is needed, not by
running one -- recorded as such rather than quietly marked done.

**Closed: the variable mapping.** Stated as an explicit table in
`docs/amr-stewardship-extension.md#0` (arm/v_a/p_a/e_a/kappa_a -> antibiotic/
benefit/impact/externality/expected harm). Previously only implicit in
docstrings and prose.

**Partially closed: the robustness leg.** The original plan was to re-derive
`(v, p, e)` from their `AMR_LeakyBalloon` sigmoid dynamics and re-run
sections 24.1/25's sweeps against it. What's actually defensible without
reverse-engineering their full reward/dose formula: `leak` is exactly a
per-step decay rate on accrued pressure, the same role `1-rho` plays in
`DegradingPool`. Reading it directly from their config: antibiotic A
(`leak=0.20`) implies `rho=0.80`, antibiotic B (`leak=0.10`) implies
`rho=0.90` -- both inside the interior of section 25's already-swept range,
not outside it. So the sign-flip and floor findings were already checked at
`rho` values bracketing their real system, not only at convenient extremes.
`v_a`, `p_a`, and `e_a` are not yet re-derived from their formulas -- see
`docs/amr-stewardship-extension.md#5` for the precise scope of what this
does and doesn't establish.

**Closed by explanation, not experiment: sequential vs. simultaneous for
`lambda_weighted`.** Theorem 6 is a structural equivalence (m agents over T
rounds under turn-taking = one learner over m*T pulls) proved independent of
which per-round scoring rule agents use, so it covers `lambda_weighted`
without a rule-specific re-derivation. Simultaneous action -- collision,
blindness -- is precisely the regime that equivalence excludes, which is why
sections 24-25 study that regime and not turn-taking. See
`docs/amr-stewardship-extension.md#6`.

**Still open: one unified five-policy table.** Private/greedy, ECI,
lambda-weighted, and the planner appear together in sections 24.1/25, in
this repo's engine. The trained RL baseline appears only in sections
24.2/24.4, in their environment. No single run has all five side by side on
common ground. Recorded as open, not attempted here.

## 27. Election track, Phase 1: a real, validated population engine — no theorem claims yet

First concrete step on the election track, per the parallel-track plan:
population infrastructure now, campaign/persuasion agents and any
regret-harm claim later, once this layer is validated on its own. Mirrors
the AMR extension's discipline (`docs/amr-stewardship-extension.md`) --
real data, checked claims, honest scope -- applied to a domain this repo
had zero code in before this session.

**Data.** `lokdhaba.ashoka.edu.in`'s live query API
(`POST /api/data/api/v2.0/getDerivedData`) returned 502 Bad Gateway when
checked (2026-08-25), independent of query size -- a real infrastructure
outage, not a client-side problem, confirmed by driving the actual site in
a browser and reading the network requests, not assumed. The Trivedi
Centre's own GitHub-hosted archive (`github.com/tcpd/tcpd-ld-data-archive`,
"LD data as on June 2023", per-state gzipped CSVs) was usable and is
arguably more reproducible regardless: static, versioned files rather than
a live query against infrastructure that may or may not be up for someone
else re-running this. `scripts/fetch_election_data.py` pulls ten real,
named, verified-to-exist constituencies (2019 General Election, the most
recent year in the archive) into `attrition/data/tcpd_lokdhaba_ge.json`:

| state | constituency | note | electors | turnout | top party |
|---|---|---|---:|---:|---|
| Gujarat | Ahmedabad East | urban | 1,811,851 | 61.61% | BJP 67.2% |
| Maharashtra | Mumbai North | urban | 1,647,350 | 60.08% | BJP 71.4% |
| Uttar Pradesh | Lucknow | urban | 2,040,367 | 54.72% | BJP 56.7% |
| Uttar Pradesh | Azamgarh | rural | 1,789,168 | 57.52% | SP 60.4% |
| West Bengal | Kolkata Dakshin | urban | 1,728,851 | 69.79% | AITC 47.5% |
| Tamil Nadu | Chennai Central | urban | 1,332,300 | 58.95% | DMK 57.1% |
| Karnataka | Bangalore Rural | rural | 2,497,458 | 64.94% | INC 54.1% |
| Bihar | Patna Sahib | urban | 2,146,008 | 45.77% | BJP 61.9% |
| Kerala | Thiruvananthapuram | urban | 1,371,427 | 73.66% | INC 41.2% |
| Assam | Gauhati | northeast | 2,181,319 | 80.86% | BJP 57.2% |

**Architecture.** `attrition/election/` is a subpackage, not a
`SCENARIOS` entry -- the population layer (real data pipelines, no single
`(v, p, e)` triple) doesn't fit the scenario-registry contract every other
domain in this repo terminates at, per the domain-extension discussion's
unification guidance. `lokdhaba.py` holds the real, observed data;
`population.py` holds `SyntheticVoter` (deliberately thin: constituency,
turned-out, voted-for -- nothing else) and `synthetic_voter_population`,
which samples turnout as Bernoulli(real turnout rate) and, among those who
turn out, party choice from a categorical distribution weighted by real
vote share. This is Level 0 of the plan's three-tier agent hierarchy: no
LLM calls, millions-scalable, exact by construction.

**Validation ladder, level 1.** `validate_population` checks the sampled
population reproduces the real turnout and vote-share numbers within 0.5
points, for all ten constituencies, not asserted once and assumed to
generalise (`tests/test_election.py`). All ten pass.

**A real bug, caught by the validation function it was building, not
shipped.** Several candidates in the real data share a party label --
every independent candidate is coded "IND" -- so `AHMEDABAD EAST` alone
has multiple distinct "IND" rows with different individual vote shares.
The first version of `validate_population` compared the *combined* sampled
share (correctly aggregated, since sampling draws by row index) against
*one* duplicate row's individual real share (1.80% sampled vs. 0.54% real,
failing a 0.5-point tolerance) instead of the real shares aggregated the
same way. Fixed by summing real shares per unique label before comparing;
`test_duplicate_party_labels_aggregate_correctly` regression-tests it.
Recorded because this is exactly the kind of silent-until-checked error the
validation ladder exists to catch, and it did.

**Scope, stated plainly.** Ten constituencies, one real cross-section
(2019), Level 0 only. No campaign agents, no persuasion dynamics, no
social network layer, no LLM calls, and -- unlike the AMR extension -- no
regret-harm claim of any kind yet. This section documents infrastructure
being validated, not a result.

## 28. Election track, Phase 1 (theory): persuasion commons is a free consequence of the existing engine, not a new proof

Where §27 built the real-data population layer, this is the abstract,
no-India theorem step the plan called for first ("Phase 1: abstract
election environment... prove the theorem holds," see the domain-extension
discussion) -- built second here because the population layer turned out to
be the faster concrete step, not because the order was reversed in
principle. The two are deliberately not connected yet (see below).

**The isomorphism, not a new mechanism.** `derive_campaign_parameters()`
(`attrition/engines.py`) reuses `LotkaVolterraTumour` completely unchanged
-- the same two-compartment competitive-release engine already relabelled
once for antibiotics (`derive_antibiotic_parameters`). A persuadable/
undecided voter segment plays the drug-sensitive compartment's role; a
saturated/unreachable segment (message fatigue or backfire, not persuasion
to the contacting campaign's side) plays the resistant compartment's role.
Once a segment saturates, no campaign -- not only the one that caused it --
can draw persuadable voters from it again. That is the same shared,
irreversible externality structure as antibiotic resistance, stated as an
isomorphism claim in the docstring and checked, not just asserted:
`test_campaign_persuasion_reproduces_zero_regret_ruin` runs it through the
same `compare`/`Greedy`/`ECI` machinery every other domain in this repo
uses.

**Result.** Registered as `campaign-persuasion` in `SCENARIOS`. Turn-taking,
one campaign: greedy is exactly zero-regret and clears 27% less real
persuasion value than ECI (`examples/11_campaign_persuasion.py`). Multi-
agent, simultaneous: the `lambda_weighted` rule built for the AMR extension
(FINDINGS.md#24) applies here with **zero new code** -- it was built
against the generic simultaneous-action machinery, not anything
AMR-specific, and `test_campaign_persuasion_lambda_weighted_gap_shrinks_
under_decentralisation` confirms the same shrinking-but-not-closing gap
pattern transfers.

**Stated plainly, not glossed over:** with default parameters, the
simultaneous-mode numbers here are numerically identical to exp62's
antibiotic ones (planner=12.899, decentralised eci=12.092, ...). That is
not a second independent confirmation of anything -- `derive_campaign_
parameters()` reuses the same default `engine_kwargs` unchanged, so the
underlying simulation IS the antibiotic one, only relabelled. The
isomorphism claim is that the *mechanism* transfers (a two-compartment
competitive-release engine describes both domains); it is not a claim that
campaigns and antibiotics produce independently-arrived-at matching
numbers, and the identical figures should not be read as more than what
they are: the same math wearing different labels until the parameters are
actually varied to reflect real campaign dynamics.

**Why this is not yet connected to the real Phase-1 population (§27).**
`derive_campaign_parameters()`'s `(v, p, e)` are not derived from anything
in `attrition/data/tcpd_lokdhaba_ge.json` -- they come from the same
synthetic engine as every other mechanistic domain in this repo. Per the
agreed build order (abstract theorem first, synthetic heterogeneous second,
India case study third), that connection is Phase 2/3 work: deriving
persuadability, contact-response, and saturation parameters from real
constituency data (turnout, margin, ENOP) instead of the engine's default
dynamics, the same step `docs/amr-stewardship-extension.md#5` took for
`rho` via the real `leak` parameter. Not attempted in this pass.

## 29. Election track, Phase 2: connecting real margins to the persuasion theorem

The step §28 explicitly left undone: anchoring one real quantity from the
Phase-1 population (FINDINGS.md#27) into the abstract persuasion-commons
engine, the same scope discipline `docs/amr-stewardship-extension.md#5`
used for AMR's `rho` -- anchor what's honestly derivable, state plainly
what still isn't, rather than either skipping the connection or overclaiming
a full recalibration.

**The anchor.** `attrition/election/persuasion.py`: a constituency's real
**margin** (winner share minus runner-up share, both real 2019 vote shares
from `attrition.election.lokdhaba`) sets the campaign-persuasion engine's
initial persuadable-fraction parameter, `S0 = 1 - margin/100` (clipped away
from the 0/1 boundary the engine is degenerate at). Reasoning stated as a
proxy claim, not a measured one: a close contest plausibly means more of
the electorate was still genuinely contestable; a landslide plausibly means
most had already settled. Nothing else about the engine (`R0`, growth-rate
asymmetry, cross-competition, contact-effectiveness ceiling) is anchored to
real data -- still the synthetic defaults, stated as such in the module and
example script, not left implicit.

**Result.** Real margins vary from 9.89 points (Thiruvananthapuram) to
47.01 points (Mumbai North), giving `S0` from 0.53 to 0.90 -- genuine
variation, checked (`test_persuadable_fraction_varies_meaningfully_across_
real_constituencies`), not assumed. Across all ten real constituencies with
that real variation actually driving the engine, greedy stays exactly
zero-regret and loses 26.3-29.6% of real persuasion value to ECI at every
single one (`examples/12_real_persuasion.py`,
`test_zero_regret_ruin_survives_real_margin_variation`). The theorem's
qualitative shape does not depend on which real seat supplies `S0` -- a
real, if partial, robustness result in the same spirit as the AMR
extension's `rho` anchor, for the same reason: it is evidence the pattern
isn't an artifact of one convenient default, not evidence the model
predicts anything about real campaigns.

188/188 tests passing (was 185).

## 30. Election track: the identification-cost result, closed out

Promised in this project's very first planning session (FINDINGS.md#1: T1
reframed as "active identification... requires deliberate abstention...
characterise the... regret price paid for it") and again in the election
plan's persuasion-commons sketch ("how much a campaign must give up... to
even know whether it's helping or harming, as a regret price"). Not a new
theorem: Theorem 3 (`THEORY.md`, "estimation impossibility under
consumption") already proves the externality vector `e` cannot be estimated
to arbitrary precision at any horizon, because the information budget is
capped by `E[N] = sum 1/p_i`, not by how long the process runs. This closes
the promise by evaluating that closed-form floor at the real, heterogeneous
`p` already derived for every Phase-1 constituency
(`attrition/election/identification.py`), the general form rather than
`THEORY.md`'s homogeneous-`p` simplification.

**Two findings, not one, both checked across all ten real constituencies,
not asserted from one:**

1. Only 2 of the 5 contact-intensity levels ever generate a saturation
   event within the derivation horizon, at every single constituency
   (`test_identification_floor_flags_unidentifiable_intensity_levels`).
   For the other 3, `e_a` is not merely hard to estimate -- it is not
   identifiable at any horizon, because the identifying event (a segment
   actually saturating) never happens. Reported explicitly rather than
   averaged away.

2. Where identification is possible, the floor is real, strictly positive
   (`test_identification_floor_is_strictly_positive_where_defined`), and
   ranges from ~0.007 (Thiruvananthapuram, Bangalore Rural, Kolkata
   Dakshin -- the closest real contests) to ~0.033 (Mumbai North, the
   widest real landslide). More-contestable seats give more identifying
   events and a lower floor -- an interpretable pattern in the real margin
   data, not noise.

**Practical reading, stated once, not oversold:** a campaign cannot
distinguish "my contact caused this segment to shift" from "it was shifting
anyway" without a deliberate holdout. That is the same logic this project
opened with, applied 30 sections later to a domain it was never proven in --
closing the loop, not a new claim.

192/192 tests passing (was 188).

---

## 31. Election track: state of open tasks, as of this session

Also closed this session, alongside identification (§30): campaigns as
genuine learning agents, not fixed policies. `attrition.persona.
CAMPAIGN_PERSONAS` plus the existing `Population`/`MockLLMClient` machinery
already proven for AMR (`examples/04_pharma_population.py`) now runs
directly over a real, margin-anchored constituency
(`examples/13_campaign_population.py`) -- zero new mechanism, only new
wiring, the same relationship every other transplant in this project has to
what it transplants.

**Genuinely still open, stated plainly rather than left implicit:**

- **Level 1/2 agents** (behavioural, then LLM cognitive, per the population-
  engine's three-tier plan) -- Level 0 (statistical) only exists.
- **Voter-to-voter social network / persuasion-propagation layer** --
  `attrition.network.AgentGraph` is now wired into the election track, but
  only campaign-to-campaign (do campaigns herd toward each other's contact
  choices? `examples/15_campaign_peer_influence.py`,
  `test_campaign_peer_visibility_runs_end_to_end`) -- information or
  misinformation propagating voter-to-voter through the synthetic
  population itself is a materially different, larger, still-open piece,
  not attempted. Stated precisely rather than let the campaign-network
  result imply more than it shows.
- **Scale past ten constituencies** -- explicitly out of scope for further
  work in this pass: this is a data-engineering and compute exercise, not a
  research question, and inflating the constituency count without a new
  question to ask of the larger population would not be a result, just a
  bigger number.
- **A real trained-policy checkpoint for campaigns**, the way FINDINGS.md#24.4
  did for AMR -- no equivalent external, independently-trained campaign
  policy exists to transplant; this would require training one from
  scratch, which is a materially different, larger undertaking than
  reproducing someone else's.

These are recorded as open, not implied to be near-term next steps by
default -- the next actual step should be chosen deliberately, not just
whichever of these is cheapest to start.

**Update, same session:** campaign-to-campaign peer visibility closed
(above). On one real run at Ahmedabad East (3 campaigns, complete
visibility graph, conformity=0.3), system value rose from 6.753 (isolated)
to 9.040 (networked) -- herding amplified the more-disciplined personas'
choices this time, not the heaviest-contact arm, but the module docstring
is explicit this is a single-run illustration, not a directional claim; see
`examples/06_peer_influence.py`'s equivalent AMR caveat for why the sign
isn't the point. 193/193 tests passing (was 192).

## 32. Election track: Level 1/2 agents and the voter-to-voter social network

Closes the two items FINDINGS.md#31 named as genuinely open, requiring
real design rather than a transplant of something already proven --
designed and built this session, not deferred again.

**Level 1 (`attrition/election/agents.py`, `BehaviouralVoter`).** No LLM,
the tier meant to scale. Each voter accrues persuasion pressure toward a
contacting campaign using the identical rho-decay update rule already
proven for `DegradingPool` (FINDINGS.md#25), reused for the same reason
`derive_campaign_parameters` reuses `LotkaVolterraTumour`: it is the
correct mechanism relabelled, not invented fresh for this tier. Switch
probability is `pressure * (1 - switch_resistance)`, with
`switch_resistance` drawn per voter from a Beta distribution centred on
`1 - persuadable_fraction_from_margin(constituency)` -- so Level 1 inherits
the same real margin anchor the persuasion theorem does
(FINDINGS.md#29), not a second, disconnected one. Checked, not assumed:
Thiruvananthapuram (persuadable_fraction=0.901) switches faster than Mumbai
North (0.530) under an identical contact sequence
(`test_level1_persuadable_fraction_drives_switching_speed`).

**Level 2 (`attrition/election/agents.py`, `VOTER_PERSONAS` +
`simulate_level2_contact`).** LLM-driven, the tier meant to be deep, not
wide. Reuses `LLMPolicy`/`State`/`MockLLMClient` completely unmodified --
the voter's stay/switch decision is framed as a genuine two-arm choice, so
`MockLLMClient` (which parses `LLMPolicy`'s specific prompt shape) works
with no API key required, the same default every other LLM-driven piece of
this repo has. `v_hat`, `p`, `e` are given an explicit, stated
reinterpretation for what a voter's decision means (stay's baseline
utility, switch's value scaling with contact intensity, `e` standing in
for a loyalist persona's cost of over-committing) -- not left implicit,
since bandit-domain vocabulary doesn't literally describe a voter's
psychology. On one seeded run, `voter-swing` switches at round 6,
`voter-loyalist` at round 7, `voter-skeptical` at round 8 -- personas
genuinely diverge, which is the actual justification for paying for a
second tier at all (`test_level2_personas_diverge_by_disposition`).

**Voter-to-voter social network (`attrition/election/network.py`).**
Reuses `attrition.network.AgentGraph` (already proven for campaign/
prescriber peer visibility) as the graph structure over a few hundred
Level-1 voters -- a demonstration scale, stated as such, not the
population engine's "millions" Level-0 aspiration. Adds one new mechanism:
a voter who switches has probability `peer_conversion_prob` of converting
each graph neighbour, independent of whether that neighbour was directly
contacted -- a distinct causal channel (word-of-mouth) from direct contact,
reported separately rather than folded into one number. On one seeded run
at Ahmedabad East (n=500), direct contact alone reaches 88.2% switched;
with propagation, 98.2% -- a +10.0 point marginal lift, isolated by
re-running the identical voters and contact sequence with propagation
disabled, not inferred by comparing across differently-seeded runs.

**Scope, stated precisely rather than left to be inferred:** this is a
*voter* social network, distinct from the *campaign* network closed last
session (FINDINGS.md#31's update) -- the two are separate causal channels
in this model (campaigns herding toward each other's contact choices;
voters propagating switches to their neighbours) and are not combined into
one simulation in this pass.

197/197 tests passing (was 193).

---

## 33. Election track: remaining open items, narrowed

Of FINDINGS.md#31's four open items, two are now closed (Level 1/2 agents,
voter-to-voter network -- above) and two remain, unchanged, deliberately
not attempted this session for the reasons already on record there:

- **Scale past ten constituencies** -- a data-engineering and compute
  exercise, not a research question, without a new question attached.
- **A real trained-policy checkpoint for campaigns** -- no external,
  independently-trained campaign policy exists to transplant the way
  FINDINGS.md#24.4 did for AMR; training one from scratch is a materially
  different, larger undertaking.

Nothing about closing the first two changes the honest status of these
last two -- recorded again rather than left to go stale by omission.

## 34. AMR extension: the last audit item, closed by reframing rather than forcing

FINDINGS.md#26 left one item explicitly open: "no single run has all five
policies (private/greedy, ECI, lambda-weighted, the trained RL baseline,
planner) compared side by side in one environment." Closed this session --
by showing that literal framing asks for something that doesn't exist, and
building the thing that actually answers the underlying question instead
of forcing a misleading merge.

**Why a literal five-in-one-environment table isn't available.** There is
no exact planner for `abx_amr_simulator`'s continuous, multi-patient
dynamics -- the same reason `DegradingPool` has none (FINDINGS.md#25's
state is continuous, not a finite alive/dead subset the DP construction
enumerates). Building all five onto one numeric scale would require either
inventing an unvalidated approximate planner for their environment, or
translating their trained policy's behaviour into this repo's discrete-arm
abstraction through some adapter -- both would manufacture false precision,
not report it.

**What's honestly comparable instead** (`experiments/
exp64_cross_environment_comparison.py`): the cost of decentralisation
relative to the best available reference *within* each environment,
computed separately, never merged into one reward scale.

| environment | reference | decentralised | cost |
|---|---:|---:|---:|
| this repo's engine (exact planner) | 12.899 | eci: 12.080 | 6.4% |
| this repo's engine (exact planner) | 12.899 | greedy: 8.003 | 38.0% |
| `abx_amr_simulator` (no exact planner; siloed is the closest proxy) | -1109.72 | shared: -1151.15 | 3.7% |

A real sign bug was caught building this (the "cost" formula's sign was
backwards for the negative-reward convention, giving -3.7% instead of
+3.7%) -- fixed and regression-tested
(`test_cross_environment_decentralisation_cost_signs_are_positive`), not
silently corrected.

**The actual finding:** both floors are real and strictly positive,
independently, from the same mechanism (Theorem 6). The magnitudes are
deliberately not read as "one environment is worse" -- the reward scales,
action spaces, and reference-point meanings differ (our planner is a
proven optimum; their siloed baseline is only the closest available proxy
for one). What FINDINGS.md#26 actually needed established -- that the
floor isn't an artefact of one convenient environment -- is established.

202/202 tests passing (was 201).

## 35. The Election Layer: does simulated persuasion flip a real seat?

Closes validation ladder level 4 ("intervention validation" -- does
changing campaign/information conditions produce plausible directional
effects?), the level of the plan's six-level ladder not yet exercised.
Nothing built before this session connected Level 1/2's switched-fraction
numbers to an actual recomputed election result -- `attrition/election/
outcome.py` (`recompute_outcome`, `tipping_point`) does that: a real
constituency's baseline shares, shifted by a stated, explicit reallocation
rule (a switched fraction of the total electorate is added to the
challenger's real baseline; every other party's real share, including the
real winner's, shrinks proportionally), producing a winner, margin, and a
flipped/not-flipped verdict against the real 2019 result.

Building this immediately surfaced two real problems in already-shipped
code, caught by connecting components that had only been tested in
isolation, not by inspection:

**Bug 1 -- an impossible outcome.** Feeding Level 1's own final switched
fraction into the first version of `recompute_outcome` produced a
challenger winning with an exact 100.00% share and 100.00pp margin.
Tracing it down: `switched_fraction` can exceed what's mathematically
available (more than every other party's combined real share), which drove
the reallocation's scale factor negative before clamping, zeroing every
other party out. Fixed by clamping the gain to the available share
(`test_recompute_outcome_clamps_gain_to_available_share`) -- caught and
fixed before being reported as a real number, not shipped and found later.

**Bug 2 -- an unintended ratchet in Level 1 itself.** Even after the clamp,
default demo inputs kept producing 85-98% switched fractions, which traced
to `agents.py`'s original per-round mechanic: `switched` was re-drawn via a
Bernoulli trial on `pressure * (1 - resistance)` every round, with a small
independent reversion chance checked only when that trial *failed*. Once
pressure was high, the main trial rarely failed, so the reversion chance
almost never got evaluated -- an unintended one-way ratchet the module
docstring had explicitly (and, it turned out, incorrectly) claimed didn't
exist by construction. Fixed to deterministic threshold-crossing,
`switched = pressure > switch_resistance`, which reverts cleanly when
pressure decays and removes the stochastic-compounding artefact
(`test_level1_switched_fraction_uses_deterministic_threshold_crossing`).
The same fix applied to `network.py`'s duplicate loop, with peer-converted
voters now tracked as a separate, deliberately sticky set rather than
folded into the same boolean as reversible direct-contact pressure.

**A property that survived both fixes, and is reported as a limitation, not
tuned away further.** Sustained near-maximum contact intensity still drives
accumulated pressure toward the ceiling for nearly the whole population,
regardless of resistance -- mathematically correct given the model (pressure
is a convex combination converging on the sustained input), and directionally
sensible (closer, more-persuadable real seats are more sensitive to any given
campaign, matching real intuition -- `test_tipping_point_correlates_with_real_
margin`), but the achievable magnitudes are well outside real campaign-swing
realism. `examples/17_election_outcome.py`'s default sequence stays well
below the ceiling for exactly this reason and says so in-line; a genuinely
close, high-persuadable-fraction seat (Kolkata Dakshin) still saturates
near-total even under that modest sequence. Not fit to any survey or
field-experiment data, same disclaimer class as `derive_campaign_
parameters`'s -- magnitudes here are illustrative of the pipeline's
mechanics, not calibrated forecasts.

**Result, with the fixes and a realistic input:** at Ahmedabad East (real
margin 38.91pp), a modest 5-round campaign (intensity 0.15->0.35) produces
18.5% switched against a 20.1% tipping point -- BJP holds, 49.81% to
runner-up, a believable near-miss. At Mumbai North (real margin 47.01pp,
the widest real landslide in the Phase-1 set), the same campaign produces
8.3% switched against a 24.2% tipping point -- a clear, unthreatened hold.

206/206 tests passing (was 202).
