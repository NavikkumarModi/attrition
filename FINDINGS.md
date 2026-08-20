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
