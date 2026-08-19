# ATTRITION
### Agent Testbed for Task, Resource and Irreversible-Tradeoff Investigation

# Research Programme: Consumable Action Sets at Scale

**Scoping document v1.0** — Navikkumar Modi, August 2026

Companion to *No-Regret is Not No-Harm* (`paper/`) and the `attrition`
library.

---

## 1. The thesis

One paper established that a zero-regret policy can destroy unbounded value when
actions consume the action set. That result is a single-learner, additive-burden,
small-`n` statement. The phenomenon it names is not.

Wherever decisions permanently remove options and the removal harms what remains
— shared API quotas, drug-sensitive cell populations, trial arms drawing on a
common control, material and regulatory envelopes in process development — the
same failure is available, and the standard metric will not report it.

The programme has two strands that feed each other:

- **Theory** supplies the claims, the evaluation criteria, and the falsifiable
  predictions.
- **A simulator** supplies scale, multi-agent dynamics, and the only setting where
  the multi-agent and non-separable results can be tested beyond exact DP.

Neither is worth much alone. Theory without an artifact stays at `n ≤ 8`; an
artifact without theory is a toy with no claims to test.

---

## 2. Name

### The problem with CASCADE

Direct collision. *Cascading bandits* (Kveton, Szepesvári, Wen & Ashkan,
ICML 2015) is an established bandit model with a decade of descendants — DCM
bandits, cascading hybrid bandits, non-stationary cascading bandits,
adversarially-robust cascading bandits. Our own related-work section
distinguishes our model *from* it. Naming the framework CASCADE guarantees
misreading at exactly the point we are establishing distinction.

Also ruled out: **IRIS** (world-model paper; canonical dataset), **NEXUS**
(generic, many claimants), **AETHER** (evocative but semantically empty here).

### Candidates

| name | expansion | reads as | risk |
|---|---|---|---|
| **ATTRITION** | Agent Testbed for Task, Resource and Irreversible-Tradeoff Investigation | arms wear away permanently | low; no ML claimant found |
| **COMMONS** | Consumable Multi-agent Objectives under Non-recoverable Shared Stock | tragedy of the commons — semantically exact | adjacency to DeepMind's Melting Pot commons-harvest scenarios |
| **BURNDOWN** | — | resources burn, never return | agile-methodology collision |
| **KAPPA** | after the invariant `κ = p·e` | owns the discovered object | Kappa architecture (data engineering) |

**Recommendation: ATTRITION**, with COMMONS as second choice if the Melting Pot
adjacency is judged useful rather than derivative — it is arguably the closest
existing artifact and a natural comparison point.

**Before committing:** search arXiv, GitHub, and PyPI for the chosen string. This
is a ten-minute check that prevents a permanent problem.

### On not naming too early

The name should attach to something that runs. Announcing a named platform before
the core exists inverts the order that made the current work credible: we built,
simulation falsified three of our own hypotheses, and the claims that survived
were the ones worth naming.

---

## 3. Theoretical agenda

Seven directions. The important and encouraging fact: **five extend machinery
already built and validated**, so they start from working code and a proven
result rather than a blank page.

### T-A. System-value versus private-value regret — *core, largely in hand*

**Have.** Theorem 4 (proven, exact construction, `gap = m·δE`), and the formal
definition of system value `W(π) = E_π[Σ r_t]` added in the current revision.

**Need.** The separation stated as a theorem in both directions:

1. zero private regret ⟹ system-value loss unbounded *(done)*;
2. near-optimal system value ⟹ **forced** linear private regret against the
   myopic benchmark *(new — the converse, and the more surprising half)*;
3. the two coincide **iff** `κ` is constant or `δ = 0` *(follows from T1)*.

Direction 2 is the sharp new result: it says a good policy is not merely
*permitted* to look bad on the dashboard, it is *required* to. That is a stronger
and more quotable claim than the current paper's.

**Effort:** weeks. The construction is a variant of the hot/safe family.

**Prior art framing:** social-welfare-versus-individual-regret separations exist
in multi-agent learning. Ours is the consumable-action instantiation, where the
separation is driven by benchmark degradation rather than by competition. State it
that way and it is defensible.

### T-B. Multi-agent consumable action sets — *the biggest genuine gap*

Multiple decision-makers drawing on a shared consumable pool. Two orchestrators
burning the same API quota; two sponsors competing for the same trial-eligible
patients; two brand teams consuming shared payer goodwill.

**New objects:**
- equilibrium system value under irreversible externalities;
- ~~price of anarchy under consumption~~ — **investigated and found not to exist**;
  the burden is symmetric, so a `κ`-blind agent behaves identically at any `m`;
- existence/non-existence of pure equilibria when the action set shrinks
  endogenously;
- free-riding: an agent benefits from *others'* restraint, so restraint is
  individually irrational — the tragedy of the commons with a bandit inside.

**Effort:** months. Start with two symmetric agents, exact DP for `n ≤ 6`, which
is directly reachable from the current solver.

**Prior art framing:** price of anarchy is a mature literature; nothing there
treats an action set that decision-makers permanently consume. Ours is a new
instance of an old measure, not a new measure.

### T-C. Learning under irreversible sample budgets, multi-agent — *extends T3*

**Have.** Theorem 3 with the repaired oracle-comparison proof, the `N_exh` sample
budget, and the empirical finding that structure helps only at intermediate
dispersion.

**Need.**
- the floor when *several* agents jointly generate the consumption trajectory and
  observations are private;
- **the value of communication**: how much does sharing destruction observations
  buy, when every observation cost someone an irreplaceable resource?
- algorithms whose system-value regret scales with the irreversible budget
  `N_exh` rather than with `T`;
- the strategic twist: an agent may prefer *not* to share, since its rival's
  ignorance protects the shared pool.

**Effort:** months. The estimator machinery is written.

### T-D. Non-separable, networked, state-dependent burden — *half done*

**Have.** §10 of the paper: T4 robust to all five coupling forms; T1 holds exactly
for additive, multiplicative and networked burden, degrading to 0.10–0.67% under
saturating and concave. The mechanism is identified (marginal burden invariance).

**Need.** The full characterisation:

> greedy is optimal ⟺ the burden is additively separable **and** `κ` is constant

with necessity constructions for each failure mode, plus an approximation bound:
how much system value is lost by applying an additive-`κ` correction in a
non-separable world? The empirical answer is "surprisingly little" — that deserves
a theorem.

**Effort:** weeks. Solver already supports all five forms.

### T-E. Terminal commitment and nested irreversibility — *built, with a proven theorem*

The CMC design-space case from the original spec: a finite budget of irreversible
experiments followed by a **one-shot declaration** that fixes the feasible set
permanently. Implemented (`attrition/commitment.py`), with empirical results in
the paper's appendix, and now a genuine proven result: boundary-only expansion
(never testing a setting non-adjacent to the current claimable envelope) is
exactly optimal, proven via an exchange argument and verified with zero gap
against the fully unrestricted DP optimum across 42 instances. The remaining
open question is the optimal *direction* choice at each step, which a
follow-up investigation showed is not simply "cheaper" and does not appear to
admit a static index.

An optimal-stopping layer atop the bandit, where the stopping action determines the
future action space. No existing bandit formulation carries it. This is the most
distinctive single unclaimed object in the programme, and it generalises well
beyond CMC: any commitment that trades future flexibility for present certainty
has this shape.

**Effort:** the boundary-only theorem is done; a full characterisation of the
optimal direction rule remains open.

### T-F. Mechanism design under irreversible externalities

Design information structures, constraints, or internal transfers so self-interested
agents approximately optimise system value. The `κ`-charge is a Pigouvian tax; the
question is whether it can be implemented when `κ` is (by Theorem 3) not reliably
estimable.

**That framing is the contribution.** Classical Pigouvian mechanism design assumes
the externality can be measured. Theorem 3 says here it cannot. *Mechanism design
under an irreducible externality-estimation floor* is a genuinely new problem, and
it is new precisely because of a result we already proved.

**Effort:** months. Needs care with a large existing literature.

### T-G. Computation and approximation at scale

Approximation guarantees for rollout over the exponential subset-consumption state
space; sample-complexity lower bounds incorporating the irreversible information
bottleneck; when is the ECI-plus-rollout stack provably near-optimal?

**Have.** Rollout reaches 99.5–99.9% of exact optimum empirically and runs at
`n = 40`. **Need.** A bound explaining why.

**Effort:** months.

### Sequencing

```
now ──► T-A(2)  ──────────────────────────► paper 2 (theory)
        T-D full ──┘
                   T-B ──► T-C ──► paper 4 (multi-agent theory)
                   T-E ─────────────────► paper 5 (commitment)
                   T-F, T-G ────────────► later
```

T-A(2) and T-D are the fast wins: both extend proven results, both are weeks not
months, and together they make paper 2 a strictly stronger version of paper 1.

---

## 4. The simulator

### Design principle

The current library is already the kernel: `ConsumableBandit`, seven policies,
exact-DP ground truth, four domains, 22 regression tests. The simulator is that
kernel plus a multi-agent layer, mechanistic engines, and scale — not a rewrite.

**Hybrid grounding is non-negotiable.** LLM agents for strategy and communication;
mechanistic engines for anything where the physics or statistics matter. Pure-LLM
simulation cannot support a claim about system value, because the dynamics would be
whatever the LLM imagined.

**Exact oracles are kept forever.** Every scale-up must remain checkable against
exact DP on a small instance. This is what let today's work falsify three of its own
hypotheses instead of shipping them.

### Stage 1 — Multi-agent core (weeks)

- Shared consumable pool with `m` agents.
- Per-agent private rewards; system value as the sum.
- Exact DP for `n ≤ 6`, `m = 2`; rollout above.
- Baselines: independent greedy, independent TS, independent ECI, central planner.
- **Outcome: no price of anarchy exists** (Theorem 6). The genuine multi-agent effect is free-riding, i.e. charge misspecification.

Directly reachable from the current solver. This alone is a paper.

### Stage 2 — Mechanistic engines (months)

One engine per domain, each replacing a hand-set parameter vector with something
that generates it:

| domain | engine |
|---|---|
| agent tools | quota/rate-limit model with shared-tenancy coupling |
| adaptive therapy | Lotka–Volterra sensitive/resistant dynamics with PK–PD |
| platform trials | group-sequential statistical engine with non-concurrent control |
| design space | process model with spec limits and a filing-envelope calculator |

**This is what turns "reduced-form analogy" into "grounded instantiation"** — the
weakest point in the current paper, fixed by construction rather than by hedging.
The therapy engine matters most: Lotka–Volterra with a resistant compartment is
standard, well-documented, and would let us claim competitive release is
*reproduced* rather than *resembled*.

### Stage 3 — Scale and API (months)

- Gym/PettingZoo-compatible interface.
- Vectorised environments; LLM calls only where semantics matter, never in the
  inner loop.
- Scenario packs: adaptive oncology platform trial, competitive evidence race,
  AMR stewardship, shared-quota agent ecosystem, CMC lock-in.
- Regression suite that **fails if a theorem's numerical claim breaks** — already
  the practice, scaled up.

### Stage 4 — Development–launch coupling: substantively addressed by T-E

This was flagged as the item most at risk of becoming unfalsifiable — a full
multi-stakeholder NPI model (pricing, payers, competitor response, formulary
sequencing) has too many free parameters to constrain against anything real, and
was deliberately deferred rather than built speculatively.

**What turned out to already answer the scoped version of the question.** Terminal
commitment (`commitment.py`, `exp36`) is structurally exactly this coupling: a
finite irreversible experimentation phase (development) followed by a one-shot
declaration (the design-space envelope) that fixes the feasible set for an extended
operating phase (launch). The two failure modes found there —

- **greedy buys yield it cannot claim**: optimising the development phase in
  isolation wastes the budget on settings that cannot be assembled into a coherent
  launch envelope;
- **ambition narrows the envelope it targets**: optimising for launch breadth
  during development can leave a *narrower* commercial envelope than a
  conservative development policy would have produced, at every budget tested;

are development-destroys-launch-value results, derived from a model with three
free parameters (budget, yield curve, failure curve) rather than the dozen-plus a
full NPI model would need. That is what "falsifiable" bought here: a real answer
to the coupling question, at a scope small enough to trust.

**What remains genuinely out of scope, and stays deferred deliberately rather than
by oversight**: competitor response, payer/formulary dynamics, indication
sequencing, and multi-product portfolio effects. These are not "not yet done" —
they are excluded because no version of them was found that stays falsifiable at
this project's evidentiary standard. A future extension should add them one
mechanism at a time, exactly as Stage 2 added engines one domain at a time, each
validated against exact DP before the next is layered on.

---

## 5. Release strategy: one flagship, not a drip

### The case for merging

Incremental extensions do not clear top-tier review. "We extend our prior result
to two agents" is a weak submission whatever the theorem's quality — reviewers
reward scope, completeness, and a contribution that reshapes how a problem is
seen. Six papers, each a step, would produce six mediocre reviews and one
forgettable line of work.

**The merge is therefore correct.** But it succeeds or fails on one structural
choice.

### Merge around a thesis, not a list

A paper presenting seven theorems and a simulator reads as unfocused and gets
rejected for exactly that. The same content organised under one claim reads as
definitive. The claim:

> When actions consume the action set, the standard metric certifies destruction.
> This holds from two arms to multi-agent ecosystems, across every coupling form
> tested, in four domains — and the correction is computable, cheap, and provably
> near-optimal.

Every result becomes evidence for that sentence rather than an item on a list.

### What goes in the flagship

| component | role in the argument | status |
|---|---|---|
| T4 vacuity + exact construction | the phenomenon exists | **proven** |
| **T-A(2) converse separation** | avoiding harm *requires* looking bad | weeks |
| T1 characterisation | exactly when it bites (`κ = p·e`) | proven (sufficiency verified) |
| **T-D full non-separability** | it is not an artefact of additive burden | half done |
| **T-B multi-agent + price of anarchy** | it survives decentralisation, and worsens | months |
| **T-C multi-agent estimation floor** | and cannot be learned away, even jointly | months |
| T3 estimation floor | why specification beats estimation | **proven** |
| ECI + rollout + **T-G guarantees** | the correction, with a bound | empirical → bound |
| **Stage 1–2 simulator** | the artifact others build on | weeks → months |
| four domains, mechanistically grounded | it is not a toy | months |

Bold items are the new work. Everything else exists.

### What was reconsidered, and what still stays out

**T-E terminal commitment** was originally planned to stay out entirely — a
different mathematical object (optimal stopping where the stopping action
fixes the feasible set), with a real risk of diluting the thesis or inviting
"unclear scope." That plan was revised: it is included, but as an appendix
rather than in the main thesis, which captures the original concern (the main
narrative stays focused on the sequential-consumption result) while not
leaving a genuine, now-proven result unused.

**T-F mechanism design** addresses a different community with different
conventions. Its framing — *Pigouvian design under an irreducible
externality-estimation floor* — is excellent and deserves proper treatment, not a
subsection.

**Stage 4 development–launch coupling** is the part most at risk of becoming
unfalsifiable. Including speculative domain modelling in a paper whose strength is
rigour would hand reviewers the attack.

Cutting these is not timidity; it is what makes the remaining argument airtight.

### Priority protection: post paper 1 to arXiv now

Merging does **not** require withholding the current paper, and withholding it
would be a real cost. Post it to arXiv immediately:

- an arXiv preprint is not a publication and precludes no venue;
- it timestamps the core result, which matters in a field this fast;
- the flagship supersedes it as v2, or cites it as the conference version;
- it gives the artifact something citable while the larger work is built.

This is the standard pattern for exactly this situation, and it removes the only
genuine risk of merging — being scooped on the core claim while building the
comprehensive version.

### Venue

| venue | fit | note |
|---|---|---|
| **JMLR** | **strongest** | Long-form, comprehensive treatments with released software are what JMLR is *for*. No page limit, theory plus artifact expected, top-tier standing. Best match for a merged paper of this shape. |
| ICML / NeurIPS main | strong | Page limits force the domains and simulator into an appendix, weakening the artifact half. Viable if the theory leads. |
| NeurIPS Datasets & Benchmarks | good for the artifact | Would split theory from simulator — the opposite of merging. |

**Recommendation: JMLR**, with an ICML/NeurIPS submission of the theory core if a
conference timestamp is wanted alongside.

### Timeline

```
now       post paper 1 to arXiv                         ── priority secured
weeks     T-A(2) converse separation
          T-D full characterisation
          Stage 1 multi-agent core + price of anarchy
months    T-B, T-C multi-agent theory
          Stage 2 mechanistic engines (therapy first)
          T-G approximation guarantees
then      flagship submission
parallel  T-E, T-F as independent papers
```

The first block is weeks of work on existing code and produces the first
price-of-anarchy-under-consumption number — publishable alone if the flagship
slips.

## 6. Risks, and how each is answered

| risk | answer |
|---|---|
| Simulator drifts toward the unfalsifiable | Exact oracles retained at every stage; no claim ships without a small-instance check |
| Mechanistic assumptions manufacture the result | Test each phenomenon under multiple engines, as §10 did for coupling forms |
| Prior art absorbs a claimed contribution | Frame every result as *the consumable-action instantiation* of a known measure, not as virgin territory. Audit before each submission |
| Engineering consumes theory time | Papers 2 and 5 need no simulator at all |
| Scope outruns capacity | Stage 1 and T-A(2)/T-D are independently publishable. Every later stage is optional |

---

## 6b. Programme status (August 2026)

**The strong version first.** Seven results are now proven exactly, not just
verified: both directions of the greedy characterisation, the pure-sequencing
optimum, the estimation floor, the zero-regret vacuity construction, its converse,
the sequential multi-agent equivalence, and ECI's exactness under constant kappa.
That is a genuinely large proven core for a project at this stage, and every one of
them ships with a regression test, so the proof and the code cannot drift apart.
The falsifications below are not a tally of failure — they are the mechanism that
made the seven proofs trustworthy, and they are the reason a reviewer checking this
repository will find claims that hold up rather than claims that were never tested
hard.

## 6b. Programme status (August 2026)

All seven theoretical directions have been attempted. Five produced results; two
produced corrections to their own premises, which is recorded rather than hidden.

| item | status | outcome |
|---|---|---|
| **T-A** system/private separation | **done** | both directions proven; converse is the stronger half |
| **T-B** multi-agent | **done, claim retracted** | no price of anarchy exists: sequential play reduces to a single learner (Thm 6); simultaneity does not change it. Free-riding survives as the genuine effect |
| **T-C** multi-agent estimation | **done, hypothesis falsified** | floor is set by *transitions*, not observations; communication is worthless |
| **T-D** non-separable coupling | **done** | T4 robust to all five forms; T1 requires additive separability |
| **T-E** terminal commitment | **done** | two new failure modes; ambition narrows the envelope it targets |
| **T-F** mechanism design | **done, overclaim corrected** | ordinal *suffices* but is not *learnable*; specification burden is lighter, not absent |
| **T-G** rollout guarantees | **done, conjecture falsified** | closes a constant fraction, not the square; better base policies improve less |

Falsified along the way and kept in the record: √m scaling of multi-agent
estimation error; value of communication; uniform-tax mechanisms; quadratic rollout
scaling; ordinal learnability; ECI exactness in pure sequencing; horizon-scaling of
the T4 gap; the endogenous/exogenous unification claim; the IPIC preservation
heuristic; the effective-horizon index refinement.

**Closed (August 2026).** The ordinal saturation point does have a closed form, and
τ ≈ 0.68 turned out to be an artefact of holding the pool at n = 8. Sweeping pool
size gives `E[τ] ≈ C(n−k,2)/C(n,2)` with **k ≈ 2.5 independent of n** — roughly the
first and last arms destroyed have one side of their transition nearly empty, and
every other arm sits in the interior. So τ → 1 as the pool grows.

The contrast with cardinal recovery is now demonstrated rather than conjectured:
raw cardinal RMSE *worsens* with pool size (0.44 → 1.35 from n=5 to n=30) while
ordinal τ *improves* (0.61 → 0.85). Cardinal error is driven by the number of
coefficients and the burden scale, both growing in n; ordinal error is driven by the
count of positionally compromised arms, which does not.

**Caveat that keeps the practical conclusion intact.** Irreversible-action settings
are typically small-pool — a handful of doses, a few process settings, a modest tool
roster — and that is the regime where both recoveries are weakest. "Specify, don't
estimate" stands, with the refinement that the specification burden is ordinal
rather than cardinal.

**Answered.** `k ≈ 2.5` is **structural, not an artefact of random allocation**. It
holds at 2.20–3.00 across four allocation rules, and every deliberate sequencing
rule tried makes rank recovery *worse* than random allocation, not better. Three
candidate explanations for the degradation were tested and falsified: collinearity
(structured rules are better conditioned), transition count (all rules exhaust the
pool identically), and the count of compromised arms (`k` is flat). The residual
explanation is the *spread* of destruction times, which the binary pair-counting
model does not capture — it predicts 0.700–0.736 for all four rules against an
observed 0.593–0.761.

> **Identification cannot be bought by design.** Random allocation is already the
> best of the rules tested.

A further tension worth noting: the two rules that improve realised value are among
the worst for identification. Playing well and learning why one should play well
pull against each other at the level of allocation design, not just sample size.

**Now open.** A refined model weighting pairs by the precision of both estimates
rather than thresholding them, which would need to reproduce the allocation-rule
dependence the current closed form misses.

---

## 7. What to do first

1. **Post paper 1 to arXiv.** Not a venue submission — a timestamp. Removes the
   only real risk of building a larger work.
2. **T-A(2)** — the converse separation. Weeks, and it is the single strongest
   available result: avoiding harm does not merely permit looking bad on the
   dashboard, it *requires* it.
3. **T-D full characterisation.** Weeks, mostly written.
4. **Stage 1 multi-agent core.** Weeks from the current solver, and it yields the
   first price-of-anarchy-under-consumption measurement.

These four are a few weeks on existing code, and they are simultaneously the
opening chapters of the flagship and a complete fallback paper if the larger work
takes longer than expected. Nothing is wasted either way.

**The one thing that decides the flagship's fate** is not how much goes in but
whether every piece serves the one-sentence thesis. Any result that does not is
better published separately, where it can be the thesis.

---

## Roadmap completion status (August 2026)

Checking the programme in Section 1 against what stands today.

| goal | status |
|---|---|
| Theory that reshapes how a standard metric is read | **done** — Theorem 4/5's two-sided separation: no-regret neither implies nor is implied by preserving value |
| Characterisation of when greedy fails | **done** — proven both directions (Theorem 1), extended to five coupling forms and three mechanistic domains |
| Estimation limits | **done** — proven floor (Theorem 3), sharpened to a transitions-based bound, closed form for the ordinal analogue |
| A corrected policy, not just a diagnosis | **done** — ECI, proven exact under constant kappa, empirically dominant elsewhere, robust to charge misspecification |
| Multi-agent extension | **done** — resolved by Theorem 6 (sequential equivalence) rather than by the equilibrium theory originally planned; a cleaner outcome than expected |
| Open-source artifact, JMLR-appropriate | **done** — `attrition`, 64 tests (one per claim), Gym/PettingZoo APIs, 7 scenarios, 3 mechanistic engines |
| Domain grounding beyond synthetic examples | **done** — 3 engines derived from real dynamics, not hand-set parameters, with an honest negative control (platform trials) |
| Terminal/commitment decisions | **done** — a distinct object (T-E) with two failure modes absent from the sequential setting |
| Development–launch coupling | **done, scoped** — answered by T-E rather than a separate speculative model; full NPI dynamics deliberately excluded |
| Mechanism design under the estimation limit | **done** — ordinal pricing works, cardinal does not; both the sufficiency and the (separate) learnability question resolved |
| JMLR-length manuscript | **done** — 21 pages, three synchronised builds (reading / arXiv / JMLR) from one shared source |

**What remains open, honestly:** a bound on ECI's approximation ratio in closed
form (the option term that has resisted proof since session 1); whether `k`'s
additive constant has its own derivation; and the full multi-stakeholder NPI
extension, excluded by design rather than left incomplete. None of these block
submission — they are stated as open in the paper's limitations section, which is
where genuine open problems belong.

**The programme designed at the start of this phase is substantially complete.**
