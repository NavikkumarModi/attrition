# Research Programme: Consumable Action Sets at Scale

**Scoping document v1.0** — Navikkumar Modi, August 2026

Companion to *No-Regret is Not No-Harm* (`paper/`) and the `consumable-bandits`
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
- **price of anarchy under consumption** — how much system value decentralised
  `κ`-blind agents destroy relative to a central planner;
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

### T-E. Terminal commitment and nested irreversibility — *specified, never built*

The CMC design-space case from the original spec, never developed: a finite budget
of irreversible experiments followed by a **one-shot declaration** that fixes the
feasible set permanently.

An optimal-stopping layer atop the bandit, where the stopping action determines the
future action space. No existing bandit formulation carries it. This is the most
distinctive single unclaimed object in the programme, and it generalises well
beyond CMC: any commitment that trades future flexibility for present certainty
has this shape.

**Effort:** months. New machinery, but well-posed.

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
- **First headline measurement: price of anarchy under consumption.**

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

### Stage 4 — Development–launch coupling (later)

Phase 2/3 decisions set the initial state for launch; anticipated launch
externalities feed back into development policy. This is where the programme
becomes interesting to industry rather than only to reviewers — but it depends on
Stages 1–3 being solid, and it is the part most at risk of becoming unfalsifiable.
Gate it on the mechanistic engines working.

---

## 5. Publication sequence

| # | paper | depends on | venue |
|---|---|---|---|
| 1 | *No-Regret is Not No-Harm* | done | arXiv → ICML/NeurIPS |
| 2 | System-value regret separation + full non-separability characterisation | T-A(2), T-D | ICML/NeurIPS theory |
| 3 | ATTRITION: multi-agent testbed for consumable action sets | Stage 1–2 | NeurIPS D&B / workshop |
| 4 | Price of anarchy and learning under shared consumption | T-B, T-C, Stage 1 | ICML/NeurIPS |
| 5 | Terminal commitment | T-E | ICML/NeurIPS or journal |
| 6 | Domain flagship: development–launch coupling | Stage 4 | domain journal |

Papers 2 and 3 can run in parallel — one is theory on existing machinery, the
other engineering.

---

## 6. Risks, and how each is answered

| risk | answer |
|---|---|
| Simulator drifts toward the unfalsifiable | Exact oracles retained at every stage; no claim ships without a small-instance check |
| Mechanistic assumptions manufacture the result | Test each phenomenon under multiple engines, as §10 did for coupling forms |
| Prior art absorbs a claimed contribution | Frame every result as *the consumable-action instantiation* of a known measure, not as virgin territory. Audit before each submission |
| Engineering consumes theory time | Papers 2 and 5 need no simulator at all |
| Scope outruns capacity | Stage 1 and T-A(2)/T-D are independently publishable. Every later stage is optional |

---

## 7. What to do first

1. **Post paper 1.** Everything else compounds off a citable artifact.
2. **T-A(2)** — the converse separation. Weeks, and it upgrades the headline claim
   from "no-regret permits harm" to "avoiding harm *requires* looking bad."
3. **T-D full characterisation.** Weeks, mostly written.
4. **Stage 1 multi-agent core.** Weeks from the current solver, and it produces the
   first price-of-anarchy-under-consumption number, which is a genuinely new
   quantity.

Those four are a strong second paper and a working multi-agent testbed inside a
few months, built almost entirely on code that already exists and results already
proven. Everything beyond is real but optional, and should be gated on these
landing.
