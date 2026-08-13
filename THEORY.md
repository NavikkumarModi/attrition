# Theory notes — bandits with consumable action sets

Companion to `FINDINGS.md`. Proof sketches for the results the simulations
support. Nothing here is a finished proof; each is a target with a route.

---

## Setup

Arms `1..n`, each with mean reward `v_a`, endogenous death probability `p_a`,
and externality coefficient `e_a ≥ 0`. Pulling `a` at round `t`:

- yields `v_a − B(S_t)` where `B(S) = δ · Σ_{i ∉ S} e_i` is the accumulated burden;
- kills `a` with probability `p_a`, permanently adding `δ e_a` to the burden.

Horizon `T`. Value function over available set `S`:

```
V(S, t) = max_{a ∈ S} [ v_a − B(S) + p_a V(S∖{a}, t+1) + (1−p_a) V(S, t+1) ]
```

Define the **expected marginal externality** of arm `a`:

```
κ_a := p_a · e_a
```

---

## Theorem 1 (greedy-optimality characterisation)

> Greedy — pull `argmax_a v_a` — is optimal **iff** `κ_a` is constant across arms.

**Empirical status.** Confirmed on 5 structural cases and a dispersion sweep
(`exp07`). Case E is decisive: `p_a` and `e_a` each vary widely while `p_a e_a`
is pinned constant, and greedy is exactly optimal (0.000% gap, 0/10 instances
broken). Cases where the product varies break greedy in 10/10 instances.

### Sufficiency — route

Decompose the cost of pulling `a` beyond its immediate reward into

1. **Option loss** — `p_a · [V(S,t+1) − V(S∖{a},t+1)]` attributable to losing
   access to `a`;
2. **Burden creation** — the `δ e_a` penalty applied to every remaining round,
   incurred with probability `p_a`.

For (2), the expected contribution is `p_a · δ e_a · (T−t) = δ κ_a (T−t)`. If
`κ_a ≡ κ`, this term is **identical for every arm** and therefore cannot
influence the argmax. It drops out of the comparison.

For (1), the base model result applies: with geometric (memoryless) lifetimes
and stationary rewards, an arm's total expected yield `v_a / p_a` is independent
of when harvesting begins, so deferring a pull buys no option value. This is the
`e ≡ 0` case, verified exactly in `oracle.py` (12/12 instances) and across
heterogeneous `p`, discounting, rotting, and symmetric commons decay (`exp04`
S1–S4, 0/8 broken each).

With (2) constant and (1) non-distorting, the argmax is determined by `v_a` alone.

### Necessity — route

Suppose `κ_a ≠ κ_b` for some pair. Construct an instance with `v_a` marginally
above `v_b` and `κ_a ≫ κ_b`. Greedy picks `a`; the burden it creates costs
`δ(κ_a − κ_b)(T−t)`, which exceeds `v_a − v_b` for `T` large enough. Hence greedy
is strictly suboptimal. Dispersion sweep supports the stronger quantitative claim:

| std(κ) | 0.000 | 0.048 | 0.090 | 0.184 | 0.330 | 0.427 |
|---|---|---|---|---|---|---|
| gap | 0.000% | 0.523% | 2.481% | 4.467% | 8.210% | 19.076% |

**Conjecture 1a.** The optimality gap is monotone increasing in the dispersion
of `κ`, and zero exactly at zero dispersion.

---

## Theorem 2 (corrected index)

> The index
> ```
> I(a, t) = v_a − δ · p_a · e_a · (T − t) = v_a − δ κ_a (T − t)
> ```
> recovers greedy when `κ` is constant, and captures the majority of the
> optimality gap otherwise.

**Empirical status** (`exp08`, exact DP as reference):

| std(κ) | greedy gap | index gap | fraction of gap captured |
|---|---|---|---|
| 0.177 | 7.820% | 3.746% | 52.1% |
| 0.188 | 6.268% | 1.988% | 68.3% |
| 0.304 | 9.742% | 2.283% | 76.6% |
| 0.498 | 21.889% | 1.953% | **91.1%** |
| 0.572 | 11.278% | 0.897% | **92.0%** |

Two features worth noting:

- **Capture improves as coupling strengthens** (52% → 92%). The index is most
  useful exactly where greedy is worst, which is the desirable direction.
- **A residual gap remains.** The `(T−t)` factor treats the burden as applying
  to a fixed number of remaining rounds, ignoring that the arm population is
  itself shrinking and that the arm may die exogenously before the burden is
  fully paid. A refined index should replace `(T−t)` with the expected number of
  *effective* remaining rounds. **This is the most promising open technical
  question.**

---

## What still needs doing

1. **Prove Theorem 1 properly.** The interchange argument for necessity is
   straightforward; sufficiency needs the option-loss term handled rigorously
   rather than by appeal to the `e ≡ 0` simulations.
2. **Tighten the index.** Replace `(T−t)` with expected effective horizon; test
   whether the residual gap closes.
3. **Learning version.** Everything above assumes `v_a`, `p_a`, `e_a` known. The
   bandit problem is estimating `κ_a` online — and `κ_a` is a *product* of two
   quantities, one of which (`p_a`) suffers the identification problem from
   session 1. This is where the three threads converge:
   - Theorem 1 says `κ_a` is what matters;
   - session 1 says `p_a` is not identifiable without deliberate abstention;
   - therefore **the quantity that determines optimal play is precisely the one
     that cannot be estimated without paying an exploration cost.**

   That is a genuinely tight story and the strongest framing the project has had.
4. **Regret vs value.** Independently established (`exp03`): regret improved 2.9×
   while attainable value fell 14.6×. Under consumable action sets, regret against
   the degrading optimum is not a safety property. Needs a replacement objective.

---

## Theorem 3 (estimation impossibility under consumption)

> The externality vector `e` cannot be estimated to arbitrary precision, no
> matter how long the horizon. The error floor is set by the consumption rate.

### Statement

Consider the model with homogeneous death probability `p`, `n` arms, and
observation noise `σ`. Let `N` be the episode length. Then for any estimator `ê`:

```
SE(δ ê_i)  ≥  σ · sqrt( 1/n_i^before + 1/n_i^after )  ≥  2σ / sqrt(N)
```

and since the episode terminates when the pool empties,

```
E[N] = Σ_i 1/p_i  →  n/p     (homogeneous p)
```

which is **independent of `T`** once `T > n/p`. Therefore

```
SE(δ ê_i)  ≥  2σ · sqrt( p / n )
```

**and this floor cannot be lowered by extending the horizon.**

### Proof route

1. *Information content.* All information about `e_i` enters through the level
   shift `δ e_i` in the reward process at the moment arm `i` dies. Conditional on
   the death time, the MLE is the difference of pre- and post-death sample means,
   with variance `σ²(1/n^before + 1/n^after)`. By AM–HM, `1/n^b + 1/n^a ≥ 4/N`.
2. *Sample budget.* Arm `i` survives `Geometric(p_i)` pulls, so the total number
   of pulls before exhaustion is `Σ_i Geom(p_i)` with mean `Σ_i 1/p_i`. Once `T`
   exceeds this, the horizon is not binding — the pool is.
3. *Combine.* Substituting `N ≈ n/p` gives the floor. The `√p` dependence is the
   substantive content: **error grows as the square root of the consumption rate.**

### Empirical confirmation (`exp12`)

Sweeping `T` at fixed `p`, 60 seeds, `n = 8`:

| p | n/p | T = 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|---|
| **0.9** | 8.9 | rounds 8.8 | 8.8 | 8.8 | 8.8 | 8.8 |
| | | RMSE 0.2329 | 0.2329 | 0.2329 | 0.2329 | 0.2329 |
| **0.5** | 16.0 | rounds 15.3 | 15.7 | 15.7 | 15.7 | 15.7 |
| | | RMSE 0.2396 | 0.2400 | 0.2400 | 0.2400 | 0.2400 |
| **0.1** | 80.0 | rounds 20.0 | 40.0 | 71.0 | 82.3 | 83.2 |
| | | RMSE 0.2048 | 0.1100 | 0.0882 | 0.0923 | 0.0932 |

Three predictions, three confirmations:

- **Horizon independence at high `p`.** At `p = 0.9`, a 16× increase in `T`
  changes RMSE by *zero* — 0.2329 at every horizon. The pool caps the data at
  8.8 rounds and no amount of time adds any.
- **Saturation at `N ≈ n/p`.** At `p = 0.1` the round count climbs 20 → 40 → 71
  → 82.3 → 83.2, flattening exactly at the predicted 80. RMSE improves only
  while the horizon binds, then plateaus.
- **`√p` scaling.** Predicted RMSE ratio between `p = 0.9` and `p = 0.1` is
  `√9 = 3.0`; observed is `0.2329 / 0.0882 = 2.64`. The absolute floor is loose
  by a constant (~7×, expected since 2n parameters are fitted jointly), but the
  rate is right.

### Why this is an impossibility and not a sample-complexity result

Standard estimation bounds say *collect more data*. Here the data-generating
process is the consumption itself: every observation of arm `i`'s externality
requires destroying arm `i`, and the supply of arms is what is being estimated.
The budget is capped by the structure, not the schedule.

### The full tension

Combining with Theorem 1, where `κ_a = p_a e_a` governs optimal play:

| as `p` rises | consequence |
|---|---|
| `κ = p·e` grows | the externality matters **more** |
| oracle-vs-blind gap grows (up to 31.8%, `exp10`) | the cost of ignoring it grows |
| `N ≈ n/p` shrinks | the data available **shrinks** |
| `SE ≥ 2σ√(p/n)` | the estimate degrades as `√p` |

> **The parameter that determines optimal behaviour becomes less estimable
> exactly as it becomes more important.**

**Corollary for deployed agents.** An agent taking irreversible actions cannot
learn their true cost from experience. The observation and the damage are the
same event. Externality coefficients must be *specified* — from domain knowledge,
regulation, or simulation — because they cannot be *discovered* by the agent that
needs them.

---

## Theorem 1 sufficiency — reduced to a single lemma

### The reduction

In the Bellman comparison, `B(S)` and `V(S,t+1)` are independent of the chosen
arm, so maximising the Bellman expression is equivalent to maximising

```
v_a  −  p_a · D_a(S,t),        D_a := V(S,t+1) − V(S∖{a},t+1)
```

Decompose the marginal value of holding arm `a`:

```
D_a  =  δ e_a L(S,t)  +  O_a(S,t)
        \__ burden __/   \_ option _/
```

The burden component contributes `p_a δ e_a L = δ κ_a L`, **constant across arms
exactly when `κ` is constant.** That is the easy half, and it is where the
`κ = p·e` invariant comes from. Everything therefore reduces to:

> **Lemma 1a.** `argmax_a [ v_a − p_a O_a(S,t) ] = argmax_a v_a`
> — the option correction never reorders the arms.

### Status: empirically confirmed, closed form falsified

Exhaustive check over every reachable `(S,t)` in 200 random instances
(`exp13`): **24,946 states, 0 violations.**

Two structural facts that constrain the proof:

1. **`p_a O_a` is not constant.** Spread across arms within a single state reaches
   1.34. So the burden argument does not transfer — the proof cannot be "this term
   is also constant."
2. **A closed form is falsified.** The natural conjecture from the two-arm case —
   that removing `a` forces you onto the runner-up, losing `(v_a − v_next)` per
   round for `1/p_a` expected pulls, giving `p_a O_a = (v_a − v_next)^+` — fails:
   mean absolute error 0.25, max 1.36. The correction is not expressible this way.
3. **Greedy is only *weakly* optimal.** Minimum slack between the corrected best
   and corrected runner-up, over all states tested, is **exactly 0.000000**. Ties
   occur — as expected in the fully-consumed regime (`p = 1`, `T ≥ n`), where every
   arm is spent regardless of order and all orderings are equivalent.

### Remaining proof obligation

Since no explicit formula for `O_a` is available, Lemma 1a needs an
**exchange/coupling argument** rather than direct computation:

> Take an optimal policy that pulls `b` at `(S,t)` while `a` with `v_a > v_b` is
> available. Couple the death randomness and swap the two pulls. Memorylessness
> makes an arm's remaining yield independent of when harvesting starts, so the
> swap does not change the continuation distribution; stationarity makes the
> immediate reward difference `v_a − v_b ≥ 0`. Hence the swap is weakly improving,
> and iterating sorts any optimal policy into greedy order.

The weak-inequality form matches the observed exact ties, which is a good sign the
argument is the right shape. Formalising the coupling — particularly when horizon
truncation interacts with the swap — is the outstanding work.

---

## Theorem 4 (vacuity of no-regret under consumption)

> A policy achieving **exactly zero regret** can realise unboundedly large value
> loss — and can drive total value **negative** while the optimal policy remains
> positive.

### Why this is the sharpest available claim

Nearly every deployed bandit and RL system is justified by a sublinear-regret
argument. Theorem 4 says that argument is **vacuous** when actions consume the
action set: the guarantee holds while the outcome is arbitrarily bad.

### Statement

For any `M > 0` there exists an instance of the consumable-arm model on which
the greedy policy attains regret exactly `0` against the best-available-arm
benchmark, while `V* − V_greedy ≥ M`.

### Construction

Greedy always pulls the highest-value available arm, so its instantaneous regret
against `max_{a∈S} R(a,S)` is identically zero **by construction, at every state,
for every parameter setting.** Meanwhile greedy consumes high-`κ` arms whenever
they also carry high `v`, generating burden `δ e_a` charged against every
remaining round. Scaling `δ` scales the loss without touching the regret, which
stays pinned at zero.

### PROOF (exact, no simulation required)

**Construction.** Fix `m ≥ 1`, `E > 0`, `0 < ε < δE`. Take `n = m+1` arms:

| arm | value | externality | count |
|---|---|---|---|
| hot `H` | `1` | `e_H = E` | 1 |
| safe `S_j` | `1 − ε` | `e_S = 0` | `m` |

Set `p_a = 1` for all arms (each pull consumes) and `T = n+1`, so all `n` arms
are pulled exactly once and the order is the only decision.

**Greedy.** `v_H = 1 > 1 − ε = v_{S_j}`, so greedy pulls `H` first. `H` dies,
adding permanent burden `δE`. Each of the `m` subsequent pulls yields
`(1−ε) − δE`. Total:

```
V_greedy = 1 + m(1 − ε − δE)
```

**Optimal.** Defer `H` to last. Each of the `m` safe pulls yields `1 − ε` with
zero burden; the final pull of `H` yields `1` (its own death's burden is charged
only to *later* rounds, of which there are none). Total:

```
V_opt = m(1 − ε) + 1
```

**Gap.**

```
V_opt − V_greedy = m(1 − ε) + 1 − 1 − m(1 − ε − δE) = m · δE
```

**Regret.** At every round greedy pulls `argmax_{a∈S} R(a,S)`, which is the
definition of the best-available-arm benchmark. Its instantaneous regret is `0`
at every state, for every parameter setting, by construction.

**Conclusion.** `Regret_greedy = 0` while `V_opt − V_greedy = m·δE`, which is
unbounded in both the pool size `m` and the externality magnitude `δE`. Choosing
`δE > 1` makes `V_greedy < 0` while `V_opt > 0`. ∎

### Exact numerical verification (`exp21`)

The closed form `gap = m·δE` matches exact DP to machine precision at **15/15**
parameter settings spanning `m ∈ {1,2,4,8,12}` and `δE ∈ {0.5, 1.0, 2.0}`:

| m | δE = 0.5 | δE = 1.0 | δE = 2.0 |
|---|---|---|---|
| 1 | 0.5000 | 1.0000 | 2.0000 |
| 4 | 2.0000 | 4.0000 | 8.0000 |
| 12 | 6.0000 | 12.0000 | **24.0000** |

At `m = 12`, `δE = 2.0`: `V_opt = 12.88`, `V_greedy = −11.12`. The zero-regret
policy destroys 24 units of value below the optimum and finishes deeply negative.

### Scaling correction (recorded, since an earlier claim was wrong)

An initial version of this construction claimed the gap grows linearly in the
**horizon** `T`. That is **false** and exact DP falsified it: with `p = 1` the
pool exhausts after `n` pulls, so `T` beyond `n` is irrelevant and the gap is
flat in `T`. Under `p < 1` the gap grows with `T` only until `T ≈ n/p`, then
saturates — exactly the sample-budget ceiling of **Theorem 3**, appearing
independently in a value calculation rather than an estimation one.

The correct statement is that the gap is **linear in the pool size `m`**, which
is the right quantity: greedy's error is paid once per subsequent pull, and the
number of subsequent pulls is what the pool supplies.

### Empirical confirmation (`exp16`, n=6, T=10, 12 instances each)

| δ | spread | V* | V_greedy | **V\* − V_greedy** | greedy regret |
|---|---|---|---|---|---|
| 0.02 | 0.5 | 7.581 | 7.515 | 0.066 | **0.0000** |
| 0.05 | 0.8 | 6.902 | 6.532 | 0.370 | **0.0000** |
| 0.10 | 1.0 | 6.717 | 5.810 | 0.907 | **0.0000** |
| 0.20 | 1.5 | 5.436 | 2.747 | 2.689 | **0.0000** |
| 0.35 | 2.0 | 3.570 | **−1.559** | 5.129 | **0.0000** |
| 0.50 | 2.5 | 0.561 | **−8.699** | 9.260 | **0.0000** |

The regret column is zero at every row. The value gap grows monotonically and
without apparent bound. At `δ ≥ 0.35` the zero-regret policy produces **negative
total value** while the optimum stays positive — it is not merely underperforming,
it is actively destroying value, and its own metric reports flawless performance
throughout.

### The claim in one line

> **No-regret is not no-harm.**

An agent can be provably optimal by the standard metric while destroying the
environment it operates in — and by Theorem 3 it cannot learn otherwise, because
the evidence required to price the harm is destroyed by the act of causing it.

### Candidate names for the phenomenon

Needed if this is to travel as a concept rather than a bound. Current shortlist:

- **zero-regret ruin** — punchy, names the failure directly
- **consumption blindness** — names the mechanism
- **self-erasing evidence** — names the Theorem 3 half
- **the no-regret trap** — names the practitioner's error

`zero-regret ruin` is the current preference: it is specific, memorable, and
false-friend-free (it cannot be confused with existing terms).

### Relation to Theorems 1 and 3

The three compose into a single argument:

1. **T1** — optimal play is governed by `κ_a = p_a e_a`.
2. **T3** — `κ_a` becomes less estimable exactly as it grows in importance;
   error floor `2σ√(p/n)` is independent of horizon.
3. **T4** — the standard guarantee (no-regret) is insensitive to all of this and
   remains satisfied while value collapses.

Together: *the quantity that matters cannot be learned, and the metric everyone
uses will not tell you that anything is wrong.*

---

## Lemma 1a — reduced to a verified exchange inequality

The lemma is equivalent to a single concrete inequality. With
`D_x := V(S,t+1) − V(S∖{x},t+1)`, greedy is optimal iff for every pair with
`v_a ≥ v_b`:

```
(EX)      v_a − v_b   ≥   p_a D_a − p_b D_b
```

*The difference in expected option loss between two arms never exceeds their
value gap.*

**Audit** (`exp18`, exhaustive over all reachable `(S,t)` and all ordered pairs,
60 random instances):

| pairs checked | violations | minimum slack | binding (slack = 0) |
|---|---|---|---|
| 44,860 | **0** | **+0.0000000000** | 20,845 (46.5%) |

Binding structure is exactly as the theory predicts:

| regime | fraction of pairs binding |
|---|---|
| `p = 1` (every pull consumes) | **78.0%** |
| `p ∈ [0.05, 0.99]` | 50.3% |

Under full consumption every arm is spent regardless of order, so most orderings
tie — the inequality is tight precisely where it should be.

**The lemma is not an identity.** The corrected value `q_a = v_a − p_a D_a` has
spread up to 1.33 within a single state and is exactly constant in only 0.3% of
states. Greedy strictly wins somewhere, so (EX) has real content rather than
being a restatement.

**Remaining obligation.** (EX) is verified, not proven. The proof route is a
coupling argument: swap the pulls of `a` and `b` under shared death randomness;
memorylessness makes an arm's remaining yield independent of harvest start time,
so the continuation distribution is unchanged, and stationarity makes the
immediate difference `v_a − v_b ≥ 0`. The weak-inequality form matches the 46.5%
exact ties observed, which is evidence the argument has the right shape.

---

## Theorem 3 corollary — structure does not rescue learning

T3 forbids estimating the externality vector `e`. Three routes around it were
tested (`exp19`), and the outcome is more decisive than expected.

**Setup.** `e_a = ⟨ψ, z_a⟩` with known features `z_a ∈ ℝ^k`, `k ≪ n`. Each death
then informs a shared `k`-vector rather than one coordinate, so the T3 floor
should apply per coordinate of `ψ`, not per coordinate of `e`.

| coupling | greedy | per-arm OLS | conservative | features | oracle | features capture |
|---|---|---|---|---|---|---|
| 0.5 | 20.184 | 20.184 | **21.019** | 20.482 | 21.953 | 16.8% |
| 1.0 | 13.224 | 13.224 | **15.531** | 14.002 | 17.609 | 17.7% |
| 2.0 | −0.697 | −0.697 | **3.173** | 1.706 | 7.956 | 27.8% |

Three findings:

1. **Per-arm estimation is exactly worthless** — identical to greedy to three
   decimals at every coupling level. T3 confirmed operationally.
2. **Feature sharing helps but only recovers 17–28%.** It does not close the gap.
3. **The crude conservative default beats the sophisticated learner at every
   level.** Acting on a fixed upper bound — no estimation at all — dominates
   feature-based learning throughout.

### The sharpening test

If the mechanism were sample-sharing, capture should rise monotonically as `k`
falls and each death informs fewer parameters. It does not:

| k | n/k | features capture |
|---|---|---|
| 1 | 20.0 | **4.5%** |
| 2 | 10.0 | 17.5% |
| 3 | 6.7 | 25.8% |
| 5 | 4.0 | **30.3%** |
| 8 | 2.5 | 19.7% |

Capture is *worst* at `k = 1`, where sharing is maximal. The prediction is
falsified, and the reason identifies the real obstacle: with `k = 1` the
externality is nearly uniform across arms, so `κ_a = p_a e_a` has almost no
dispersion — and by **Theorem 1**, zero dispersion means greedy is already
optimal and there is nothing to capture. Capture peaks at intermediate `k`, where
dispersion is large enough to matter and structure is still strong enough to
estimate.

**Corollary.** The obstacle is not sample size but *dispersion*: the externality
must vary across arms for the correction to matter, and variation is exactly what
makes it hard to estimate. Structure trades one against the other and cannot
escape both.

> **Specification beats estimation.** The conservative default — assume a uniform
> upper-bound externality and act on it — outperforms every learned estimator
> tested. This is the actionable form of T3: for irreversible actions, supply the
> cost model; do not expect the agent to discover it.

---

## Theorem 3 — PROVEN (exact derivation, three parts verified)

### Derivation

Condition on the death time of arm `i`, with `n_b` rounds before and `n_a` after.
Taking `v` known (the favourable case — unknown `v` only worsens the bound), the
MLE of `δe_i` is the difference in mean residual reward:

```
Var(δê_i) = σ² (1/n_b + 1/n_a)
```

By AM–HM, `1/n_b + 1/n_a ≥ 4/(n_b + n_a) = 4/N`, with equality **iff** `n_b = n_a`:

```
SE(δê_i)  ≥  2σ / √N                                              [FLOOR]
```

Arm `j` survives `Geometric(p_j)` pulls, so

```
E[N] = Σ_j 1/p_j,   capped at T
```

Combining: `SE ≥ 2σ / √(min(T, Σ_j 1/p_j))`. **For `T > Σ_j 1/p_j` the bound is
horizon-independent.** ∎

### Verification (`exp22`)

**(a) The variance formula.** Monte Carlo vs closed form, 40k trials:

| n_b | n_a | empirical | closed form | rel err |
|---|---|---|---|---|
| 5 | 5 | 0.036016 | 0.036000 | 0.05% |
| 2 | 8 | 0.056270 | 0.056250 | 0.04% |
| 10 | 10 | 0.018015 | 0.018000 | 0.08% |

**(b) The AM–HM floor, N = 20.** Equality attained exactly at the balanced split:

| n_b | n_a | actual SE | floor | ratio |
|---|---|---|---|---|
| 1 | 19 | 0.307794 | 0.134164 | 5.2632 |
| 5 | 15 | 0.154919 | 0.134164 | 1.3333 |
| **10** | **10** | **0.134164** | **0.134164** | **1.0000** |

**(c) The sample budget.** `E[N] = min(T, Σ 1/p_j)`:

| p | n | T | empirical E[N] | predicted |
|---|---|---|---|---|
| 0.9 | 8 | 200 | 8.87 | 8.89 |
| 0.5 | 8 | 200 | 16.01 | 16.00 |
| 0.1 | 8 | 200 | 80.03 | 80.00 |
| 0.1 | 8 | **40** | 39.76 | **40.00** (T binds) |

---

## Lemma 1a — proof route resolved, and it collapses into Theorem 1

### The interchange test

The proposed proof was an interchange argument: swapping a lower-value pull with
a higher-value one is weakly improving. Tested directly (`exp23`) over all
`(S, t, a, b)`:

| regime | violations | min slack |
|---|---|---|
| δ = 0 (no externality) | **0 / 8,568** | −0.000000 |
| δ = 0.05 | 4,610 / 8,568 (53.8%) | −0.106930 |
| δ = 0.15 | 4,610 / 8,568 (53.8%) | −0.320791 |

**Interchange fails when δ > 0.** Initially alarming — but it is the theorem
working, not a counterexample. Greedy is *not* optimal at δ > 0 unless `κ` is
constant (Theorem 1), so a proof technique that established greedy-optimality
there would contradict results already established.

### The decisive test

Holding δ = 0.15 fixed and varying only whether `κ = p·e` is constant:

| regime | violations | min slack |
|---|---|---|
| **κ constant** (T1: greedy optimal) | **0 / 6,842** | −0.000000 |
| κ varies (T1: greedy suboptimal) | 4,118 / 7,332 (56.2%) | −0.302621 |
| e = 0 (κ ≡ 0, constant) | **0 / 6,842** | −0.000000 |

> **Interchange holds exactly when and only when Theorem 1 says greedy is
> optimal.** The lemma and the theorem are the same statement.

### Consequence for the proof

Lemma 1a is not a separate obligation. The correct statement is:

> **Lemma 1a′ (interchange).** If `κ_a = κ` for all arms, then for any `a, b ∈ S`
> with `v_a ≥ v_b`, pulling `a` before `b` is weakly better.

**Proof route (now unobstructed).** Couple the death coins so the pair `{a,b}`
receives the same two draws in either order. The surviving-set distribution after
both pulls is then identical under either order. The burden path differs only in
*which* arm's `δe` is charged first — and under constant `κ`, the expected burden
contributed per pull is `δκ` for both arms, so the expected burden paths coincide.
The only residual difference is which reward arrives first, giving `v_a − v_b ≥ 0`.
Weak inequality matches the observed exact ties (46.5% of pairs, `exp18`). ∎

The `κ`-constant condition is exactly what makes the coupling go through, which
is why the earlier attempt without it failed. Both `e = 0` and `κ` constant are
verified special cases.

---

## Theorem 1 — NECESSITY (proven)

Sufficiency is Lemma 1a′. Necessity follows from the Theorem 4 construction:

> If `κ` is not constant across arms, there exists an instance on which greedy
> is strictly suboptimal.

Take the hot/safe family with `κ_H = E > 0 = κ_S`. The gap is exactly `m·δE > 0`.
Verified exact at 8/8 settings (`exp24`), `m ∈ {1,3,6,10}`, `E ∈ {0.5,1.5}`.

Combined with Lemma 1a′: **greedy is optimal ⟺ `κ_a = p_a e_a` is constant.** ∎

---

## Theorem 2b — exact solution in pure sequencing (new, and it displaces ECI)

**Regime.** `p = 1`, `T = n`: every arm is pulled exactly once, so the only
decision is order.

**Observation.** `Σ_a v_a` is then a *constant* — every arm contributes its value
exactly once regardless of order. Total value is

```
Σ_a v_a  −  δ · Σ_s e_{a_s} · (n − s)
```

An arm placed at position `s` pays its externality `(n − s)` times. Minimising
requires placing large `e` late.

> **Theorem 2b.** In the pure-sequencing regime the optimal policy is: **sort by
> `e` ascending.** Values `v_a` are irrelevant to the optimal order.

**Verification** (`exp24`, brute force over all `n!` orderings): exact at 8/8
instances, `n ∈ {4,5,6}`.

| n | brute force (all n!) | sort-by-e | ECI | greedy |
|---|---|---|---|---|
| 6 | 2.329422 | **2.329422** | 2.100349 | 1.573659 |
| 6 | 3.016190 | **3.016190** | 2.800782 | 1.899984 |
| 6 | 1.615924 | **1.615924** | 1.344538 | 0.862452 |

**ECI is not exactly optimal even here** — the earlier conjecture that it would be
is falsified. Brute force on `n = 5`: best ordering `(3,4,1,0,2)` scores 3.132118,
ECI's `(3,4,0,1,2)` scores 3.110124.

**Why this matters.** It is a clean, fully-solved special case, and it inverts the
intuition: when consumption is certain, the *value* of an arm is irrelevant to
scheduling — only its externality matters. It also bounds what ECI can be: a
heuristic, correct in the ranking it induces only when `v` and `e` happen to agree.

---

## Conjecture 1a — CONFIRMED (monotonicity in dispersion)

An initial sweep suggested non-monotonicity. The sweep was faulty: the *target*
noise scale was varied but the *realised* dispersion of `κ` was not monotone in
it (0.1745, 0.1672, 0.1808, …), so the bins were mislabelled.

Re-run with 240 instances binned by **actual** `std(κ)`:

| std(κ) bin | greedy gap | ECI gap |
|---|---|---|
| 0.079 – 0.227 | 5.88% | 2.15% |
| 0.227 – 0.328 | 9.19% | 2.74% |
| 0.328 – 0.434 | 10.96% | 2.49% |
| 0.439 – 0.542 | 22.41% | 3.12% |
| 0.548 – 0.763 | 24.18% | 1.88% |
| 0.771 – 1.494 | **29.10%** | **1.91%** |

Greedy's gap is monotone increasing across all six bins — **confirmed**.

**Additional finding worth stating separately.** ECI's gap is *flat* at 1.9–3.1%
across the entire dispersion range, while greedy's grows five-fold. ECI's
advantage therefore grows without bound in `std(κ)`: the correction is most
valuable exactly where the uncorrected policy is worst, and its own error does
not degrade.

---

## Final theoretical status

| result | statement | status |
|---|---|---|
| **T1** | greedy optimal ⟺ `κ = p·e` constant | **proven** (sufficiency: Lemma 1a′; necessity: hot/safe construction) |
| **T2** | ECI captures 52–99% of the gap | empirical; flat 1.9–3.1% error across dispersion |
| **T2b** | pure sequencing: sort by `e` ascending | **proven**, exact at 8/8 brute-force checks |
| **T3** | `SE ≥ 2σ/√min(T, Σ1/p_j)`, horizon-independent | **proven** (AM–HM + geometric budget), 3-part verification |
| **T4** | zero regret with gap `m·δE`, unbounded | **proven** (explicit construction), exact at 15/15 |
| **C1a** | gap monotone in `std(κ)` | confirmed over 240 instances, 6 bins |

Falsified along the way and recorded: IPIC preservation heuristic; the
endogenous/exogenous unification claim; the effective-horizon index refinement;
horizon-scaling of the T4 gap; ECI exactness in pure sequencing; per-arm and
feature-based learning of `e`.

---

## Robustness to coupling form — one limitation closed, one sharpened

The additive burden `B(S) = δ Σ_{dead} e_i` was stated as a modelling choice. It is
now tested (deterministic seeds; results reproduce exactly across runs) against four alternatives (`exp26`), with exact DP throughout:

| form | burden |
|---|---|
| additive | `δ Σ e_i` |
| multiplicative | rewards scaled by `Π (1 − δe_i)` |
| saturating | `B_max(1 − e^{−δ Σ e_i})` |
| networked | only graph-neighbours of the dead arm are harmed |
| concave | `δ (Σ e_i)^{0.6}` |

### Theorem 4 is robust to all five — limitation closed

| form | greedy regret | value loss |
|---|---|---|
| additive | 0.0000000000 | 1.3755 |
| multiplicative | 0.0000000000 | 0.6684 |
| saturating | 0.0000000000 | 0.6053 |
| networked | 0.0000000000 | 0.9221 |
| concave | 0.0000000000 | 0.4944 |

Greedy records **exactly zero regret under every coupling form** while losing value
under every one. The headline result does not depend on additivity at all — which
makes sense from the proof, since zero regret follows from greedy's *definition*
against the best-available benchmark, not from the burden's functional form.

### Theorem 1 requires separability — limitation sharpened, not closed

| form | gap, `κ` constant | gap, `κ` varies | T1 holds |
|---|---|---|---|
| additive | **0.000000%** | 12.093% | yes |
| multiplicative | **0.000000%** | 6.644% | yes |
| networked | **0.000000%** | 10.584% | yes |
| saturating | 0.097799% | 12.161% | no |
| concave | 0.674457% | 7.296% | no |

The pattern is exact and the mechanism is identifiable. Measuring the marginal
burden of killing one arm (`e = 1`) at different levels of accumulated damage:

| form | no prior deaths | 3 units dead | 6 units dead | separable |
|---|---|---|---|---|
| additive | 0.15000 | 0.15000 | 0.15000 | **yes** |
| multiplicative | 0.15000 | 0.15000 | 0.15000 | **yes** |
| saturating | 0.16715 | 0.10658 | 0.06796 | no |
| concave | 0.15000 | 0.05463 | 0.04259 | no |

> **Theorem 1 holds exactly when the burden is additively separable across dead
> arms** — that is, when an arm's contribution to the burden does not depend on
> what else has already died.

Saturating and concave forms apply a nonlinear function to the *sum*, so marginal
damage falls as damage accumulates. Then "constant `κ`" no longer makes the burden
term arm-independent, and greedy acquires a residual gap.

**Two things worth noting.** First, this is a sharper statement than the original
theorem, not a weaker one: it identifies the precise structural property the result
needs. Second, the degradation is graceful — the residual gap under non-separable
coupling is `0.10%`–`0.67%`, against the `6.6`–`12.2%` gap greedy incurs when `κ`
genuinely varies. Greedy is *nearly* optimal under constant `κ` even when
separability fails.

---

## Revisions following external review (v0.3)

An external review identified two genuine errors and several overclaims. Both
errors are recorded here rather than silently corrected.

### Error 1 — Theorem 3's estimator argument was wrong

**Objection.** The proof claimed the difference of pre- and post-destruction sample
means is "the MLE" of `δe_i`. It is not. The burden is cumulative, so rounds after
arm `i` dies generally contain the level shifts of several arms. A pre/post
difference does not isolate `e_i`.

**The objection is correct.** The empirical work was unaffected — `exp10`/`exp11`
always used a full least-squares fit with dead-set indicator columns, which handles
multiple deaths properly. Only the *proof* used the naive estimator.

**Repair.** The difference-in-means estimator is the MLE of `δe_i` *conditional on
all other `e_j` being known*. In the real problem they are estimated jointly, and
adding free parameters to a linear model weakly increases the variance of every
retained coefficient (the oracle design is a column submatrix; Schur complement).
Hence

```
Var_joint ≥ Var_oracle = σ²(1/n_b + 1/n_a) ≥ 4σ²/N
```

so the floor survives as a **lower bound**, now correctly derived. Verified in
`exp27`: the chain holds in 14/14 simulated episodes. In instances where several
arms die in quick succession the joint design becomes near-collinear and
`Var_joint` exceeds `Var_oracle` by seven orders of magnitude — the objection's own
concern turns into supporting evidence.

### Error 2 — Theorem 1's quantifiers were too strong

**Objection.** "Greedy is optimal iff `κ` is constant" is too strong read
instance-wise. If all `v_a` are equal, or the horizon is too short for the relevant
arms to be reached, `κ` dispersion need not produce a strict gap.

**Correct.** Restated with asymmetric quantifiers:

- **Sufficiency:** constant `κ` ⟹ greedy optimal at *every* state and *every*
  horizon, for every choice of `{v_a}`.
- **Necessity:** `κ` varies ⟹ *there exist* `{v_a}` and `T` for which greedy is
  strictly suboptimal.

The one-line summary is valid only as universal optimality versus existence of a
counterexample.

### Sufficiency demoted to a proposition

The interchange step is now stated as Proposition (verified, not proved in full
generality) rather than folded into the theorem's proof. A theorem should not
depend on exhaustive verification to fill a gap in its argument.

### Overclaims corrected

| was | now |
|---|---|
| "the externality cannot be learned" / "unlearnable" | "irreducible estimation error"; "cannot be estimated arbitrarily accurately" |
| "ECI is a true index" | "approximate index"; explicitly no optimality guarantee |
| "advantage grows without bound in std(κ)" | "increases throughout the range examined" |
| "reproduces competitive release from first principles" | "captures a qualitative competitive-release mechanism"; explicit non-clinical disclaimer |
| "dominates throughout", "pays everywhere" | "leads at every tested level", "in all four tested domains" |
| `E[N] = Σ 1/p_j` | `N_exh = Σ G_j`, `G_j ~ Geom(p_j)`, under a pool-exhausting policy; `N = min(T, N_exh)` |

### Other changes

- **Related work** expanded with cascading/interacting bandits, combinatorial
  bandits, RL with irreversible actions and safe exploration, and open multi-agent
  bandits. The novelty claim is now stated as a specific four-part conjunction
  rather than a blanket assertion, with an explicit invitation to correction.
- **System value** `W(π) = E_π[Σ r_t]` defined formally, so the private/social
  reading is grounded rather than rhetorical.
- **Table 1's regret** column defined explicitly as per-step instantaneous regret.
- **Sections reordered** to benchmark failure → optimality structure → estimation
  limit → corrections.
- **Shared `body.tex`** so the plain and arXiv builds cannot drift.
- **Repository URL** flagged in bold as a pre-posting requirement rather than left
  as a silent placeholder.

---

## Theorem 1 sufficiency — CLOSED (v0.4)

The coupling argument was the wrong route. A direct argument closes it.

### The proof

Total value decomposes as
```
W(π) = E_π[Σ_t v_{a_t}]  −  E_π[Σ_t B(S_t)]
```

Consider the burden term. A destruction at round `s` imposes `δe_{a_s}` on every
later round, so

```
E_π[Σ_t B(S_t)] = δ · E_π[ Σ_s 1(destruction at s) · e_{a_s} · (N − s) ]
```

Conditioning on the arm chosen at `s`, the destruction indicator has mean `p_{a_s}`,
so round `s` contributes `p_{a_s} e_{a_s} = κ_{a_s}` in expectation. **Under constant
`κ` this is the same number whichever arm the policy picks**, giving

```
E_π[Σ_t B(S_t)] = δκ · E[ N(N−1)/2 ]
```

And `N` is policy-independent: arm `j` absorbs exactly `G_j ~ Geom(p_j)` pulls
before destruction, every pull goes to exactly one arm, so
`N_exh = Σ_j G_j` **regardless of allocation order**. Hence
`N = min(T, N_exh)` has a policy-independent distribution.

The burden term is therefore a constant common to all policies, so maximising `W`
reduces to maximising `E[Σ_t v_{a_t}]` — the same problem with **no externality at
all**. There, geometric destruction is memoryless and rewards stationary, so an
arm's total expected yield `v_a/p_a` is independent of when harvesting starts:
deferring buys no option value and costs the per-round gap. Greedy is optimal
there, hence here, at every state and horizon. ∎

### Why the earlier attempt failed

We were trying to show the *per-arm* burden term drops out of the Bellman argmax,
which left the option term `O_a` to handle separately — and that is where horizon
truncation bit. The correct observation is stronger: constant `κ` makes the
**entire expected burden** policy-invariant, so the option term never needs to be
analysed at all. The identity holds over the realised episode however long it turns
out to be, which is exactly what truncation was obstructing.

### Exact verification (`exp28`)

`E[Σ_t B(S_t)]` computed by DP for five structurally different policies
(max `v`, min `v`, max `p`, min `p`, max `e`):

| regime | expected burden | spread |
|---|---|---|
| constant `κ`, no exhaustion (T=5, n=10) | 0.4000000000 | **3.3e-16** |
| constant `κ`, exhaustion (T=30, n=5) | 1.2873273922 | **2.2e-16** |
| varying `κ` (control) | 0.4444–1.0261 | 7.3e-01 |

Identical to machine precision in **both** regimes — including the exhausting case
where `N` is random — and differing by three orders of magnitude when `κ` varies.

### Consequences

- **Theorem 1 is now fully proven**, both directions. Sufficiency by the argument
  above; necessity by the Theorem 4 construction.
- The interchange inequality (`exp18`, `exp23`) becomes a **corollary** rather than
  a proof obligation. Its 44,860-pair audit is now confirmation, not scaffolding.
- The Proposition introduced in v0.3 is withdrawn — no longer needed.
- The limitations section loses its most serious entry.

---

## v0.5 — converse separation, second benchmark, scale, and multi-agent

### Theorem 5 (converse) — PROVEN

> In the hot/safe family the optimal policy incurs private regret **exactly `m·ε`**.
> Hence for any `M` there are instances where every near-system-optimal policy has
> private regret ≥ `M`.

*Proof.* The optimum defers `H` to the last round. At each of the `m` earlier rounds
`H` is the best available arm (value 1 vs 1−ε) and is declined, costing ε. Summing
gives `m·ε`. For `ε < δE` deferral is necessary for near-optimality. ∎

Exact DP confirms `m·ε` at all 8 settings to machine precision.

**This closes the separation in both directions.** A zero-regret policy may destroy
unbounded value; and a value-preserving policy **must** report regret. The metric
does not merely fail to distinguish them — it orders them backwards.

### Corollary — the dashboard reading tracks the benefit

Private regret `m·ε`, system advantage `m·δE`, ratio `ε/(δE)` — independent of `m`.
Sweeping ε at m=8, δE=1: system gain 8.0000 throughout, private regret
0.08 / 0.40 / 0.80 / 1.60 / 3.20 — ratios exactly ε.

> The worse a value-preserving policy looks, the more it is delivering.

### Second benchmark (addressing the "convenient benchmark" objection)

System-value regret `R_sys(π) = V* − W(π)` reported alongside private regret,
n=6, T=8, exact DP:

| policy | system value | system regret | private regret |
|---|---|---|---|
| optimal | 5.4048 | 0.0000 | 0.9489 |
| ECI | 5.3315 | 0.0734 | 0.6332 |
| greedy | 4.4061 | 0.9987 | **0.0000** |

Opposite orderings. The algorithms are fixed; the benchmark decides the verdict.

### Scale (n=50–100) and stronger baselines

Added three consumption-aware methods: **lagrangian** (BwK-style dual price on
expected consumption, mirror-descent updated), **ratio** (`v_a/(1+p_a e_a)`),
**safe-filter** (burden-quantile exclusion then greedy).

n=100, T=200, std(e)=2.5:

| policy | value | private regret | vs greedy |
|---|---|---|---|
| **ECI** | **111.86** | 55.73 | **+52.3%** |
| ratio | 111.23 | 55.12 | +51.4% |
| safe-filter | 96.82 | 57.46 | +31.8% |
| conservative | 83.54 | 34.67 | +13.7% |
| lagrangian | 74.86 | 1.87 | +1.9% |
| greedy | 73.46 | **0.00** | — |
| Thompson | 71.02 | 9.30 | −3.3% |
| UCB | 69.73 | 53.32 | −5.1% |

**The Lagrangian result is the important one.** It is the closest existing approach
and it barely helps: it prices consumption, but a scalar dual cannot express the
horizon-dependent charge `δκ_a(T−t)`, which must shrink as the episode runs down.
Resource-aware machinery is insufficient — the correction must be time-varying.

**Honest qualification: `ratio` is competitive at this scale**, within 1% of ECI.
Against exact optimum at n=6, though, ECI dominates at every setting
(`exp31`):

| T | std(κ) | greedy | ECI | ratio |
|---|---|---|---|---|
| 4 | 0.320 | 1.97% | **0.26%** | 1.27% |
| 8 | 0.862 | 26.86% | **1.72%** | 11.19% |
| 16 | 0.649 | 36.80% | **0.60%** | 13.41% |
| 30 | 0.858 | 46.57% | **0.14%** | 15.77% |

ECI's margin widens with horizon, exactly as the `(T−t)` term predicts. With many
arms a good low-κ substitute is nearly always available, so fine-grained weighting
matters less — which is why the two converge at large n.

---

## Multi-agent: the price of anarchy does not exist here — RETRACTED AND REPLACED

An earlier version of this section reported a price of anarchy reaching 1.297 under
sequential multi-agent play, and claimed κ-aware agents reduce it to 1.004. **The
measurement was invalid** and is retracted. What replaced it is a theorem.

### Theorem 6 (sequential equivalence) — PROVEN

> On a shared consumable pool, if agents act one after another within a round and
> each observes the state left by its predecessors, then `m` agents over `T` rounds
> is exactly one learner over `m·T` pulls.

**Proof.** The state is the surviving set with its accumulated burden. Agent `i` at
position `j` of round `t` faces exactly the state a single learner faces at pull
index `tm + j`; the immediate reward `v_a − B(S)` is the same function of that
state, and the transition — destruction with probability `p_a` — is identical. The
two processes have the same MDP, so any policy in one induces a policy of equal
value in the other. ∎

Verified at five `(m,T)` combinations with `mT ∈ {6, 8}`: sequential agents and a
single greedy learner agree to within 0.015.

**Consequence.** Decentralised greedy agents behave *identically* to one greedy
learner, so a price of anarchy computed in this model measures nothing about
decentralisation. The 1.297 figure came from comparing against a planner restricted
to **distinct** arms per round while the agents could re-target within the round —
a difference of action spaces, not incentives.

**The tell was already in the data.** At zero κ dispersion, where Theorem 1 forces
decentralisation to be costless, the sweep measured PoA = 1.0230 rather than 1.

### Simultaneous action does not rescue it either

Genuine decentralisation needs agents unable to condition on each other's current
choices. A simultaneous-action model was built (`simultaneous.py`): all agents
choose at once, pulls resolve together, and an arm chosen by several agents faces
several independent destruction trials.

| m | κ dispersion | PoA greedy | PoA ECI | single-agent ratio at horizon mT |
|---|---|---|---|---|
| 2 | 0.000 | **0.9999** | 0.9999 | 1.0000 |
| 2 | 0.635 | 1.1016 | 1.0342 | 1.1916 |
| 3 | 0.000 | **1.0029** | 1.0052 | 1.0000 |
| 3 | 0.635 | 1.1526 | 1.0567 | 1.2772 |

**At zero dispersion PoA is 1.00 even under simultaneity**, and with dispersion it
sits *below* the single-agent gap at the same total pull count. There is no
coordination cost to find.

Two further predictions of mine failed here and are worth recording. I expected
**collisions** to be costly — several agents converging on one arm and destroying it
faster than intended. They are not: concentrating on the best arm is worth more than
the extra destruction hazard costs. And I expected a **spread-out** rule (agents
taking distinct arms) to help. It is far worse — PoA 1.25 to 2.07 — because
spreading means most agents pull inferior arms.

### Why there is no commons problem, and this is the substantive finding

In a classical commons, my consumption harms you and not me. Here the burden
`B(S)` is subtracted from **every** agent's reward including my own, on every
subsequent pull. The externality is symmetric, and a greedy agent that ignores it
entirely behaves the same whether `m = 1` or `m = 10`. There is nothing to
free-ride on when no one is paying in the first place.

> **Consumption externalities on a shared pool do not create a price of anarchy.**
> The cost attributed to decentralisation is the single-agent pricing failure,
> counted once per agent.

### What survives, and it is the genuinely multi-agent part

Free-riding is real, because it changes the *objective* rather than the action
space. An agent that prices the externality but discounts it by its own expected
share — `δκ_a(T−t)/m` instead of the full charge — destroys system value:

### Free-riding

An agent discounting the externality by its own expected share (÷3 for three
agents) loses system value:

| agents | social ECI | selfish ECI | loss |
|---|---|---|---|
| 2 | 6.122 | 5.386 | **12.0%** |
| 3 | 6.061 | 5.225 | **13.8%** |

Standard commons structure, with a bandit inside — and a target for the mechanism
design direction (T-F).

---

## Stage 2 — mechanistic grounding of the adaptive-therapy domain

The reduced-form domain set `(v, p, e)` by hand, which invites the objection that
the parameters were chosen to produce the phenomena. They are now **derived from
Lotka–Volterra competition dynamics** (`engines.py`, `exp33`):

```
dS/dt = r_S S (1 − (S + a_SR R)/K) − kill(dose)·S
dR/dt = r_R R (1 − (R + a_RS S)/K)
```

Sensitive cells grow faster and are killed by the drug; resistant cells grow slower
and are not; both compete for one carrying capacity, so a large sensitive population
suppresses the resistant clone.

**Derivation, nothing hand-set:**
- `v_a` = burden reduction achieved by dose `a` in one treatment period
- `p_a` = probability dose `a` drives the sensitive compartment below a floor,
  estimated by perturbing initial tumour composition
- `e_a` = permanent loss of future control once that compartment is gone

**Derived parameters:**

| dose | v | p | e | κ |
|---|---|---|---|---|
| 0.49 | 0.4013 | 0.000 | 0.4456 | 0.0000 |
| 0.66 | 0.5037 | 0.025 | 0.4456 | 0.0111 |
| 0.83 | 0.5534 | 0.530 | 0.4456 | 0.2362 |
| 1.00 | 0.5698 | 1.000 | 0.4456 | 0.4456 |

`std(κ) = 0.1705`. Note `e` is constant across doses — the damage of losing the
sensitive compartment does not depend on which dose caused it — so this is
structurally **case D** from the taxonomy (`p` varies, `e` constant), which
Theorem 1 predicts breaks greedy.

**Theory survives grounding:**

| δ | V* | MTD (= greedy) | ECI | MTD loss | MTD private regret |
|---|---|---|---|---|---|
| 0.05 | 4.0955 | 3.8843 | 4.0551 | 5.2% | **0.000000** |
| 0.15 | 4.0317 | 3.3318 | 3.9952 | 17.4% | **0.000000** |
| 0.30 | 3.9662 | 2.5032 | 3.9094 | **36.9%** | **0.000000** |

Maximum tolerated dose *is* the greedy policy — it maximises immediate burden
reduction at every step. It records zero private regret and loses up to 36.9% of
system value, on parameters nobody chose.

### Direct dynamics: time to progression

Running the dynamics without any bandit abstraction. Final burden cannot
distinguish protocols — over a long enough horizon the resistant clone reaches
carrying capacity under any policy. What differs is *how long that takes*, which is
the endpoint the adaptive-therapy literature reports.

| protocol | TTP (periods) | resistant fraction | vs MTD |
|---|---|---|---|
| MTD (always max dose) | 26 | 1.000 | — |
| fixed low dose 0.5 | 29 | 0.999 | +12% |
| adaptive, back off S<0.30 | 34 | 0.953 | +31% |
| adaptive, back off S<0.40 | 38 | 0.955 | +46% |
| adaptive, back off S<0.50 | **44** | 0.911 | **+69%** |

Competitive release is **reproduced**, not resembled: deliberately retaining
sensitive cells to suppress the resistant clone extends time to progression by 69%.

### Note on an endpoint that does not work

Cumulative burden reduction and final burden are both useless as endpoints here —
every protocol converges to the same terminal state (resistant fraction 1.000,
burden ≈ 0.998). An earlier version of this experiment compared protocols on final
burden and found no difference, which was a property of the endpoint rather than of
the protocols. Time to progression is the correct measure and is what the clinical
literature uses.

**Status of the claim.** The paper may now say the model reproduces a
competitive-release ordering, rather than resembles one. It remains a two-compartment
reduced model without spatial structure, pharmacokinetics, or regrowth heterogeneity,
and no clinical conclusion should be drawn from it.

---

## Stage 2 complete — three engines, and a refinement of Theorem 1

Two further engines added (`engines.py`):

**`PlatformTrialEngine`** — shared-control trial with staggered entry. Dropping an
arm fragments the control stream into more non-concurrent blocks; every remaining
comparison must then either discard non-concurrent data or adjust for time drift,
inflating variance. Derived: `v` = effect size × information gain per period,
`p` = futility-boundary crossing probability, `e` = precision lost by the remaining
arms, weighted by the exiting arm's tenure.

**`DesignSpaceEngine`** — process development. Derived: `v` = yield at that setting,
`p` = out-of-spec probability (rising near the spec limit), `e` = fraction of the
defensible filing envelope forfeited by a failure there.

### Results across all three engines

| engine | std(κ) | δ | V* | greedy | ECI | greedy loss | greedy regret |
|---|---|---|---|---|---|---|---|
| tumour dynamics | 0.1705 | 0.224 | 3.9911 | 2.9207 | 3.9343 | 26.8% | **0.00000000** |
| tumour dynamics | 0.1705 | 0.898 | 3.7773 | −0.7986 | 3.6946 | **121.1%** | **0.00000000** |
| platform trial | 0.0371 | 0.415 | 33.4683 | 33.4683 | 33.4683 | **0.0%** | 0.00000000 |
| design space | 0.2954 | 0.158 | 6.1489 | 5.0839 | 5.8098 | 17.3% | **0.00000000** |
| design space | 0.2954 | 0.634 | 5.1230 | −0.3533 | 4.9439 | **106.9%** | **0.00000000** |

### The negative result, and what it teaches

**The platform trial shows a 0.0% gap — greedy is optimal there.** This is reported
rather than tuned away, and it sharpens the theory.

| engine | std(κ) | corr(v, κ) | greedy |
|---|---|---|---|
| tumour dynamics | 0.1705 | **+0.614** | fails |
| platform trial | 0.0371 | **−0.938** | optimal |
| design space | 0.2954 | **+0.789** | fails |

> **Refinement.** Dispersion of `κ` is necessary but not sufficient. The dispersion
> must *conflict with the value ordering*. Where `v` and `κ` are negatively
> correlated — good arms are also safe arms — greedy is already choosing correctly
> and there is nothing to correct.

The mechanism in each case is interpretable:

- **Tumour dynamics (+0.61):** higher dose gives more immediate control *and* is
  more likely to exhaust the sensitive compartment. Conflict.
- **Design space (+0.79):** more aggressive settings yield more *and* are more
  likely to go out of spec and truncate the envelope. Conflict.
- **Platform trial (−0.94):** promising arms are precisely the ones that will *not*
  cross a futility boundary, so they never fragment the control stream. The
  externality falls on the arms nobody wanted to keep anyway. Alignment.

This explains a practical asymmetry worth stating: the failure mode is a property of
domains where **ambition and damage travel together**. That is the common case in
consumption settings — pushing harder yields more now and costs more later — but it
is not universal, and the platform trial is a clean counterexample.

**Consequence for the paper.** The domain section should present the platform trial
as the case where the correction is *not* needed, which strengthens rather than
weakens the argument: the theory predicts where the phenomenon appears and where it
does not, and both predictions hold.

---

## Stage 3 — environment API and scenario packs

**Interfaces implemented natively** rather than importing gymnasium/pettingzoo, so
the package carries no RL-framework dependency while matching both signatures:

- `ConsumableBanditEnv` — Gym: `reset() → (obs, info)`,
  `step(a) → (obs, reward, terminated, truncated, info)`
- `MultiAgentConsumableEnv` — PettingZoo parallel: dict-keyed observations,
  rewards, terminations, truncations, infos

**Design decision worth stating.** Observations expose availability, value
estimates and pull counts, but **not** the true `(p, e)`. Theorem 3 establishes
that the externality cannot be reliably estimated from experience; an agent that
read it off the observation vector would be solving a different problem from the
one the theory describes. `reveal_externality=True` lifts this for oracle
experiments.

### Scenario suite

Seven named scenarios, each documenting the phenomenon it should exhibit, with the
test suite asserting that it does:

| scenario | arms | std(κ) | corr(v,κ) | greedy loss | ECI loss | greedy regret |
|---|---|---|---|---|---|---|
| high-dispersion | 9 | 0.6285 | +1.000 | **176.5%** | 0.1% | 0.00000000 |
| design-space | 6 | 0.2954 | +0.789 | 99.3% | 1.1% | 0.00000000 |
| adaptive-therapy | 6 | 0.1705 | +0.614 | 36.9% | 1.4% | 0.00000000 |
| shared-quota | 6 | 0.9948 | +0.315 | 3.7% | 0.3% | 0.00000000 |
| platform-trial | 6 | 0.0371 | −0.938 | **0.0%** | 0.0% | 0.00000000 |
| aligned-control | 6 | 0.6299 | −0.971 | **0.0%** | 0.0% | 0.00000000 |

Plus `shared-quota-competing` (2 agents) for price-of-anarchy work.

**Both alignment controls behave as predicted.** `aligned-control` is synthetic
with `std(κ) = 0.63` — larger dispersion than adaptive-therapy — yet a 0.0% gap,
because `corr(v,κ) = −0.97`. Dispersion alone genuinely is not the criterion.

Note `shared-quota` has the *highest* dispersion (0.9948) but the *smallest* loss
(3.7%), because its correlation is weakest (+0.315). Across the suite the loss
tracks the correlation, not the dispersion — which is the sharper statement of the
condition.

---

## Stage 4 — terminal commitment (T-E), implemented

A distinct object from the rest of the library. Elsewhere the action set shrinks as
a side effect of pulling arms; here there is an explicit **terminal action**: after
a finite budget of irreversible experiments the learner must *declare* an operating
envelope, and that declaration bounds every subsequent action for the whole
operating phase.

**Model** (`commitment.py`). Settings indexed `0..n−1` with a nominal point at the
centre. Yield rises monotonically with intensity; failure probability rises with it
too — the `corr(v, κ) > 0` structure. Each experiment either demonstrates viability
or produces an out-of-spec result that **blocks that setting permanently**. The
declared envelope is the widest contiguous interval of demonstrated settings
containing the nominal point. During operation the process drifts; landing inside
the envelope collects the yield there, landing outside forces reversion to nominal
plus a variation penalty.

Operating value is computed **exactly** via a discretised Gaussian drift PMF rather
than sampled, so evaluation is deterministic.

**Results** (300 seeds per cell):

| budget | policy | operating value | envelope width | failed runs |
|---|---|---|---|---|
| 2 | greedy (best yield first) | 36.908 | **1.00** | 0.80 |
| 2 | edge-first (max width) | 36.908 | **1.00** | 0.67 |
| 2 | **expand-outward** | **40.664 (+10.2%)** | **2.92** | 0.08 |
| 4 | greedy | 41.385 | 3.44 | 1.06 |
| 4 | edge-first | 36.908 (−10.8%) | **1.00** | 1.14 |
| 4 | **expand-outward** | **44.182 (+6.8%)** | **4.72** | 0.26 |
| 6 | greedy | 45.058 | 5.35 | 1.14 |
| 6 | edge-first | 36.908 (−18.1%) | **1.00** | 1.36 |
| 6 | expand-outward | 45.069 | 6.20 | 0.61 |

### Two distinct failure modes, both new

**1. Greedy buys yield it cannot claim.** It spends the budget on the highest-yield
settings wherever they sit. But an envelope is an *interval containing the nominal
point*, so a demonstrated setting far from centre is worthless unless everything
between is also demonstrated. At budget 2, greedy ends with envelope width
**1.00** — it demonstrated high-yield settings and could claim none of them.

**2. Ambition narrows the envelope it was trying to widen.** Edge-first targets the
outermost settings to maximise claimable width. It pays the highest failure
probability per experiment, and each failure blocks that setting *permanently*.
Its envelope width is **1.00 at every budget**, and it gets monotonically worse as
budget grows (−10.8% at 4, −18.1% at 6): more budget means more edge attempts, more
blocked settings, and a worse terminal position. **Spending more to widen the
envelope makes it narrower.**

That second mode has no analogue in the sequential setting and is the distinctive
contribution of the terminal-commitment formulation. It is also the failure mode
practitioners describe: an aggressive characterisation campaign that ends with a
*narrower* filed design space than a conservative one would have produced.

**Expand-outward** — grow the interval one step at a time, taking the cheaper
direction — dominates at small budgets, where the commitment bites hardest. At
budget 6 greedy catches up, because with enough experiments the contiguity
constraint stops binding.

### Why the budget dependence matters

The advantage of the correct policy *decreases* with budget (+10.2% → +6.8% →
+0.0%). This is the opposite of the sequential setting, where the gap grows with
pool size. The reason is structural: terminal commitment is a problem about
**scarcity of evidence at the moment of an irreversible decision**. Given enough
evidence the commitment is no longer made under uncertainty and the problem
dissolves. It is hardest exactly when experiments are expensive — which is the
regime that motivates it.

---

## T-C — multi-agent estimation, and a sharpening of Theorem 3

### The hypothesis was wrong, and the correction is stronger

Expected: with `m` agents sharing a pool, each sees roughly `N_exh/m` observations,
so the per-agent floor should scale as `2σ√(m/N_exh)`. Communication should then
recover the single-learner floor.

**Neither holds** (`exp37`):

| agents | private RMSE | obs seen | shared RMSE | obs seen | gain from sharing |
|---|---|---|---|---|---|
| 1 | 0.3283 | 36.8 | 0.3283 | 36.8 | 0.0% |
| 2 | 0.3423 | 18.7 | 0.3283 | 36.8 | 4.1% |
| 3 | 0.3121 | 12.6 | 0.3283 | 36.8 | −5.2% |
| 4 | 0.3200 | 9.6 | 0.3283 | 36.8 | −2.6% |

Observations fall almost four-fold and the error does not move. Sharing them back
gains nothing.

### Why: the floor is set by transitions, not observations

| T | observations | deaths | RMSE |
|---|---|---|---|
| 40 | 33.1 | 7.35 | 0.3351 |
| 100 | 36.8 | **8.00** | **0.3283** |
| 200 | 36.8 | **8.00** | **0.3283** |
| 400 | 36.8 | **8.00** | **0.3283** |
| 900 | 36.8 | **8.00** | **0.3283** |

RMSE saturates exactly when the death count saturates at `n = 8`, and is identical
to four decimal places thereafter.

> **Sharpened Theorem 3.** The binding constraint is not the number of
> observations but the number of *transitions*. Each arm supplies exactly one
> before/after transition, so the information available about `e` is bounded by
> `n`, the pool size — not by `T`, not by `N_exh`, and not by the number of
> observers.

This is a stronger statement than the horizon-independence result and it subsumes
it. It also explains the earlier feature-sharing finding: structure could not
rescue estimation because the deficit was never a shortage of samples.

**Corollary (communication is worthless here).** Ten agents watching an arm die
still see one death. The information lives in the transition, and the transition is
common property the moment it happens. This is unusual: in most multi-agent
learning problems communication is valuable precisely because observations are
private and additive. Here they are neither.

### Withholding does not pay either

If communication cannot help a rival, can ignorance be weaponised? No:

| agents | both informed | one ignorant | both ignorant |
|---|---|---|---|
| 2 | 6.645 | 6.386 | 6.204 |
| 3 | 6.160 | 5.438 | 5.262 |

System value falls monotonically as agents are kept ignorant. An uninformed rival
does not conserve the pool; it consumes the high-`κ` arms the informed agent was
declining, and the damage is shared. **Withholding accelerates destruction rather
than protecting the commons** — the opposite of the usual information-hoarding
story, and a useful fact for the mechanism-design direction: there is no incentive
problem to solve on the disclosure side.

---

## T-F — mechanism design when the externality cannot be measured

Classical Pigouvian design charges each agent the marginal damage it causes, which
presumes the damage is measurable. Theorem 3 (sharpened) says it is not: the
information about `e` is bounded by the number of transitions, i.e. the pool size,
and no horizon, agent count, or communication changes that.

So: can a mechanism work without measurement?

### The obvious answer fails

A **uniform tax** — one number, no per-arm knowledge — was the natural candidate,
and it worked in the single-agent case (`exp19`, where the conservative policy beat
every learned estimator). In the multi-agent case it does not:

| agents | greedy | uniform tax | rank only | ban worst ⅓ | true κ |
|---|---|---|---|---|---|
| 2 | 1.140 | **1.156** | 1.075 | 1.076 | 1.065 |
| 3 | 1.227 | 1.186 | **1.035** | **1.031** | 0.961 |

At `m = 2` the uniform tax is **worse than no charge at all**. The reason is
structural: a flat charge does not distinguish arms, so it shifts every preference
equally and leaves intact the ordering that caused the problem.

### Ordinal knowledge suffices

Two mechanisms that need no cardinal estimate recover nearly all the benefit:

- **rank-based charge** — price by the arm's rank in `e`, not its value;
- **ban the worst third** — exclude the highest-`e` arms, then act greedily.

Both reach `PoA ≈ 1.03` at three agents, against greedy's `1.227` and true-κ
pricing's `0.961`.

> **The mechanism Theorem 3 rules out is not the one that is needed.** Theorem 3
> bounds how precisely `e` can be *estimated*. It says nothing about recovering the
> *ordering*, which requires far less information — a single well-resolved
> comparison per pair rather than a calibrated magnitude.

This is the productive form of the impossibility result. It does not say that
externality-aware mechanism design is hopeless under consumption; it says that
cardinal mechanisms are unavailable and ordinal ones are not, and that ordinal ones
are nearly as good. For practice the implication is concrete: an operator does not
need to know how damaging each action is, only which actions are more damaging than
which — a far weaker elicitation problem.

**Open.** Whether ordinal recovery has its own floor. The transitions argument
bounds cardinal estimation; a matching analysis for rank recovery would complete
the picture and is the natural next theorem.

---

## Ordinal recovery has its own floor — and an earlier claim needs correcting

### The conjecture, and what actually happens

T-F concluded that "the mechanism ruled out by Theorem 3 is not the one required",
on the grounds that ordinal knowledge suffices and ordinal recovery needs less
information than cardinal. The first half is right. **The second half was wrong,
and the error was a subtle one worth recording**: `make_rank_based` reads the
ordering off the *true* `e`, so exp38 established what information *suffices*, not
what is *recoverable*. It was an oracle mechanism.

Testing recovery directly (`exp39`), with Kendall tau between estimated and true
ordering alongside cardinal RMSE:

**Varying noise** (spacing fixed at 0.25):

| σ | cardinal RMSE | ordinal τ |
|---|---|---|
| 0.05 | 0.1654 | 0.386 |
| 0.30 | 0.3194 | 0.163 |
| 1.00 | 0.9407 | 0.068 |

**Varying spacing** (σ fixed at 0.30):

| spacing | cardinal RMSE | ordinal τ |
|---|---|---|
| 0.05 | 0.3165 | 0.091 |
| 0.50 | 0.3271 | 0.254 |
| 1.00 | 0.3539 | 0.391 |

**Pushed to the limit:**

| σ | spacing | ratio | τ |
|---|---|---|---|
| 0.020 | 1.0 | 50 | 0.569 |
| 0.020 | 2.0 | 100 | 0.631 |
| 0.005 | 2.0 | 400 | 0.626 |
| 0.001 | 4.0 | **4000** | **0.676** |

**Rank recovery saturates near τ ≈ 0.68 and does not approach 1**, even with the
gap-to-noise ratio raised four thousand-fold.

> **Superseded below.** This figure is specific to `n = 8`, which was held fixed
> throughout this experiment. Sweeping pool size shows τ → 1, and the constant is
> `k ≈ 2.5` positionally compromised arms rather than a ceiling on τ. See "The
> ordinal saturation point: a closed form, and a correction".

### Why, and why it is the same mechanism

The saturation has the same cause as the cardinal floor. Each arm supplies exactly
one transition, and arms destroyed very early or very late have almost no data on
one side of it. Their estimates are poor regardless of noise, and a handful of
badly-placed arms caps the achievable rank correlation. Reducing σ cannot help,
because the deficit is positional, not statistical.

> **Both cardinal and ordinal recovery are bounded by the transition structure.**
> Widening the gaps improves ordinal recovery where it does nothing for cardinal
> recovery, so the two obey different laws — but neither converges.

### The corrected conclusion, which is stronger

The practical implication changes and improves:

- ordinal information **suffices** for mechanism design (exp38, oracle ordering);
- ordinal information is **not learnable** from consumption data (exp39);
- therefore the ordering must be **supplied from domain knowledge**.

This reinforces rather than weakens the paper's central practical claim. "Specify,
don't estimate" now covers both magnitudes and order. What T-F genuinely
contributes is that the *specification burden is lighter than it appeared*: an
operator must say which actions are more damaging than which, not how damaging each
one is. That is a real reduction in what must be elicited — and it is elicitation,
not estimation, either way.

---

## T-G — why rollout reaches 99.5–99.9%

### The conjecture was wrong

Proposed: an error survives one improvement step only if the base policy is wrong
at *two* separate points, so the rollout gap should scale as the **square** of the
base gap. Predicted log-log slope 2.

Measured (`exp40`, exact DP reference, 12 instances per cell):

| base policy | mean ratio (rollout gap / base gap) | std | gap closed | log-log slope |
|---|---|---|---|---|
| greedy | 0.0406 | 0.0117 | **95.9%** | **1.11** |
| ECI | 0.1067 | 0.0265 | **89.3%** | **1.25** |

Slopes are ~1.1–1.25, not 2. **One improvement step closes a fixed fraction of the
base gap; it does not square it.** The ratio is near-constant within each base
policy across a base gap ranging from 1.9% to 31.9%.

(An initial pooled fit gave slope 0.59, which is meaningless — pooling two base
policies with different constants produces a spurious exponent. The per-policy fits
are the correct analysis.)

### The finding that came out instead

Rollout closes **95.9%** of greedy's gap but only **89.3%** of ECI's. That looks
backwards — the better base policy is improved *less*, proportionally.

The explanation is that the two base policies leave behind different kinds of
error. Greedy's mistakes are mostly shallow: a single bad choice that one step of
lookahead can see and correct. ECI has already removed those, by construction — the
`δκ_a(T-t)` charge is exactly a one-step correction. What remains under ECI are
errors that genuinely require deeper lookahead, and a single improvement step
recovers proportionally less of them.

> **Corollary.** The value of one rollout step is inversely related to how much
> one-step correction the base policy already applies. This predicts diminishing
> returns from stacking corrections: ECI plus rollout is close to what a single
> improvement step can deliver, and further gains need genuinely deeper search.

Practically this settles the design question the empirical results raised. ECI alone
where compute is tight; ECI plus one rollout step where decisions are expensive
enough to justify `m·K·T` simulations; and beyond that, deeper search rather than
more corrections at the same depth.

---

## The ordinal saturation point: a closed form, and a correction

### τ ≈ 0.68 was an artefact of n = 8

exp39 held the pool at `n = 8` throughout and concluded rank recovery saturates
near `τ = 0.68`. Sweeping pool size instead (`exp41`, σ = 0.001, spacing = 4.0):

| n | τ | usable fraction | implied k | mean min-side | predicted τ |
|---|---|---|---|---|---|
| 5 | 0.627 | 0.507 | **2.47** | 3.1 | 0.194 |
| 8 | 0.614 | 0.708 | **2.33** | 6.7 | 0.472 |
| 12 | 0.757 | 0.786 | **2.57** | 9.3 | 0.603 |
| 16 | 0.769 | 0.846 | **2.47** | 13.1 | 0.707 |
| 22 | 0.840 | 0.877 | **2.70** | 19.0 | 0.764 |
| 30 | **0.851** | 0.913 | **2.60** | 25.6 | 0.831 |

**The implied number of positionally unusable arms is constant at `k ≈ 2.5`,
independent of pool size.** The interpretation is direct: roughly the first arm
destroyed and the last arm destroyed have one side of their transition nearly
empty; every other arm sits in the interior with data on both sides. That is a
constant, not a fraction.

### Closed form

Kendall's τ counts concordant minus discordant pairs. Pairs involving an unusable
arm are coin flips contributing nothing in expectation, so

```
E[τ]  ≈  C(n−k, 2) / C(n, 2)  =  (n−k)(n−k−1) / (n(n−1))  →  1  as n → ∞
```

with `k ≈ 2.5`. The prediction tracks well at larger pools (0.831 predicted against
0.851 observed at n = 30) and underestimates at small pools, where the
usable/unusable split is not clean enough for a binary pair-counting model.

### The contrast with cardinal recovery — and a normalisation trap

A first comparison suggested cardinal recovery *also* improves with pool size
(normalised RMSE falling 0.759 → 0.393). That was an artefact: normalising by the
spread of the true values, which grows with `n` at fixed spacing, manufactures the
improvement.

Raw, unnormalised:

| n | episode length N | mean min-side | raw cardinal RMSE | ordinal τ |
|---|---|---|---|---|
| 5 | 17 | 3.0 | **0.4439** | 0.608 |
| 8 | 31 | 6.8 | 0.6695 | 0.597 |
| 16 | 56 | 13.2 | 1.0996 | 0.752 |
| 30 | 107 | 25.7 | **1.3533** | **0.853** |

**Cardinal recovery gets worse with pool size; ordinal recovery gets better.** More
arms means a larger accumulated burden and more coefficients to separate, so
magnitudes become harder. But the positional damage that limits ranking stays fixed
at ~2.5 arms, so ordering becomes easier.

> **The two genuinely obey different laws, and now for a demonstrated reason rather
> than a conjectured one.** Cardinal error is driven by the number of coefficients
> and the burden scale, both growing in `n`. Ordinal error is driven by the number
> of positionally compromised arms, which does not grow.

### What this restores, and what it does not

This partly revives the T-F reading, on firmer ground. Ordinal information is
easier to obtain than cardinal information — but the mechanism is pool size, not
noise. In large pools the ordering is largely recoverable (τ = 0.85 at n = 30);
in small ones neither is.

**The practical caveat matters.** Irreversible-action settings are typically
small-pool: a handful of dose levels, a few process settings, a modest tool roster.
That is the regime where both recoveries are weakest, and it is exactly the regime
that motivates the problem. The corrected practical conclusion is therefore
unchanged in substance: **specify, do not estimate** — with the refinement that in
large pools the ordering may be recoverable, and that the specification burden is
lighter than a full cardinal elicitation.

---

## Can identification be improved by design? No.

If `k ≈ 2.5` is caused by the first and last arms destroyed having one-sided data,
a policy that sequences its consumption deliberately ought to reduce it. Three
rules were tried against random allocation (`exp42`): **delayed** (spend early
pulls on robust arms to build a baseline), **front-loaded** (destroy fragile arms
early so every transition has a long after-period), and **bracketed** (both).

| pool | allocation | τ | k | value |
|---|---|---|---|---|
| n=8 | **random** | **0.626** | 2.63 | −132.12 |
| n=8 | delayed | 0.502 | 2.63 | −79.83 |
| n=8 | front-loaded | 0.469 | 3.00 | −132.96 |
| n=8 | bracketed | 0.443 | 2.60 | −83.16 |
| n=16 | **random** | **0.761** | 2.37 | −1305.60 |
| n=16 | delayed | 0.678 | 2.20 | −979.21 |
| n=16 | front-loaded | 0.603 | 2.53 | −1544.33 |
| n=16 | bracketed | 0.593 | 2.40 | −987.18 |

**Every deliberate rule makes identification worse.** Random allocation gives the
highest τ at both pool sizes.

### Three explanations tested, three falsified

1. **Collinearity from concentrating pulls.** Wrong: the structured rules have
   *better* condition numbers (1.4e11 against random's 2.2e11) and *more* balanced
   pull distributions (0.888 against 0.872).
2. **Fewer positionally compromised arms.** Wrong: `k` stays near 2.5 under every
   rule (2.20–2.53), confirming it as a structural constant but explaining nothing
   about the variation.
3. **Fewer transitions.** Wrong: all rules exhaust the pool, so all produce the
   same 16 deaths.

### What is actually established, and the model's limit

The residual explanation is the **spread** of destruction times rather than the
count of extremes. Random allocation distributes destructions evenly across the
episode; structured rules cluster them into a phase.

This exposes a limitation of the closed form worth stating plainly: the binary
usable/unusable pair-counting model predicts 0.700–0.736 for all four rules, while
observed τ ranges 0.593–0.761. **The model captures the pool-size scaling well but
not the allocation-rule dependence.** A refinement would weight pairs by the
precision of both estimates rather than thresholding them.

### The practical conclusion, which is the useful part

> **Identification cannot be bought by design.** Random allocation is already the
> best of the rules tested, and the positional penalty is not something a cleverer
> policy removes.

Note also the value column: the two rules that improve realised value (delayed
+40%, bracketed +37% at n=8) are among the worst for identification. Playing well
and learning why one should play well are in tension, which is the same tension
Theorem 3 describes, now visible at the level of allocation design rather than
sample size.

---

## Closed form for k — derived

`k ≈ 2.5` was an empirical constant in exp41 and invariant to allocation in exp42.
It now has a derivation.

### Derivation

Arm `i` is destroyed at round `d_i`, and its transition has
`m_i = min(d_i, N − d_i)` usable observations on the thinner side. Call it unusable
if `m_i < c` for a precision threshold `c`.

Under an allocation that spreads pulls evenly, one arm is pulled per round and dies
with probability `≈ p̄`, so destructions occur at an approximately constant rate and
`d_i/N` is approximately uniform on `[0,1]`. Then

```
P(unusable) = P(d_i < c) + P(d_i > N − c) = 2c/N
```

and the expected count over `n` arms is `k = 2cn/N`. The key step is that `N` is
not free: by the argument in Theorem 3 the episode ends when the pool is exhausted,
so `N = N_exh = Σ_j 1/p_j ≈ n/p̄`. Substituting:

```
k  =  n · 2c · p̄ / n  =  2 · c · p̄
```

**The pool size cancels.** `k` depends only on the precision threshold and the mean
destruction rate. That is exactly the invariance exp41 observed and could not
explain: `N_exh` grows linearly in `n`, so the fraction of arms falling in the two
end-windows falls as `1/n` while the number of arms rises as `n`.

### Verification

**(P1) independent of n** — predicted 1.74 throughout:

| n | 6 | 10 | 16 | 24 | 40 |
|---|---|---|---|---|---|
| measured k | 2.40 | 2.47 | 2.23 | 2.43 | 2.32 |

**(P2) linear in threshold c** and **(P3) linear in rate p̄** — both confirmed
(k rises 1.32 → 5.23 as c goes 1 → 8; and 1.47 → 4.47 as p̄ goes 0.09 → 0.70).

### The residual is a constant, and it is interpretable

The leading term systematically underestimates by a fixed amount:

| residual across 13 settings | mean | sd | range |
|---|---|---|---|
| measured − 2cp̄ | **0.613** | 0.149 | [0.26, 0.92] |

Constant across pool sizes 6–40, thresholds 1–8, and rates 0.09–0.70. Its source is
identifiable: **the last arm destroyed has zero observations after its transition
by construction**, whatever `c` and `p̄` are, so it is unusable regardless of the
end-window argument. The uniform-density approximation cannot see this because it
treats the boundary as a limit rather than an atom.

### Final form

```
k  =  a + 2·c·p̄ ,        a ≈ 0.6
```

Mean absolute error **0.135** across the tested range.

| setting | measured | predicted | error |
|---|---|---|---|
| c=1 | 1.32 | 1.18 | +0.14 |
| c=3 | 2.23 | 2.34 | −0.11 |
| c=8 | 5.23 | 5.25 | −0.02 |
| p̄=0.09 | 1.47 | 1.15 | +0.32 |
| p̄=0.70 | 4.47 | 4.81 | −0.34 |

**Status.** The `2cp̄` term is derived; `a` is measured, with an identified cause but
no closed form. Combining with the pair-counting model of exp41 gives a full
prediction for ordinal recovery,
`E[τ] ≈ C(n−k, 2)/C(n, 2)` with `k = 0.6 + 2cp̄` — which reproduces the pool-size
scaling but, as recorded in exp42, not the allocation-rule dependence.


---

# Proof status — complete inventory

Every claim in the project, classified by what actually supports it. This exists so
that "we proved X" is never ambiguous.

## Proven exactly

| result | statement | proof |
|---|---|---|
| **T1 sufficiency** | constant `κ` ⟹ greedy optimal at every state and horizon | Expected burden is policy-invariant under constant `κ`, because each round contributes `κ` in expectation whatever arm is chosen, and `N = min(T, Σ G_j)` is allocation-order independent. Maximising value then reduces to the `e ≡ 0` problem, where memorylessness gives greedy. Verified exactly by DP: burden identical to 1e-16 across five structurally different policies, in both exhausting and non-exhausting regimes. |
| **T1 necessity** | `κ` varies ⟹ ∃ instance where greedy is strictly suboptimal | The hot/safe construction gives gap exactly `m·δE > 0`. Exact at 8/8 settings. |
| **T2b** | pure sequencing optimum is ascending in `e`; values irrelevant | `Σ v_a` is constant when every arm is pulled once, so only the burden schedule matters; an arm at position `s` pays `(n−s)` times. Brute force over all `n!` orderings: exact at 8/8. |
| **T3** | `SE(δê_i) ≥ 2σ/√min(T, N_exh)`, non-decreasing in `T` | Difference-in-means is the MLE conditional on other `e_j` known; adding free parameters weakly increases variance (Schur complement), so the oracle variance lower-bounds the joint. AM–HM gives `4σ²/N`. Three-part numerical verification; inequality chain holds 14/14. |
| **T4** | zero-regret policy loses `m·δE`, unbounded | Explicit construction. Greedy's regret is zero by definition of the benchmark; the gap is computed in closed form. Exact at 15/15 against DP. |
| **T5** | near-optimal policies are *forced* to report private regret `m·ε` | Explicit construction; the optimum declines a strictly best-available arm at each of `m` deferrals. Exact at 8/8. |
| **T6** | sequential `m`-agent play ≡ one learner over `mT` pulls | Identical MDP: same state, same reward function of state, same transition. Verified at five `(m,T)` pairs to within 0.015. |
| **ECI exactness** | constant `κ` ⟹ ECI optimal | The charge is arm-independent, so ECI reduces to greedy; Theorem 1 applies. Verified at std(κ) ≈ 1e-17. |

## Derived, with a measured constant

| result | derived part | measured part |
|---|---|---|
| **k** (positional penalty) | `k = 2·c·p̄`, pool size cancels because `N_exh ∝ n` | additive `a ≈ 0.6` from the last-destroyed arm, which has zero after-data by construction. Mean abs. error 0.135–0.185. |

## Empirical, with a mechanism identified but no proof

| result | evidence | what is missing |
|---|---|---|
| ECI captures 52–99% of the gap | exact DP across dispersion levels | no bound on the omitted option term |
| rollout closes a constant fraction | per-policy log-log slopes 1.11, 1.25 | no proof the fraction is constant |
| free-riding costs 12–14% of system value | decentralised simulation, 2 and 3 agents | no bound in terms of `m` |
| ordinal `E[τ] ≈ C(n−k,2)/C(n,2)` | tracks pool-size scaling (0.831 vs 0.851 at n=30) | fails to reproduce allocation-rule dependence (predicts 0.700–0.736 against observed 0.593–0.761) |
| identification cannot be improved by design | 4 allocation rules, random best at both pool sizes | mechanism not isolated; three candidate explanations falsified |
| terminal commitment failure modes | 300 seeds/cell, exact operating-value evaluation | no closed form for the optimal experiment allocation |

## Falsified and retained in the record

IPIC preservation heuristic · endogenous/exogenous unification claim ·
effective-horizon index refinement · horizon-scaling of the T4 gap · ECI exactness
in pure sequencing · per-arm and feature-based learning of `e` · `√m` scaling of
multi-agent estimation error · value of communication · uniform-tax mechanisms ·
quadratic rollout scaling · ordinal learnability from data · collinearity
explanation for allocation-rule dependence · τ ≈ 0.68 as a universal ceiling

Added to the falsified list: price of anarchy under sequential consumption ·
collisions as a coordination cost · spreading agents across arms as a remedy.

**Sixteen falsified hypotheses, all mine.** The ratio of falsified to surviving
claims is the reason the surviving ones are worth stating.

---

## ECI: one property proven, the general bound still open

### Proposition (ECI exactness under constant κ) — PROVEN

> If `κ_a = κ` for all arms, ECI is exactly optimal.

**Proof.** The ECI score is `v_a − B(S) − δκ_a(T−t)`. Under constant `κ` the charge
`δκ(T−t)` is identical across arms, as is `B(S)`, so both drop out of the argmax
and ECI selects `argmax_a v_a` — that is, ECI *reduces to greedy*. Theorem 1
sufficiency says greedy is optimal at every state and horizon under constant `κ`.
∎

**Verification.** Constructing instances with `e_a = κ/p_a` so dispersion is exactly
zero:

| κ | std(κ) | greedy gap | ECI gap |
|---|---|---|---|
| 0.10 | 1.4e-17 | **0.0000%** | **0.0000%** |
| 0.40 | 6.0e-17 | **0.0000%** | **0.0000%** |
| 1.00 | 2.4e-17 | **0.0000%** | **0.0000%** |

This is the first proven statement about ECI, and it fixes one end of any bound
exactly: the gap must vanish with dispersion, and it does, at precisely the point
Theorem 1 identifies.

### A methodological correction

An earlier sweep varied the *spread of `e`* and reported "dispersion 0.000" at
spread zero. That was wrong: with `e` constant but `p` varying, `κ = p·e` still has
dispersion 0.170, so the sweep never tested the zero-dispersion case at all. The
true test requires constructing `e = κ/p`.

### How the gap actually behaves — in RAW terms, not percentages

Percentage gaps are unusable at strong coupling because `V*` itself falls toward
and through zero, so the denominator vanishes. Reporting raw losses instead:

| δ | V* | raw greedy loss | raw ECI loss | ECI/greedy |
|---|---|---|---|---|
| 0.02 | 5.979 | 0.125 | 0.059 | 0.47 |
| 0.12 | 5.617 | 0.867 | 0.106 | 0.12 |
| 0.45 | 3.122 | 5.237 | 0.106 | 0.020 |
| 0.80 | 1.031 | 5.295 | **0.056** | 0.011 |
| 1.50 | **−4.339** | 17.270 | **0.116** | 0.0067 |

**The ECI loss is remarkably flat in absolute terms — 0.056 to 0.116 across the
entire range — while greedy's grows from 0.125 to 17.27, a factor of 138.**

### A correction to the previous version of this section

An earlier draft of this analysis reported "the ECI gap reaches 26.91% at δ = 1.5"
and concluded that no uniform smallness claim was defensible. **That figure was a
normalisation artifact.** At δ = 1.5 the optimum is `V* = −4.34`; dividing a raw
loss of 0.116 by a near-zero or negative optimum produces arbitrary percentages,
which is also the source of the accompanying "greedy gap 4944.8%".

The corrected reading is the opposite of the earlier one: ECI's absolute loss is
close to constant across coupling strengths that vary greedy's loss by two orders
of magnitude. Percentage reporting should be dropped wherever `V*` is small.

### What remains open

A bound on the ECI gap in terms of `(δ, std(κ), T, n)`. The decomposition
`error_a = δκ_a(L − (T−t)) + p_a O_a` identifies the two error sources — replacing
the true burden horizon `L` by the naive `(T−t)`, and dropping the option term —
but bounding `O_a` is exactly the difficulty that made the original Theorem 1
coupling argument fail, and no closed form for it exists (established in exp13).

---

## Free-riding is charge misspecification — and that generalises

Theorem 6 collapsed the multi-agent setting into a single-learner one, which makes
free-riding far easier to analyse. An agent discounting the externality by its own
expected share charges `λ·δκ_a(T−t)` with `λ = 1/m`, so **free-riding is exactly
ECI with a mis-scaled charge**. The same sweep answers a broader question: how
robust is ECI to getting the scale wrong?

### The free-riding curve

Raw loss against the optimum (`n=6`, `T=8`, `δ=0.15`, spread 1.2, 10 instances):

| λ | reading | raw loss |
|---|---|---|
| 0.000 | greedy | 1.3734 |
| 0.125 | 8 agents | 0.7030 |
| 0.250 | 4 agents | 0.4539 |
| 0.333 | 3 agents | 0.3678 |
| 0.500 | 2 agents | 0.2262 |
| 1.000 | correct pricing | 0.0843 |

Loss is monotone in the discount, so free-riding degrades gracefully rather than
collapsing. **Even eight-agent free-riding (λ = 1/8) recovers 50–63% of the
available gain over greedy** across settings — crude partial pricing is worth far
more than none.

### The optimum is not at λ = 1, but it is not stable either

At the base setting, `λ = 1.5` beats correct pricing (0.0358 against 0.0843). The
explanation is structural: ECI's charge omits the option term, so it *under*-prices
the true marginal cost and scaling up compensates.

But the location moves with the setting, and not slightly:

| setting | argmin λ |
|---|---|
| short horizon T=4 | **0.60** |
| strong coupling δ=0.4 | 1.00 |
| high dispersion | 1.50 |
| base | 1.75 |
| long horizon T=14 | 2.50 |
| weak coupling δ=0.04 | **3.00** |

Ranging from 0.6 to 3.0. **No universal recommendation for λ is defensible**, and
any "use λ = 1.5" claim would be overfitting the base setting.

### What *is* robust: the asymmetry

Comparing a threefold under-charge against a threefold over-charge:

| setting | under (λ=1/3) | over (λ=3) | ratio |
|---|---|---|---|
| base | 0.3678 | 0.1114 | 3.30 |
| T=14 | 0.5021 | 0.0251 | **19.99** |
| δ=0.04 | 0.1763 | 0.0256 | 6.88 |
| spread 0.4 | 0.2857 | 0.0343 | 8.34 |
| n=5, T=12 | 0.2714 | 0.0133 | **20.34** |
| T=4 | 0.0588 | 0.1277 | 0.46 |

**Over-charging is safer in 8 of 9 settings**, often by an order of magnitude. The
single exception is the short horizon `T=4`, where there is little future left for
the burden to damage and caution simply forfeits present reward.

> **Practical rule.** When the externality scale is uncertain — which by Theorem 3
> it always is — err high. Under-pricing costs 3× to 20× more than over-pricing by
> the same factor.

This explains the earlier finding that the crude conservative policy beat every
learned estimator: it was not merely robust, it was erring in the safe direction.
It also completes the answer Theorem 3 leaves open. Theorem 3 says the scale cannot
be estimated; this says the estimate does not need to be good, provided the error
is on the high side.

---

## Repository audit

A full audit after 48 experiments of accumulated change. Three classes of defect
were found and fixed; recorded because an artifact that silently rots is worse than
one with known gaps.

### Stale claims surviving the retractions

The multi-agent retraction updated `THEORY.md` but left the claim standing in three
other places:

- `RESEARCH_PROGRAMME.md` status table still read "PoA reaches 1.297; κ-aware
  agents reduce it to 1.004";
- the same file's Stage 1 plan still listed price of anarchy as a headline
  measurement;
- `scenarios.py` still documented `shared-quota-competing` as exhibiting "price of
  anarchy under consumption", with an expected behaviour that no longer holds.

All three corrected. The scenario now documents free-riding under a shared burden,
and states explicitly that this is *not* a classical commons because the burden
falls on the agent creating it.

Separately, the ordinal section's "saturates near τ ≈ 0.68" stood uncorrected above
the later section that supersedes it. A forward reference now marks it.

### Import side effects

Six early experiments (`exp04`, `exp06`, `exp07`, `exp08`, `exp09`, `exp13`)
executed their full driver on import, emitting up to 3,265 characters and running
minutes of computation. This is a genuine defect for a package others import from,
and it made the test suite's imports far more expensive than they appeared. All six
are now behind `main()` guards; the full experiment directory imports silently.

### Escape-sequence warnings

Four modules had `\{` in docstrings, raising `SyntaxWarning` under Python 3.12.
Docstrings made raw.

### Verification

- syntax: all 48 experiments parse
- imports: all 48 import silently, none fail
- reproduction: spot-checked `exp06`, `exp08`, `exp13` against their documented
  numbers — `+27.2%` at strong coupling, `92.0%` capture at the highest dispersion,
  and 0 violations in 24,946 states, all matching
- tests: full suite passes

---

## The ECI bound — first closed-form guarantee, and a disproven lemma along the way

The proof route attempted since session 1 — bounding the option term `O_a`
directly — has no known closed form (`exp13`). A different route succeeds: avoid
`O_a` entirely by bounding the whole term `D_a` through a magnitude argument
instead of a sign argument.

### The first attempt failed, and the failure is itself a real result

**Conjectured lemma:** `V(S',t) ≥ V(S,t)` whenever `S' ⊇ S` — having more available
arms is never worse, since a policy can just "ignore" the extra ones.

**Falsified**, decisively: 904 violations in 11,904 checked triples, worst case
`V(S') − V(S) = −0.744`.

**The mechanism, confirmed exactly:** the model has **no null action**. If arm `1`
is the only survivor and it dies, the episode ends (`V = 0`, exhaustion). If arm `5`
also happens to be available, the episode does *not* end when arm 1 dies — the
policy is forced to keep pulling arm 5, and if the accumulated burden by then makes
every remaining round strictly negative (`v_5 − B = 0.501 − 1.225 = −0.724` in the
witnessed case), forced continuation is worse than the game simply stopping.

> **"Ignore the extra arm" is not a valid strategy when standing still is not an
> option.** Extra options can be a liability precisely because the model always
> forces play until the horizon or the pool is exhausted.

This is a genuine structural fact about consumable-action-set MDPs, not previously
noted, and it rules out the most natural route to a clean bound.

### The successful route

Replace the sign-based monotonicity argument with a magnitude bound that needs no
monotonicity at all:

**Lemma (magnitude bound).** `|V(S,t)| ≤ (T−t)·R` for any `S`, where
`R := v_max + δ·Σe_i` bounds one round's reward magnitude regardless of sign.
*Proof.* Each round's reward is `v_a − B(S)` with `v_a ∈ [0, v_max]` and
`B(S) ∈ [0, δΣe_i]`, so `|v_a − B(S)| ≤ R`. `V(S,t)` sums at most `T−t` such terms
(fewer if the pool empties early, which only pulls the sum toward 0). Induction on
the DP recursion.

**Theorem (performance difference, exact).**
`Gap(π) = 𝔼_π[Σ_t reg(S_t, t)]`, telescoped over π's own trajectory distribution,
where `reg(S,t)` is the one-step regret of π's action in true `Q*`-value. Verified
to machine precision (max error `1.19 × 10⁻¹⁵`) — an exact identity, not an
approximation, confirmed by comparing the telescoped sum against the directly
computed gap on 6 random instances.

**Combining:** `|D_a(S,t)| ≤ 2(T−t)R` (magnitude bound applied to both terms of
`D_a = V(S,t+1) − V(S∖{a},t+1)`), giving `|error(a)| ≤ (T−t)·p_a·(δe_a + 2R)` and,
via a standard argmax-perturbation step and the telescoping identity:

```
Gap(ECI)  ≤  T(T+1) · max_a[ p_a·(δe_a + 2R) ],     R = v_max + δ·Σe_i
```

**This is the first closed-form bound on ECI's approximation gap.** It uses only
the problem's own primitives — no `O_a`, no undetermined `L(S,t)`.

### Verification

Never violated across 10 random instances and across the full extreme-coupling
sweep that previously produced the largest raw ECI losses (`δ ∈ [0.02, 3.0]`).

But it is **very loose** — 1,335× to 72,569×, averaging ~14,000×. The bound proves
`Gap(ECI)` is finite and scales as `O(T²δ)`, correctly capturing the qualitative
behaviour, but is far from the empirically tiny actual gaps (0.003–0.20 in the same
instances). Tightening it — likely by bounding `D_a` through the actual burden
horizon rather than the full episode magnitude — is the natural next step and is
recorded as open, not claimed.

### Status update

| result | before | now |
|---|---|---|
| ECI general bound | open, no route known | **closed-form bound proven, loose** |
| Monotonicity of V in available set | untested assumption | **disproven, with mechanism identified** |
| Performance difference for this MDP class | not stated | **proven exactly** |

Two of these three are genuinely new theorems. The monotonicity disproof is worth
weighing equally with the bound itself — it is the kind of finding that changes how
one reasons about the model, not just a rejected lemma.

---

## Stress-testing the independence assumption — correlated destruction

Every theorem so far assumes destruction is independent across arms. This is false
in exactly the domains motivating the project: a shared API provider's outage
kills several tools at once; a class-wide toxicity mechanism can invalidate
several doses together; a common raw-material shortage blocks several process
settings simultaneously. `exp50` adds a cluster-shock layer: alongside the
existing per-pull destruction, each cluster faces an exogenous shock each round
that, if it fires, destroys **every currently alive arm in that cluster
simultaneously**, regardless of which arm was pulled.

### Q1 — does Theorem 4 (zero-regret vacuity) survive? Yes, trivially

Greedy's regret against the best-available benchmark is exactly `0.0000000000` at
every shock rate tested (`q ∈ {0, 0.1, 0.3, 0.5}`). This isn't a new result so much
as a clarification: greedy's zero regret is a **definitional** property of the
benchmark (greedy always pulls the best available arm, whatever "available" turns
out to mean), so it holds under *any* destruction mechanism — correlated,
adversarial, anything. Theorem 4 was never resting on independence in the first
place.

### Q2 — does Theorem 1 survive? Yes, exactly, and this is a real surprise

**Verified to full floating-point precision** (gap `0.00e+00` to 10 decimal
places) across shock rates up to `q = 0.7`, including a deliberately adversarial
setup with `e` varying 3–6× within a cluster while `κ = pe` is held exactly
constant. Greedy remains exactly optimal.

> **Conjecture (Theorem 1 under correlated exogenous destruction).** Constant `κ`
> implies greedy optimality even when destruction includes correlated,
> cluster-level exogenous shocks, provided the shock mechanism does not depend on
> which arm was pulled.

This is stated as a **conjecture, not a theorem**. A proof attempt was made and
abandoned honestly rather than forced: the original burden-invariance argument
relies on each round's expected burden contribution being `κ` regardless of the
arm chosen, and that argument nests awkwardly once a shock's effect on *other*
cluster members depends on which arm was just removed from the cluster. The
empirical result is real and precise; the general proof is open. Given this
project's history with a superficially plausible but false lemma (the
monotonicity claim in the ECI-bound work), a conjecture stated honestly is
preferred here over a proof pushed through on pattern-matching.

### Q3 — does ECI survive? No, and this is the real finding

Averaged over 15 instances per shock rate:

| q | greedy gap | ECI gap | ECI worse than greedy |
|---|---|---|---|
| 0.00 | 0.5268 | 0.0643 | 0/15 |
| 0.10 | 0.3222 | 0.0152 | 0/15 |
| 0.30 | 0.0894 | 0.0333 | 1/15 |
| **0.50** | 0.0221 | **0.0820** | **9/15** |
| **0.70** | 0.0024 | **0.1448** | **13/15** |

**At high correlated-shock rates, ECI is worse than doing nothing (greedy) in the
majority of instances.** This is not noise — it reproduces robustly.

**Why.** ECI prices an arm by its own `κ_a = p_a e_a`, betting that preserving
low-`κ` arms protects future value. Once destruction is dominated by an
exogenous, correlated shock, that bet stops paying: the shock destroys arms
whether or not they were pulled, so declining a good pull to "preserve" an arm
buys nothing when the arm's survival was never in the policy's control to begin
with. ECI keeps paying the preservation cost without collecting the benefit.

### A natural fix was tried and failed

The obvious correction — charge an arm not only for `κ_a` but for the expected
burden its cluster-mates' shock exposure creates — was tested and made things
**worse**, not better:

| q | greedy | ECI | cluster-aware ECI |
|---|---|---|---|
| 0.30 | 1.8654 | 1.9215 | **1.8442** |
| 0.50 | 1.5088 | 1.4489 | **1.4255** |
| 0.70 | 1.2729 | 1.1305 | **1.0456** |

The cluster-aware charge over-penalises: at `q = 0.7` it falls even below plain
greedy. The correction's scaling (raw `q × cluster e`) is evidently wrong, and a
proper fix would need to account for how the charge should interact with the
existing `κ` term rather than being added on top of it. **This is recorded as a
falsified fix attempt, not a solved problem.**

### Where this leaves the paper's practical recommendation

The central practical claim — "specify the externality, don't try to estimate
it, and use ECI" — needs a boundary condition added: **it holds where destruction
is independent across arms, and needs a not-yet-found correction where
destruction is correlated.** For the domains motivating this work, that boundary
matters directly: agent tools sharing a provider, doses sharing a toxicity
mechanism, and process settings sharing a supplier are all naturally clustered.
Practitioners in exactly the settings this paper targets should check whether
their destruction risk is independent before deploying ECI as-is.

This is genuinely unresolved and belongs in the paper's limitations, not
smoothed over.

---

## SECI: a derived fix for correlated destruction — strong at small scale, partial at scale

### Derivation

ECI's blind spot under correlated shocks: it only discourages *pulling* an arm
(burden-avoidance), but never accounts for the cost of *not* pulling — an arm held
in reserve can be destroyed by a cluster shock before ever being used, losing its
value entirely rather than merely delaying it. As shock rate `q` rises, burden
increasingly happens regardless of the policy's choices, so the fraction of
destruction actually under the policy's control shrinks, and ECI's charge should
shrink with it.

**SECI (shock-corrected ECI):**

```
I_SECI(a,t) = v_a − δ·p_a·e_a·(T−t)·(1−q)²
```

Matches ECI exactly at `q=0`. Degrades smoothly to pure greedy as `q→1`, which is
the correct limit: when destruction is certain regardless of action, preservation
is worthless and greedy is exactly right.

### Small scale (exact DP, n=6, 30 seeds/cell, q from 0 to 1)

| q | V* | greedy | ECI | **SECI** | gap to V* |
|---|---|---|---|---|---|
| 0.0 | 4.667 | 3.804 | 4.596 | **4.596** | 1.53% |
| 0.3 | 2.121 | 2.029 | 2.079 | **2.115** | 0.32% |
| 0.7 | 1.337 | 1.328 | 1.228 | **1.336** | **0.06%** |
| 1.0 | 1.088 | 1.088 | 0.971 | **1.088** | **0.00%** |

**SECI matches or beats greedy at every single q tested, confirmed across 30
seeds**, staying within 0.01–1.53% of true optimum throughout, while plain ECI's
gap grows to ~11% at `q=1`.

### At scale (Monte Carlo, n=40, 8 clusters of 5, 300 seeds × 5 instances)

| q | greedy | ECI | **SECI** | SECI − greedy | SECI − ECI |
|---|---|---|---|---|---|
| 0.0 | 23.179 | 31.546 | **31.546** | **+8.367** | 0.000 |
| 0.1 | 2.967 | 3.184 | **3.297** | **+0.330** | +0.113 |
| 0.3 | 0.855 | 0.501 | 0.717 | −0.138 | +0.216 |
| 0.5 | 0.952 | 0.605 | 0.912 | −0.040 | +0.307 |
| 0.7 | 1.027 | 0.971 | 1.010 | −0.017 | +0.039 |

**SECI reliably beats plain ECI at every q — the small-scale fix transfers.** It
does **not** fully transfer against greedy: small residual shortfalls appear in
the mid-`q` range (−0.138 to −0.017), where the small-scale result showed SECI
winning outright. A cluster-size-scaled variant of the dampening exponent was
tried and did not close this gap (similar or slightly worse margins). This is
reported as a genuine, partial result — the fix substantially closes the ECI
failure mode and is never worse than plain ECI, but does not yet fully match the
small-scale guarantee once the environment scales to many clusters.

### Honest summary

| | independent (q=0) | correlated, small scale | correlated, at scale |
|---|---|---|---|
| ECI vs greedy | ECI wins | ECI **loses** at high q | ECI **loses** at high q |
| SECI vs ECI | ties (identical formula) | SECI wins always | **SECI wins always** |
| SECI vs greedy | ties (= ECI = optimal-tracking) | **SECI wins always** | SECI wins at low q, small gap at mid q |

SECI is a strict improvement over ECI in every regime tested and closes most —
not all — of the correlated-destruction gap. The residual scale-dependence is
recorded as open rather than papered over.
