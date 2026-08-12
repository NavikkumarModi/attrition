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

## Multi-agent: price of anarchy under consumption (flagship material)

Stage 1 of the programme, implemented (`exp32`). Several agents draw on one
consumable pool; each collects its own reward but every destruction raises the
burden for all. Restraint is a public good.

`PoA = W(planner) / W(decentralised)`, exact DP planner over subsets, round-robin
turn order:

| agents | std(κ) | planner | dec-greedy | dec-ECI | **PoA greedy** | **PoA ECI** |
|---|---|---|---|---|---|---|
| 1 | 0.192 | 4.552 | 4.506 | 4.558 | 1.010 | 0.999 |
| 1 | 0.544 | 4.216 | 4.005 | 4.203 | 1.053 | 1.003 |
| 2 | 0.268 | 5.483 | 4.863 | 5.212 | 1.127 | 1.052 |
| 2 | 0.465 | 6.460 | 5.396 | 6.157 | 1.197 | 1.049 |
| 3 | 0.224 | 6.764 | 6.162 | 6.514 | 1.098 | 1.038 |
| 3 | 0.583 | 6.152 | 4.743 | 6.129 | **1.297** | **1.004** |

**Two results.**

1. **Price of anarchy grows with both agent count and κ dispersion**, reaching
   **1.297** at three greedy agents with high dispersion — decentralisation costs
   30% of system value.
2. **κ-aware agents nearly eliminate it.** PoA under ECI stays at 1.00–1.05
   throughout, and at the worst greedy setting (1.297) ECI gives 1.004. Pricing the
   externality locally recovers almost all of the central planner's value *without
   any coordination*.

Result 2 is the substantive finding: the commons problem here is not a coordination
failure requiring a mechanism, it is a *mispricing* failure that each agent can fix
unilaterally.

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
