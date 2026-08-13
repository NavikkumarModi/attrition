# Changelog

## v0.2.0 — engineering pass

- **Bug fix:** `exp03_reward_vs_regret.py` (a session-1 experiment predating the
  guarding convention) executed on import. Guarded.
- **Bug fix:** stale `evolving_bandits`/`consumable-bandits` references in
  README's clone URL and citation key after the project rename.
- Renamed the project to **ATTRITION** (Agent Testbed for Task, Resource and
  Irreversible-Tradeoff Investigation) throughout the package, docs, and paper.
- Added `examples/` (quickstart, scenario sweep, mechanistic-therapy demo), CI
  (`.github/workflows/`: tests across Python 3.9–3.12, paper build), and
  `CONTRIBUTING.md`.
- **New theorem:** closed-form bound on ECI's approximation gap,
  `Gap(ECI) ≤ T(T+1)·max_a[p_a(δe_a + 2R)]`. First closed-form guarantee for ECI;
  loose (~14,000× on average) but never violated, including across the
  extreme-coupling regime.
- **New theorem, disproven:** monotonicity of `V` in the available arm set is
  false. Extra available arms can be a liability because the model has no null
  action — pool exhaustion (`V=0`) can beat forced continuation under high
  accumulated burden. Mechanism identified and verified.
- **New theorem, proven exactly:** the performance-difference identity for this
  MDP class, `Gap(π) = 𝔼_π[Σ_t reg(S_t,t)]`, verified to `1.19×10⁻¹⁵`.

## v0.1.0 — initial release

- Core library: `ConsumableBandit`, seven policies (`Greedy`, `ECI`,
  `Conservative`, `SortByE`, `Rollout`, `ThompsonSampling`, `UCB`).
- Six theorems proven exactly (T1 both directions, T2b, T3, T4, T5) plus the
  sequential multi-agent equivalence (T6).
- Three mechanistic engines (tumour dynamics, platform trials, design space),
  four reduced-form domains, seven named scenarios.
- Gym- and PettingZoo-compatible environment APIs.
- Terminal-commitment model (T-E) with two failure modes absent from the
  sequential setting.
- Three paper builds (reading, arXiv, JMLR) sharing one source.
- 64 regression tests, one per theorem/claim.
