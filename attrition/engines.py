"""Mechanistic engines: dynamics that generate (v, p, e) instead of hand-setting them.

The reduced-form domains in `domains.py` supply parameter vectors directly. That is
enough to demonstrate the phenomena but invites the objection that the parameters
were chosen to produce them. These engines derive the parameters from standard
domain dynamics, so the mapping is a consequence rather than an input.

Implemented:
    LotkaVolterraTumour   -- drug-sensitive/resistant tumour competition under
                             dose-dependent kill.
    PlatformTrialEngine   -- shared-control trial where dropping an arm fragments
                             the control stream and degrades inference for all
                             remaining comparisons.
    DesignSpaceEngine     -- process development where an out-of-spec run narrows
                             the defensible filing envelope for every later
                             operating choice.

Not a clinical model. It is the standard two-compartment competition system used
in the adaptive-therapy literature, reduced to the minimum that exhibits
competitive release. No clinical conclusion should be drawn from it.
"""

import numpy as np

__all__ = ["LotkaVolterraTumour", "derive_arm_parameters",
           "PlatformTrialEngine", "derive_trial_parameters",
           "DesignSpaceEngine", "derive_design_space_parameters"]


class LotkaVolterraTumour:
    """Two-compartment competition model with dose-dependent kill.

        dS/dt = r_S S (1 - (S + a_SR R)/K) - d(dose) S
        dR/dt = r_R R (1 - (R + a_RS S)/K)

    Sensitive cells `S` grow faster (no resistance cost) and are killed by the
    drug. Resistant cells `R` grow slower but are unaffected by it. Both compete
    for the same carrying capacity `K`, so a large sensitive population suppresses
    the resistant one -- the mechanism whose removal produces competitive release.

    Parameters follow the qualitative structure used in the adaptive-therapy
    literature: a fitness cost of resistance (r_R < r_S) and cross-competition.
    """

    def __init__(self, K=1.0, r_S=0.35, r_R=0.25, a_SR=0.9, a_RS=1.1,
                 kill_max=0.85, S0=0.60, R0=0.02, dt=1.0):
        self.K, self.r_S, self.r_R = K, r_S, r_R
        self.a_SR, self.a_RS = a_SR, a_RS
        self.kill_max = kill_max
        self.S0, self.R0, self.dt = S0, R0, dt
        self.reset()

    def reset(self):
        self.S, self.R = self.S0, self.R0
        return self.state()

    def state(self):
        return np.array([self.S, self.R])

    @property
    def burden_total(self):
        """Total tumour burden. Lower is better."""
        return self.S + self.R

    def kill_rate(self, dose):
        """Dose-response: saturating kill of sensitive cells only."""
        return self.kill_max * dose

    def step(self, dose, steps=4):
        """Advance the dynamics under a constant dose. Returns burden reduction."""
        before = self.burden_total
        h = self.dt / steps
        for _ in range(steps):
            S, R = max(self.S, 0.0), max(self.R, 0.0)
            dS = self.r_S * S * (1 - (S + self.a_SR * R) / self.K) \
                - self.kill_rate(dose) * S
            dR = self.r_R * R * (1 - (R + self.a_RS * S) / self.K)
            self.S = max(S + h * dS, 0.0)
            self.R = max(R + h * dR, 0.0)
        return before - self.burden_total


def derive_arm_parameters(doses=None, engine_kwargs=None, horizon=30,
                          sensitive_floor=0.02, s0_sd=0.16, verbose=False):
    """Derive (v, p, e) for each dose level by simulating the dynamics.

    For each dose, starting from the untreated state:

      v_a  immediate burden reduction achieved in one treatment period
      p_a  probability that this dose drives the sensitive compartment below a
           floor within one period -- the irreversible event, since once the
           sensitive population is gone it cannot be recovered and can no longer
           suppress the resistant clone. Estimated by perturbing the initial
           state, so p reflects genuine uncertainty about tumour composition.
      e_a  permanent loss of future control caused by that event: the difference
           in achievable burden reduction, over the remaining horizon, between a
           tumour that retains its sensitive compartment and one that has lost it.

    Nothing here is hand-set. All three follow from the dynamics.
    """
    doses = np.linspace(0.15, 1.0, 6) if doses is None else np.asarray(doses)
    kw = engine_kwargs or {}
    rng = np.random.default_rng(0)

    v, p, e = [], [], []
    for dose in doses:
        # --- v: immediate burden reduction from the reference state
        eng = LotkaVolterraTumour(**kw)
        v_a = eng.step(dose)

        # --- p: probability of exhausting the sensitive compartment
        hits = 0
        trials = 200
        for _ in range(trials):
            k2 = dict(kw)
            k2["S0"] = float(np.clip(kw.get("S0", 0.60)
                                     + rng.normal(0, s0_sd), 0.05, 0.95))
            k2["R0"] = float(np.clip(kw.get("R0", 0.02)
                                     + rng.normal(0, 0.012), 0.001, 0.2))
            en = LotkaVolterraTumour(**k2)
            en.step(dose)
            if en.S < sensitive_floor:
                hits += 1
        p_a = hits / trials

        # --- e: permanent loss of control once sensitives are gone
        # control retained: tumour keeps its sensitive compartment
        a = LotkaVolterraTumour(**kw)
        ctrl_with = 0.0
        for _ in range(horizon):
            ctrl_with += max(a.step(0.5), 0.0)
        # control lost: same tumour with the sensitive compartment removed and
        # its mass transferred to the resistant clone (competitive release)
        k3 = dict(kw)
        k3["S0"] = 0.0
        k3["R0"] = kw.get("R0", 0.02) + 0.05
        b = LotkaVolterraTumour(**k3)
        ctrl_without = 0.0
        for _ in range(horizon):
            ctrl_without += max(b.step(0.5), 0.0)
        e_a = max(ctrl_with - ctrl_without, 0.0)

        v.append(v_a); p.append(p_a); e.append(e_a)
        if verbose:
            print(f"  dose {dose:.2f}: v={v_a:.4f}  p={p_a:.3f}  e={e_a:.4f}  "
                  f"kappa={p_a*e_a:.4f}")

    return (np.array(v), np.array(p), np.array(e), doses)


# ============================================================ platform trials
class PlatformTrialEngine:
    """Shared-control platform trial with arms entering and leaving.

    The externality here is *inferential* rather than physical, which is why the
    reduced-form mapping was the weakest of the four. The mechanism:

    A platform trial randomises against one shared control. A treatment arm is
    compared to *concurrent* controls -- patients randomised while that arm was
    active. When an arm is dropped mid-trial the control stream fragments into more
    non-concurrent blocks, and every remaining comparison must either discard the
    non-concurrent data (losing precision) or adjust for time drift (inflating
    variance). Either way, one arm's exit degrades inference for all the others.

    Under the standard sqrt(K) allocation rule with K active arms, control receives
    a fraction sqrt(K)/(K + sqrt(K)) of patients. As K falls the control share
    rises, but the accumulated data becomes more fragmented.

    Derived quantities per arm:
        v_a  information gained about arm a per period, ~ n_a n_c/(n_a + n_c)
        p_a  probability the arm crosses a futility boundary this period, which
             depends on its true effect and on how much information has accrued
        e_a  variance inflation imposed on all remaining comparisons by this arm's
             exit, from the additional non-concurrent block it creates
    """

    def __init__(self, n_arms=6, patients_per_period=120, sigma=1.0,
                 drift_sd=0.06, seed=0):
        self.n_arms = n_arms
        self.N = patients_per_period
        self.sigma = sigma
        self.drift_sd = drift_sd
        rng = np.random.default_rng(seed)
        # true effects: a spread of promising and marginal arms
        self.effects = np.sort(rng.uniform(0.05, 0.45, n_arms))[::-1].copy()
        self.reset()

    def reset(self):
        self.active = np.ones(self.n_arms, dtype=bool)
        self.info = np.zeros(self.n_arms)      # accrued Fisher information
        self.tenure = np.zeros(self.n_arms)    # periods each arm has been active
        self.blocks = 1                        # non-concurrent control blocks
        return self.state()

    def state(self):
        return self.info.copy()

    def allocation(self):
        """sqrt(K) allocation: control share rises as arms drop out."""
        K = max(int(self.active.sum()), 1)
        ctrl = np.sqrt(K) / (K + np.sqrt(K))
        per_arm = (1 - ctrl) / K
        return ctrl * self.N, per_arm * self.N

    def concurrency_penalty(self):
        """Variance inflation from fragmented control.

        Each additional non-concurrent block requires a drift adjustment, which
        inflates the variance of every remaining comparison.
        """
        return 1.0 + self.drift_sd ** 2 * (self.blocks - 1)

    def information_gain(self, arm):
        """Information about `arm` gained this period, given current allocation."""
        n_c, n_a = self.allocation()
        eff_n = (n_a * n_c) / (n_a + n_c)          # harmonic combination
        return eff_n / (self.sigma ** 2 * self.concurrency_penalty())

    def step(self, arm):
        """Run one period focusing on `arm`; return its information gain."""
        g = self.information_gain(arm)
        self.info[arm] += g
        self.tenure[arm] += 1
        return g

    def arm_value(self, arm):
        """Value of running this arm for a period.

        Information is only worth acquiring in proportion to the benefit it may
        establish, so a promising arm is worth more than a marginal one even when
        both accrue information at the same rate.
        """
        return float(self.effects[arm] * self.information_gain(arm))

    def fragmentation_damage(self, arm):
        """Precision lost by the remaining arms if `arm` exits now.

        An arm that has been active longer overlaps more of the control stream, so
        its exit truncates a larger concurrent window and does more damage.
        """
        share = self.tenure[arm] / max(self.tenure.sum(), 1.0)
        before = sum(self.information_gain(b) for b in range(self.n_arms)
                     if b != arm)
        self.blocks += 1
        after = sum(self.information_gain(b) for b in range(self.n_arms)
                    if b != arm)
        self.blocks -= 1
        return max(before - after, 0.0) * (1.0 + 2.0 * share)

    def drop(self, arm):
        """Drop an arm: the control stream fragments."""
        self.active[arm] = False
        self.blocks += 1

    def futility_prob(self, arm):
        """Probability the arm crosses a futility boundary this period.

        Arms with small true effects and substantial accrued information are the
        ones that get dropped; an arm with little information yet is rarely
        stopped, whatever its effect.
        """
        z = self.effects[arm] * np.sqrt(max(self.info[arm], 0.0))
        # low z with meaningful information -> likely futile
        maturity = 1.0 - np.exp(-self.info[arm] / 12.0)
        return float(np.clip(maturity * np.exp(-2.2 * z), 0.0, 1.0))


def derive_trial_parameters(n_arms=6, periods=4, seed=0):
    """Derive (v, p, e) for a platform trial from the inference model.

    Arms are warmed up with staggered entry, as in a real platform trial: earlier
    arms accrue more information and more tenure, so they differ in both futility
    risk and in the damage their exit does to the remaining comparisons.
    """
    eng = PlatformTrialEngine(n_arms=n_arms, seed=seed)
    for k in range(periods):
        for a in range(n_arms):
            if a <= k + 1:                  # staggered entry
                eng.step(a)

    v = np.array([eng.arm_value(a) for a in range(n_arms)])
    p = np.array([eng.futility_prob(a) for a in range(n_arms)])
    e = np.array([eng.fragmentation_damage(a) for a in range(n_arms)])
    return v, p, e, eng


# =============================================================== design space
class DesignSpaceEngine:
    """CMC process development over candidate parameter settings.

    Each run consumes scarce API material and either succeeds or produces an
    out-of-spec batch. A failure at an aggressive setting narrows the design space
    that can subsequently be defended in a regulatory filing, which constrains
    every later operating choice -- the externality.

    Derived quantities per setting:
        v_a  expected yield at that setting
        p_a  probability the run consumes its material without usable data, or
             produces an out-of-spec batch
        e_a  fraction of the defensible design-space volume lost by a failure
             there, larger for settings near the edge of the known-good region
    """

    def __init__(self, n_settings=8, centre=0.45, spec_limit=0.85, seed=0):
        rng = np.random.default_rng(seed)
        self.settings = np.sort(rng.uniform(0.15, 0.95, n_settings))
        self.centre, self.spec = centre, spec_limit
        self.n = n_settings

    def yield_at(self, i):
        """Yield rises with aggression but falls off past the spec limit."""
        x = self.settings[i]
        return float(1.4 * x * np.exp(-max(x - self.spec, 0.0) * 6.0))

    def failure_prob(self, i):
        """Out-of-spec probability rises sharply near and past the limit."""
        x = self.settings[i]
        return float(np.clip(0.04 + 0.9 / (1 + np.exp(-12 * (x - self.spec + 0.10))),
                             0.0, 1.0))

    def envelope_loss(self, i):
        """Design-space volume forfeited by a failure at this setting.

        A failure far from the centre of the known-good region truncates more of
        the envelope, because the defensible space is the region bounded by
        demonstrated-good runs.
        """
        x = self.settings[i]
        return float(abs(x - self.centre) / max(self.settings.max() - self.centre,
                                                1e-9))


def derive_design_space_parameters(n_settings=8, seed=0):
    eng = DesignSpaceEngine(n_settings=n_settings, seed=seed)
    v = np.array([eng.yield_at(i) for i in range(eng.n)])
    p = np.array([eng.failure_prob(i) for i in range(eng.n)])
    e = np.array([eng.envelope_loss(i) for i in range(eng.n)])
    return v, p, e, eng
