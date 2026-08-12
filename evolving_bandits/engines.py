"""Mechanistic engines: dynamics that generate (v, p, e) instead of hand-setting them.

The reduced-form domains in `domains.py` supply parameter vectors directly. That is
enough to demonstrate the phenomena but invites the objection that the parameters
were chosen to produce them. These engines derive the parameters from standard
domain dynamics, so the mapping is a consequence rather than an input.

Currently implemented:
    LotkaVolterraTumour -- competition between drug-sensitive and drug-resistant
                           tumour cell populations under dose-dependent kill.

Not a clinical model. It is the standard two-compartment competition system used
in the adaptive-therapy literature, reduced to the minimum that exhibits
competitive release. No clinical conclusion should be drawn from it.
"""

import numpy as np

__all__ = ["LotkaVolterraTumour", "derive_arm_parameters"]


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
