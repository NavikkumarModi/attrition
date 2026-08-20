"""Domain instantiations of the consumable-action-set model.

Each domain supplies a mapping from real quantities to the model primitives
(v, p, e, delta). The mappings differ in how well they fit, and the docstrings
say so explicitly rather than asserting uniform applicability.

    agent_tools          strong fit   -- shared quota/rate-limit infrastructure
    adaptive_therapy     strong fit   -- competitive release is literally this
    platform_trial       moderate fit -- externality is statistical, not physical
    design_space         partial fit  -- has an extra terminal-commitment layer
                                         the base model does not capture

Every instantiation is a real parameterisation that runs, not an illustration.
"""

import numpy as np

from .consumable import ConsumableBandit

__all__ = ["agent_tools", "adaptive_therapy", "platform_trial",
           "design_space", "DOMAIN_NOTES"]


DOMAIN_NOTES = {
    "agent_tools": (
        "Strong fit, and increasingly the live case rather than a "
        "hypothetical one: an autonomous agent -- a coding agent, a "
        "browser/computer-use agent, anything doing multi-step tool calls "
        "without a human approving each one -- can look flawless on every "
        "individual call while quietly burning a shared resource (an API "
        "quota, a cloud credit pool, a service account's standing) that "
        "every other agent and the human operator both depend on. Arms are "
        "API-backed tools; v is task success rate; p is the chance a call "
        "trips a rate limit, exhausts a quota, or gets a key revoked; e is "
        "how much shared upstream infrastructure the tool sits on, so "
        "burning it throttles the rest. "
        "Real supporting evidence, not used to set the numbers below: "
        "Postman's 2025 State of the API Report (89% of developers use "
        "generative AI daily; 7.53M calls to AI APIs in the past 12 months, "
        "+40% YoY) found unauthorized agent access is developers' TOP-cited "
        "security risk (51%), with 46% specifically worried about agents "
        "leaking or sharing API credentials -- both exactly the shared-"
        "quota failure mode this domain models, not a related-but-different "
        "one. MCP (the protocol that lets an agent call tools like these in "
        "the first place) is already used regularly by 10% of developers "
        "and explored by another 24%, dating this as a current, not "
        "speculative, mechanism. Datadog's 'State of AI Engineering' report "
        "separately measured 2-5% of production LLM API call spans erroring "
        "in Feb-Mar 2026, the majority from exceeded rate limits (8.4M "
        "rate-limit errors observed in March 2026 alone). Neither figure is "
        "used to recalibrate p/e here: both are per-call/opinion rates, a "
        "different quantity from this roster's p (probability a pull "
        "exhausts/revokes the tool for the rest of the session); "
        "conflating the two would be a fake precision upgrade."),
    "adaptive_therapy": (
        "Strong fit. Arms are dose levels. Pulling a high dose irreversibly "
        "destroys drug-sensitive tumour cells (p rises with dose intensity), "
        "and losing that population releases resistant cells which degrade the "
        "efficacy of every subsequent treatment (e rises with dose intensity). "
        "The model reproduces competitive release from first principles: the "
        "maximum-tolerated-dose policy is exactly greedy, and it loses. "
        "Caveat: real tumour dynamics are a population process, not additive "
        "burden from dead arms; this is a reduced form, not a clinical model."),
    "platform_trial": (
        "Moderate fit. Arms are treatment arms; v is expected treatment "
        "benefit; p is the chance an arm is dropped for futility on evidence "
        "the allocation itself generated; e is the arm's reliance on shared "
        "control data. Dropping arms mid-trial creates non-concurrent-control "
        "problems that degrade the statistical power of every remaining "
        "comparison. Caveat: the externality here is inferential rather than "
        "physical, and 'value' mixes patient benefit with information gain."),
    "design_space": (
        "Partial fit. Arms are process parameter settings; p is the chance a "
        "run consumes its scarce API material or produces an out-of-spec "
        "batch; e is how much a failure at that setting narrows the design "
        "space that can subsequently be defended in a filing. Caveat: the real "
        "problem carries a terminal-commitment layer -- a one-shot declaration "
        "of design-space width that fixes the feasible set forever -- which "
        "the base model does not capture. Included as a partial instantiation "
        "and an open direction."),
}


# ----------------------------------------------------------------- agents
def agent_tools(seed=0, horizon=40, delta=0.06, n_tools=None):
    """Orchestrator routing across API-backed tools.

    Default roster spans premium tools on dedicated endpoints (low p, low e)
    and bulk tools on shared quota pools (high p, high e).
    """
    roster = [
        # name,                quality, fragility, shared-infra
        ("premium-search",        1.15,      0.15,   0.2),
        ("bulk-search",           1.05,      0.75,   2.4),
        ("premium-codegen",       1.00,      0.10,   0.1),
        ("bulk-scraper",          0.98,      0.85,   2.8),
        ("internal-db",           0.80,      0.05,   0.0),
        ("cached-lookup",         0.62,      0.02,   0.0),
    ]
    if n_tools:
        roster = roster[:n_tools]
    v = np.array([r[1] for r in roster])
    p = np.array([r[2] for r in roster])
    e = np.array([r[3] for r in roster])
    env = ConsumableBandit(v, p, e, delta=delta, horizon=horizon,
                           sigma=0.05, seed=seed)
    env.labels = [r[0] for r in roster]
    return env


# ----------------------------------------------------------------- oncology
def adaptive_therapy(seed=0, horizon=40, delta=0.05, n_doses=6):
    """Adaptive cancer therapy under competitive release.

    Dose intensity indexes the arms. Higher intensity gives greater immediate
    tumour control (v), but is both more likely to exhaust the drug-sensitive
    population (p) and more damaging when it does, because that population was
    suppressing the resistant clone (e).

    The key structural feature: v, p and e all increase together with dose
    intensity, so kappa = p*e is highly dispersed -- exactly the regime where
    Theorem 1 says greedy fails. Maximum tolerated dose IS greedy.
    """
    intensity = np.linspace(0.2, 1.0, n_doses)
    v = 0.45 + 0.75 * intensity              # immediate tumour control
    p = 0.05 + 0.85 * intensity ** 1.4       # chance of exhausting sensitives
    e = 0.10 + 2.60 * intensity ** 1.6       # released resistant burden
    env = ConsumableBandit(v, p, e, delta=delta, horizon=horizon,
                           sigma=0.06, seed=seed)
    env.labels = [f"dose-{x:.2f}" for x in intensity]
    return env


# ----------------------------------------------------------------- trials
def platform_trial(seed=0, horizon=50, delta=0.04, n_arms=7):
    """Platform trial with arms entering and leaving, sharing a control.

    Arms differ in expected benefit (v) and in maturity: early-stage arms are
    more likely to be dropped for futility (high p) and lean more heavily on
    shared control data, so their removal does more damage to the concurrency
    of remaining comparisons (high e).
    """
    rng = np.random.default_rng(seed)
    v = np.sort(rng.uniform(0.45, 1.10, n_arms))[::-1].copy()
    maturity = rng.uniform(0.0, 1.0, n_arms)          # 1 = mature
    p = 0.60 - 0.50 * maturity                        # mature arms survive
    e = 2.20 - 1.90 * maturity                        # mature arms lean less
    env = ConsumableBandit(v, np.clip(p, 0.05, 1.0), np.clip(e, 0.0, None),
                           delta=delta, horizon=horizon, sigma=0.07, seed=seed)
    env.labels = [f"arm-{i}" for i in range(n_arms)]
    return env


# ----------------------------------------------------------------- CMC
def design_space(seed=0, horizon=30, delta=0.05, n_settings=8):
    """CMC process development over candidate parameter settings.

    Aggressive settings give higher yield (v) but are more likely to consume
    their scarce material without producing usable data or to produce an
    out-of-spec batch (p), and a failure at an aggressive setting narrows the
    defensible design space more (e).
    """
    rng = np.random.default_rng(seed)
    aggression = np.sort(rng.uniform(0.1, 1.0, n_settings))
    v = 0.55 + 0.60 * aggression
    p = 0.08 + 0.70 * aggression ** 1.3
    e = 0.15 + 2.20 * aggression ** 1.5
    env = ConsumableBandit(v, p, e, delta=delta, horizon=horizon,
                           sigma=0.05, seed=seed)
    env.labels = [f"setting-{i}" for i in range(n_settings)]
    return env
