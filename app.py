"""Streamlit MVP: pick a built-in scenario (or upload your own v/p/e arms),
run classical policies against it, and see the results as the same
self-contained HTML dashboard `render_dashboard()` already produces for
every other run in this repo -- nothing new was built to render results,
this app just wires existing pieces (scenarios.py, consumable.py,
dashboard.py) behind a form.

Deliberately classical-policy-only for this first version: no LLM calls, so
a publicly hosted deployment has no API-cost or abuse surface. LLM-driven
population runs (see examples/04, 08, 09) are a natural v2 once this proves
useful, gated behind the visitor supplying their own API key client-side --
never stored server-side.

Install:  pip install "attrition[web]"
Run:      streamlit run app.py
"""

import csv
import io
import tempfile

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from attrition import (ConsumableBandit, Greedy, ECI, SECI, Conservative,
                       run, compare, SCENARIOS, get_arrays, render_dashboard)

st.set_page_config(page_title="attrition", layout="wide")
st.title("attrition — consumable-bandit simulator")
st.caption("Bandits with consumable action sets: actions that destroy the "
          "arms they're taken on, with a permanent externality on "
          "everything that remains. No-regret is not no-harm.")

POLICY_CLASSES = {"Greedy": Greedy, "ECI": ECI, "SECI": SECI,
                  "Conservative": Conservative}

with st.sidebar:
    st.header("Scenario")
    source = st.radio("Source", ["Built-in", "Custom (upload CSV)"])

    if source == "Built-in":
        scenario_name = st.selectbox("Scenario", sorted(SCENARIOS))
        spec = SCENARIOS[scenario_name]
        st.info(f"**{spec['phenomenon']}**\n\n{spec['description']}")
        with st.expander("Expected behaviour (from theory)"):
            st.write(spec["expected_behaviour"])
        v, p, e, kw = get_arrays(scenario_name)
        title = scenario_name
    else:
        st.caption("CSV with columns v,p,e — one row per arm.")
        upload = st.file_uploader("Upload", type="csv")
        if upload is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        reader = csv.DictReader(io.StringIO(upload.getvalue().decode()))
        rows = list(reader)
        if not rows or not {"v", "p", "e"} <= set(rows[0]):
            st.error("CSV needs columns v, p, e.")
            st.stop()
        v = np.array([float(r["v"]) for r in rows])
        p = np.array([float(r["p"]) for r in rows])
        e = np.array([float(r["e"]) for r in rows])
        kw = {"delta": 0.05, "horizon": 50}
        title = "custom"

    st.header("Parameters")
    delta = st.number_input("delta (externality scale)",
                            value=float(kw.get("delta", 0.05)),
                            min_value=0.0, format="%.4f")
    horizon = st.number_input("horizon", value=int(kw.get("horizon", 50)),
                              min_value=1)
    n_seeds = st.slider("seeds (aggregate comparison)", 1, 200, 20)
    dashboard_seed = st.number_input("seed (detailed dashboard)", value=0,
                                     min_value=0)

    st.header("Policies")
    chosen = st.multiselect("Compare", list(POLICY_CLASSES),
                            default=["Greedy", "ECI"])

    run_clicked = st.button("Run simulation", type="primary")

if not run_clicked:
    st.write(f"{len(v)} arms loaded. Pick policies in the sidebar and click "
            "**Run simulation**.")
    st.stop()

if not chosen:
    st.warning("Pick at least one policy.")
    st.stop()

policies = [POLICY_CLASSES[c]() for c in chosen]
env_factory = lambda seed: ConsumableBandit(v, p, e, delta=delta,
                                            horizon=horizon, seed=seed)

st.subheader("Aggregate comparison")
with st.spinner(f"Running {len(policies)} polic{'y' if len(policies)==1 else 'ies'} "
                f"x {n_seeds} seeds..."):
    result = compare(env_factory, policies, seeds=n_seeds)
cols = st.columns(len(result))
for col, (name, stats) in zip(cols, result.items()):
    col.metric(name, f"value {stats['value']:.3f}",
              f"regret {stats['regret']:.3f}", delta_color="inverse")

st.subheader(f"Detailed trace (seed={dashboard_seed})")
combined_trace = []
for pol in policies:
    r = run(env_factory(dashboard_seed), pol, log=True)
    for row in r["log"]:
        row["agent"] = pol.name
        combined_trace.append(row)

with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
    html_path = render_dashboard(combined_trace, path=f.name, title=title)
with open(html_path) as f:
    components.html(f.read(), height=900, scrolling=True)
