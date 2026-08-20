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

st.set_page_config(page_title="attrition", page_icon="⚗️", layout="wide")

# Fine-tuning the .streamlit/config.toml dark theme can't reach: card
# borders on custom containers, tighter vertical rhythm, monospace stat
# figures. Everything else (buttons, sliders, focus rings) already inherits
# the theme, so this stays small on purpose.
st.markdown("""
<style>
  .block-container { padding-top: 2.5rem; max-width: 1200px; }
  h1 { font-weight: 600 !important; letter-spacing: -.01em; }
  [data-testid="stMetricValue"] { font-size: 1.7rem; }
  [data-testid="stMetricLabel"] { text-transform: uppercase; font-size: .72rem;
    letter-spacing: .04em; opacity: .65; }
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
  section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.08); }
</style>
""", unsafe_allow_html=True)

st.title("attrition — consumable-bandit simulator")
st.caption("Bandits with consumable action sets: actions that destroy the "
          "arms they're taken on, with a permanent externality on "
          "everything that remains. **No-regret is not no-harm.**")

POLICY_CLASSES = {"Greedy": Greedy, "ECI": ECI, "SECI": SECI,
                  "Conservative": Conservative}

with st.sidebar:
    st.subheader("Scenario")
    source = st.radio("Source", ["Built-in", "Custom (upload CSV)"],
                      label_visibility="collapsed")

    if source == "Built-in":
        scenario_name = st.selectbox("Scenario", sorted(SCENARIOS))
        spec = SCENARIOS[scenario_name]
        with st.container(border=True):
            st.markdown(f"**{spec['phenomenon']}**")
            st.caption(spec["description"])
        with st.expander("Expected behaviour (from theory)"):
            st.write(spec["expected_behaviour"])
        v, p, e, kw = get_arrays(scenario_name)
        title = scenario_name
    else:
        st.caption("CSV with columns v,p,e — one row per arm.")
        upload = st.file_uploader("Upload", type="csv",
                                  label_visibility="collapsed")
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

    st.subheader("Parameters")
    delta = st.number_input("delta (externality scale)",
                            value=float(kw.get("delta", 0.05)),
                            min_value=0.0, format="%.4f")
    horizon = st.number_input("horizon", value=int(kw.get("horizon", 50)),
                              min_value=1)
    n_seeds = st.slider("seeds (aggregate comparison)", 1, 200, 20)
    dashboard_seed = st.number_input("seed (detailed dashboard)", value=0,
                                     min_value=0)

    st.subheader("Policies")
    chosen = st.multiselect("Compare", list(POLICY_CLASSES),
                            default=["Greedy", "ECI"],
                            label_visibility="collapsed")

    run_clicked = st.button("Run simulation", type="primary",
                            use_container_width=True)

if not run_clicked:
    with st.container(border=True):
        st.write(f"**{len(v)} arms** loaded from "
                f"{'the ' + scenario_name + ' scenario' if source == 'Built-in' else 'your CSV'}. "
                "Pick policies in the sidebar and click **Run simulation**.")
    st.stop()

if not chosen:
    st.warning("Pick at least one policy.")
    st.stop()

policies = [POLICY_CLASSES[c]() for c in chosen]
env_factory = lambda seed: ConsumableBandit(v, p, e, delta=delta,
                                            horizon=horizon, seed=seed)

st.subheader("Aggregate comparison")
st.caption(f"Mean over {n_seeds} seeds")
with st.spinner(f"Running {len(policies)} polic{'y' if len(policies)==1 else 'ies'} "
                f"x {n_seeds} seeds..."):
    result = compare(env_factory, policies, seeds=n_seeds)
cols = st.columns(len(result))
for col, (name, stats) in zip(cols, result.items()):
    with col:
        with st.container(border=True):
            st.metric(name, f"{stats['value']:.3f}",
                     f"-{stats['regret']:.3f} regret" if stats['regret'] > 0
                     else "0.000 regret")

st.subheader("Detailed trace")
st.caption(f"Single run, seed={dashboard_seed}")
combined_trace = []
for pol in policies:
    r = run(env_factory(dashboard_seed), pol, log=True)
    for row in r["log"]:
        row["agent"] = pol.name
        combined_trace.append(row)

with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
    html_path = render_dashboard(combined_trace, path=f.name, title=title)
with open(html_path) as f:
    components.html(f.read(), height=760, scrolling=True)
