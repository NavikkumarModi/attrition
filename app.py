"""Streamlit app: pick a built-in scenario (or upload your own v/p/e arms),
run classical policies against it, and see the results as the same
self-contained HTML dashboard `render_dashboard()` already produces for
every other run in this repo -- nothing new was built to render results,
this app just wires existing pieces (scenarios.py, consumable.py,
dashboard.py) behind a form.

The classical-policy comparison costs nothing to run publicly. A second,
opt-in section runs a real LLM-driven population (Anthropic or Groq) using a
visitor-supplied API key -- entered in a password-masked field, held only in
this session's memory, never written to disk or logged, and never used for
anything but that visitor's own requests. The key still passes through this
app's own server process to make the call (that's how Streamlit works,
unlike a purely client-side page); it just never persists past the session
or leaves this process's memory.

Install:  pip install "attrition[web,llm,groq]"
Run:      streamlit run app.py
"""

import csv
import io
import tempfile

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from attrition import (CallableLLMClient, ConsumableBandit, Greedy, ECI, SECI,
                       Conservative, PHARMA_PERSONAS, Population,
                       simulate_population, run, compare, SCENARIOS,
                       get_arrays, render_dashboard)
from attrition.llm_policy import _CHOICE

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
LLM_MODELS = {"Anthropic": "claude-haiku-4-5-20251001",
             "Groq": "openai/gpt-oss-20b"}


def _build_llm_client(provider, api_key, model):
    if provider == "Anthropic":
        import anthropic
        sdk_client = anthropic.Anthropic(api_key=api_key)

        def call(system, user):
            msg = sdk_client.messages.create(
                model=model, max_tokens=300, system=system,
                messages=[{"role": "user", "content": user}])
            return msg.content[0].text
    else:
        import groq
        sdk_client = groq.Groq(api_key=api_key)

        def call(system, user):
            kwargs = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
            completion = sdk_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                max_tokens=300, **kwargs)
            return completion.choices[0].message.content
    return CallableLLMClient(call)


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

    st.divider()
    st.subheader("Real LLM population")
    st.caption("Bring your own key -- used only for this session's "
              "requests, never logged, never stored. Runs on this "
              "scenario's own v/p/e/delta.")
    provider = st.selectbox("Provider", ["None", "Anthropic", "Groq"],
                            label_visibility="collapsed")
    llm_run_clicked = False
    if provider != "None":
        api_key = st.text_input(f"{provider} API key", type="password",
                                placeholder="sk-... / gsk_...")
        model = st.text_input("Model", value=LLM_MODELS[provider])
        persona_names = st.multiselect("Personas", list(PHARMA_PERSONAS),
                                       default=list(PHARMA_PERSONAS)[:2])
        max_h = max(1, min(15, int(horizon)))
        llm_horizon = (st.slider("shared pull budget", 1, max_h, min(8, max_h))
                      if max_h > 1 else 1)
        st.caption(f"Up to **{llm_horizon} real API calls** to `{model}` "
                  f"(one shared pull budget across "
                  f"{len(persona_names) or 0} personas -- not "
                  f"{len(persona_names) or 0}x{llm_horizon}).")
        llm_run_clicked = st.button("Run real LLM population",
                                    use_container_width=True,
                                    disabled=not api_key or not persona_names)

if not run_clicked and not llm_run_clicked:
    with st.container(border=True):
        st.write(f"**{len(v)} arms** loaded from "
                f"{'the ' + scenario_name + ' scenario' if source == 'Built-in' else 'your CSV'}. "
                "Pick policies (or a real LLM population) in the sidebar.")
    st.stop()

env_factory = lambda seed: ConsumableBandit(v, p, e, delta=delta,
                                            horizon=horizon, seed=seed)

if run_clicked:
    if not chosen:
        st.warning("Pick at least one policy.")
        st.stop()

    policies = [POLICY_CLASSES[c]() for c in chosen]

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

if llm_run_clicked:
    st.subheader(f"Real {provider} population")
    personas = [PHARMA_PERSONAS[n] for n in persona_names]
    client = _build_llm_client(provider, api_key, model)
    population = Population.from_personas(personas, client=client)
    llm_env = ConsumableBandit(v, p, e, delta=delta, horizon=llm_horizon,
                               seed=dashboard_seed)

    with st.spinner(f"Running {len(personas)} real {provider}-driven "
                    f"personas ({model})..."):
        try:
            pop_result = simulate_population(llm_env, population)
        except Exception as ex:
            st.error(f"The API call failed before any decision was logged: {ex}")
            st.stop()

    cols = st.columns(2)
    cols[0].metric("System value", f"{pop_result['system_value']:.3f}")
    cols[1].metric("System regret", f"{pop_result['system_regret']:.3f}")

    log = population.members[persona_names[0]].log
    genuine = sum(1 for row in log if row["response"] and _CHOICE.search(row["response"]))
    st.caption(f"**{persona_names[0]}**'s log: {genuine}/{len(log)} calls produced "
              f"a genuine parseable decision from the model -- the rest fell "
              f"back to Greedy (e.g. a rate limit, or a response that didn't "
              f"parse). Report this number, not just the aggregate above.")
    errors = {}
    for row in log:
        if row.get("error"):
            errors[row["error"]] = errors.get(row["error"], 0) + 1
    if errors:
        with st.expander(f"{sum(errors.values())} call(s) raised an error"):
            for msg, count in errors.items():
                st.text(f"x{count}  {msg}")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        html_path = render_dashboard(pop_result["trace"], path=f.name,
                                     title=f"{title} -- real {provider} population")
    with open(html_path) as f:
        components.html(f.read(), height=760, scrolling=True)
