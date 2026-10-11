"""Single-page Streamlit interface for the RCAEval Bayesian Network."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT, MODELS = ROOT / "outputs", ROOT / "models"
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Root Cause Assistant", layout="wide")

SYMPTOMS = {
    "CPU usage anomaly": "cpu", "Memory anomaly": "mem", "Disk I/O anomaly": "diskio",
    "Network/socket anomaly": "socket", "Traffic/workload anomaly": "workload",
    "Slow response / latency": "latency", "Error-rate anomaly": "error",
}
UNKNOWN = "Don't know / not measured"
QUIET = "No clear anomaly"
OTHER = "A different service seems affected"

@st.cache_resource
def load_model():
    from rca_model import RCAModel
    return RCAModel.load(MODELS / "rca_bn.joblib")

st.markdown("# Root Cause Assistant")
st.markdown("Select the symptoms you observe. For each selected symptom, indicate what appears affected. The Bayesian Network uses Variable Elimination to rank five likely root-cause services.")
st.caption("Research prototype trained on RCAEval. Results are probabilistic hypotheses, not guaranteed diagnoses.")

model_path = MODELS / "rca_bn.joblib"
summary_path = OUT / "run_summary.json"
if not model_path.exists():
    st.error("The trained model is missing. In the project folder, run `python train_model.py`, then restart Streamlit.")
    st.stop()
try:
    model = load_model()
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

with st.container(border=True):
    st.subheader("1. Choose observed symptoms")
    selected = st.multiselect("Symptoms present", list(SYMPTOMS), placeholder="Choose one or more symptoms…", help="You can select any number of symptoms.")
    services = sorted(model.builder.candidates)
    symptoms = {}
    if selected:
        st.caption("For each selected symptom, choose the service that appears most affected, or mark it unknown.")
        cols = st.columns(2)
        for i, label in enumerate(selected):
            fam = SYMPTOMS[label]
            node = f"Loc_{fam}"
            if node not in model.state_names:
                st.warning(f"The trained model does not contain the '{fam}' symptom family. Retrain with a representation that includes it.")
                continue
            choices = [UNKNOWN, QUIET, OTHER] + services
            with cols[i % 2]:
                choice = st.selectbox(label, choices, key=f"symptom_{fam}")
            value = "Unknown" if choice == UNKNOWN else "Quiet" if choice == QUIET else "Other" if choice == OTHER else choice
            symptoms[node] = value

    st.subheader("2. Optional symptom severity")
    with st.expander("Add severity information (optional)"):
        levels = {}
        for label in selected:
            fam = SYMPTOMS[label]
            node = f"Level_{fam}"
            if node in model.state_names:
                level = st.selectbox(f"{label} severity", ["Unknown", "Low", "Medium", "High"], key=f"level_{fam}")
                levels[node] = level
        symptoms.update(levels)

    run = st.button("Find top 5 root causes", type="primary", use_container_width=True)

if run:
    if not selected:
        st.warning("Select at least one symptom before requesting a prediction.")
    elif not any(v not in ("Unknown",) for v in symptoms.values()):
        st.warning("Choose at least one known symptom observation to make an evidence-based prediction.")
    else:
        with st.spinner("Calculating posterior probabilities…"):
            result = model.predict_incident("(unknown application)", symptoms, top_k=5)
        ranking = result.get("ranking", [])
        if not ranking:
            st.error("No root-cause candidates could be produced. Check the model artifacts and selected evidence.")
        else:
            st.markdown("## Top 5 likely root causes")
            st.caption("Ranked by posterior probability given the observations above.")
            df = pd.DataFrame(ranking, columns=["Root-cause candidate", "Posterior probability"])
            df["Posterior probability"] = df["Posterior probability"].clip(lower=0)
            total = df["Posterior probability"].sum()
            if total > 0:
                df["Posterior probability"] /= total
            for i, row in df.iterrows():
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{i+1}. {row['Root-cause candidate']}**")
                c2.markdown(f"**{row['Posterior probability']:.1%}**")
                st.progress(float(row["Posterior probability"]))
            with st.expander("View results as a table"):
                st.dataframe(df, hide_index=True, use_container_width=True)
            st.info("Prediction uses symptom evidence only; no application or incident selector is required.")
            for note in result.get("notes", []):
                st.caption(note)
            st.download_button("Download these predictions (CSV)", df.to_csv(index=False).encode("utf-8"), "root_cause_predictions.csv", "text/csv")
            st.caption("The candidate list is learned from RCAEval; this tool does not guarantee correct predictions for every possible real-world system or fault.")

with st.expander("Model information"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Bayesian Network nodes", len(model.nodes()))
    c2.metric("Directed edges", len(model.edges()))
    c3.metric("Training incidents", summary.get("train_records", getattr(model, "n_train", "—")))
    st.write("**Inference algorithm:** Variable Elimination")
    st.write("**Parameter learning:** Bayesian Estimator with BDeu smoothing")
    st.write("**Graph and CPTs:** generated from the fitted network and saved under `outputs/`.")
    graph = OUT / "bayesian_network.png"
    if graph.exists():
        st.image(str(graph), caption="Actual fitted Bayesian Network", use_container_width=True)
        st.download_button("Download Bayesian Network graph", graph.read_bytes(), "bayesian_network.png", "image/png")
    if summary.get("bayesian_network"):
        bn = summary["bayesian_network"]
        st.write("**Held-out evaluation (Bayesian Network):**")
        st.write(f"Top-1: {bn.get('top1_accuracy', float('nan')):.1%} · Top-3: {bn.get('top3_accuracy', float('nan')):.1%} · Macro-F1: {bn.get('macro_f1', float('nan')):.3f}")
        st.caption("These scores describe the evaluation protocol recorded during training; they may not represent symptom-only use with an unknown application.")
