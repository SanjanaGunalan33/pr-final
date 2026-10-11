"""Automated tests. Run from the project root:  python -m pytest tests -q"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing import FeatureBuilder, load_data      # noqa: E402
from rca_model import RCAModel                           # noqa: E402

DATA = ROOT / "data"
pytestmark = pytest.mark.skipif(not (DATA / "rcaeval_cases.csv").exists(), reason="dataset files missing")


@pytest.fixture(scope="module")
def loaded():
    return load_data(DATA / "rcaeval_cases.csv", DATA / "rcaeval_features.csv")


@pytest.fixture(scope="module")
def split(loaded):
    df = loaded[0]
    df = df[df.has_metrics].reset_index(drop=True)
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, df.root_cause, df.group))
    return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)


@pytest.fixture(scope="module")
def model(split):
    return RCAModel("localized", "relative", 0.1).fit(split[0])


def test_join_integrity(loaded):
    df, inv = loaded
    assert inv["joined_records"] == inv["metadata_records"] == len(df)
    assert df.incident_id.is_unique
    assert inv["label_mismatches_between_files"] == 0


def test_missing_metric_records_kept_as_missing(loaded):
    df, inv = loaded
    miss = df[~df.has_metrics]
    assert len(miss) == len(inv["records_without_metrics"]) == inv["metadata_records"] - inv["feature_records"]
    metric_cols = [c for c in df.columns if c.endswith("__relative_change")]
    assert miss[metric_cols].isna().all().all()          # nothing was invented


def test_no_label_or_id_columns_used_as_features(model):
    used = set(model.builder.columns)
    assert all("__" in c for c in used)                  # only metric columns
    for forbidden in ("incident_id", "root_cause", "fault_type", "group", "Dataset"):
        assert forbidden not in used
    assert set(model.state_names) - set(model.builder.feature_nodes) == {"Application", "RootCause", "FaultType"}


def test_preprocessing_fitted_on_train_only(split):
    train, test = split
    assert not set(train.group) & set(test.group)
    assert not set(train.incident_id) & set(test.incident_id)
    b1 = FeatureBuilder("localized", "relative", 0.1).fit(train, train.root_cause)
    b2 = FeatureBuilder("localized", "relative", 0.1).fit(train, train.root_cause)
    assert b1.thresholds == b2.thresholds and b1.quiet_thr == b2.quiet_thr
    before = b1.transform(test)
    changed = test.copy()                                # transform must not refit on new data
    mcols = [c for c in changed.columns if c.endswith("__relative_change")]
    changed[mcols] = changed[mcols] * 1000.0
    b1.transform(changed)
    assert b1.thresholds == b2.thresholds and b1.quiet_thr == b2.quiet_thr
    assert before.equals(b1.transform(test))


def test_dag_and_cpt_normalisation(model):
    assert nx.is_directed_acyclic_graph(nx.DiGraph(model.model.edges()))
    assert model.validate_cpds()
    for cpd in model.model.get_cpds():
        assert np.allclose(cpd.get_values().sum(axis=0), 1.0)
        assert (cpd.get_values() > 0).all()               # BDeu smoothing -> no zero probabilities
    long = model.all_cpds_long()
    assert np.allclose(long.groupby(["node", "parents"]).probability.sum(), 1.0)


def test_inference_topk_and_normalisation(model, split):
    test = split[1]
    feats = model.builder.transform(test.head(1)).iloc[0].to_dict()
    out = model.predict_incident(test.iloc[0].application, feats)
    probs = [p for _, p in out["ranking"]]
    assert len(out["ranking"]) == 5 and probs == sorted(probs, reverse=True)
    post, _, _ = model.posterior(test.iloc[0].application, feats)
    assert abs(post.sum() - 1) < 1e-9
    assert set(post.index) <= set(model.app_services[test.iloc[0].application])   # only services of that app


def test_unknown_symptoms_and_no_evidence(model):
    app = "Sock Shop"
    out = model.predict_incident(app, {n: "Unknown" for n in model.builder.feature_nodes})
    assert abs(model.posterior(app, {})[0].sum() - 1) < 1e-9
    assert len(out["ranking"]) == 5 and out["evidence_used"] == {"Application": app}


def test_unsupported_application_and_state(model):
    out = model.predict_incident("Quantum Shop", {"Level_cpu": "Gigantic", "NotANode": "High"})
    assert len(out["ranking"]) == 5
    assert abs(model.posterior("Quantum Shop", {})[0].sum() - 1) < 1e-9
    assert any("not seen in training" in n for n in out["notes"]) and len(out["notes"]) >= 3


def test_hand_calculation_matches_variable_elimination(model, split):
    test = split[1]
    for i in (0, 40, 100):
        feats = model.builder.transform(test.iloc[[i]]).iloc[0].to_dict()
        tab, ev, diff = model.explain(test.iloc[i].application, feats)
        assert diff < 1e-9 and abs(tab["posterior (hand)"].sum() - 1) < 1e-9


def test_node_roles_and_root_prior(model):
    roles = model.node_roles()
    assert roles["Application"] == "root" and roles["RootCause"] == "intermediate"
    assert all(roles[n] == "leaf" for n in model.builder.feature_nodes)
    prior = model.cpd_frame("Application")
    assert prior.shape[0] == 1 and abs(prior.sum(axis=1).iloc[0] - 1) < 1e-9
    for n in model.nodes():                               # every node's table is displayable and row-normalised
        assert np.allclose(model.cpd_frame(n).sum(axis=1), 1.0)


def test_full_posterior_rows_sum_to_one(model, split):
    rows = model.predict_frame(split[1].head(10), top_k=None)
    assert all(abs(r.sum() - 1) < 1e-9 for r in rows)


def test_reproducible_predictions(split):
    train, test = split
    a = RCAModel("localized", "relative", 0.1).fit(train).predict_frame(test.head(15))
    b = RCAModel("localized", "relative", 0.1).fit(train).predict_frame(test.head(15))
    for x, y in zip(a, b):
        assert list(x.index) == list(y.index) and np.allclose(x.values, y.values)


def test_save_and_load_roundtrip(model, split, tmp_path):
    model.save(tmp_path / "m.joblib")
    again = RCAModel.load(tmp_path / "m.joblib")
    test = split[1].head(8)
    for x, y in zip(model.predict_frame(test), again.predict_frame(test)):
        assert list(x.index) == list(y.index)


def test_missing_dataset_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="rcaeval_cases.csv"):
        load_data(tmp_path / "rcaeval_cases.csv", tmp_path / "rcaeval_features.csv")


def test_saved_outputs_consistent():
    out = ROOT / "outputs"
    if not (out / "test_predictions.csv").exists():
        pytest.skip("run train_model.py first")
    pred = pd.read_csv(out / "test_predictions.csv")
    m = pd.read_csv(out / "evaluation_metrics.csv")
    bn = m[m.algorithm.str.startswith("Bayesian") & (m.metric == "top1_accuracy")].value.iloc[0]
    assert abs(pred.correct_top1.mean() - bn) < 1e-9
    edges = pd.read_csv(out / "network_edges.csv")
    mdl = RCAModel.load(ROOT / "models" / "rca_bn.joblib")
    assert set(map(tuple, edges.values)) == set(mdl.model.edges())
