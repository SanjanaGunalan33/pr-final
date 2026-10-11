"""Train, select, evaluate and export the Bayesian-Network RCA model.

    python train_model.py              # prints every stage in the terminal and saves all files
    python train_model.py --step       # same, but waits for Enter before each stage (good for presenting)
    python train_model.py --all-cpts   # also print every conditional probability table

Protocol (see README):
  1. Grouped, stratified 80/20 split (group = application|service|fault, so repeated runs of one
     scenario never straddle train and test).
  2. Feature-representation / hyper-parameter selection by grouped 5-fold CV INSIDE the training set.
  3. Parameter learning on the training set; posteriors for ALL training incidents (sanity check)
     and for the unseen test incidents (the real accuracy). The test set is scored exactly once.
The whole terminal output is also saved to outputs/run_log.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

import evaluate_model as ev                                  # noqa: E402
from preprocessing import load_data                          # noqa: E402
from rca_model import RCAModel, draw_network, EQUIVALENT_SAMPLE_SIZE   # noqa: E402

ROOT = Path(__file__).parent
SEED = 42            # train/test split seed
CV_SEED = 7          # seed for the inner grouped CV

FILE_GUIDE = [
    ("outputs/run_log.txt", "Copy of everything printed in the terminal during this run."),
    ("outputs/dataset_inventory.json", "Record counts, missing-data share, class counts, the 2 incidents without metrics."),
    ("outputs/split_info.json", "How the data was split: seed, method, train/test sizes, class coverage."),
    ("outputs/representation_comparison.csv", "Cross-validation score of every feature representation tried (training data only)."),
    ("outputs/network_structure.csv", "Every node: role (root / intermediate / leaf), parents, number of states."),
    ("outputs/network_edges.csv", "The 23 directed edges parent -> child of the fitted network."),
    ("outputs/bayesian_network.png", "Picture of the fitted network (roots blue, intermediate orange, leaves green)."),
    ("outputs/conditional_probability_tables.csv", "Every learned CPT in long form: node, state, parent combination, probability."),
    ("outputs/discretization_thresholds.csv", "Low/Medium/High cut-offs and 'Quiet' cut-off learned from training data."),
    ("outputs/train_posteriors.csv", "Posterior probability of every candidate service for every TRAINING incident."),
    ("outputs/test_posteriors.csv", "Same for the unseen TEST incidents, with the true label and a correct/incorrect flag."),
    ("outputs/test_predictions.csv", "Compact test result: true label, top-5 predictions and their probabilities."),
    ("outputs/evaluation_metrics.csv", "Top-1/3/5, balanced accuracy, macro-F1 per algorithm (BN, baselines, Extra Trees) + training check."),
    ("outputs/per_class_metrics.csv", "Precision / recall / F1 / support for each root-cause service (Bayesian Network, test set)."),
    ("outputs/confusion_matrix.csv", "True-vs-predicted counts for the Bayesian Network on the test set."),
    ("outputs/robustness_cv.csv", "Supplementary grouped 5-fold results of the selected configuration."),
    ("outputs/run_summary.json", "One-file summary of the headline numbers (read by the website)."),
    ("models/rca_bn.joblib", "The fitted network + preprocessing; the website loads this, it never retrains."),
    ("models/preprocessing_config.json", "Families, candidate services and state lists used by the model."),
]


class Log:
    """Print to the terminal immediately and keep a copy for outputs/run_log.txt."""

    def __init__(self, step=False):
        self.lines, self.step, self.n = [], step, 0

    def say(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.lines.append(s)

    def table(self, df, **kw):
        self.say(df.to_string(**kw))

    def section(self, title, why=""):
        self.n += 1
        if self.step and sys.stdin.isatty():
            input("\n[Press Enter to continue to the next stage] ")
        self.say("\n" + "=" * 100)
        self.say(f"STAGE {self.n}: {title}")
        if why:
            self.say(why)
        self.say("=" * 100)


def candidate_configs():
    cfgs = []
    for change in ("standardized", "relative"):
        cfgs.append(dict(representation="family_levels", change=change, quiet_quantile=0.5))
        for q in (0.1, 0.25, 0.5, 0.75):
            cfgs.append(dict(representation="localized_nofault", change=change, quiet_quantile=q))
            cfgs.append(dict(representation="localized", change=change, quiet_quantile=q))
        cfgs.append(dict(representation="individual", change=change, quiet_quantile=0.5, n_individual=40))
    return cfgs


def inner_cv(train, cfg, folds=5):
    res = []
    for a, b in StratifiedGroupKFold(folds, shuffle=True, random_state=CV_SEED).split(train, train.root_cause, train.group):
        m = RCAModel(**cfg).fit(train.iloc[a])
        v = train.iloc[b]
        rk = m.predict_frame(v)
        res.append((ev.topk_accuracy(v.root_cause, rk, 1), ev.topk_accuracy(v.root_cause, rk, 3)))
    return np.mean(res, axis=0)


def posterior_frame(df, rankings, classes):
    """Wide table: one row per incident, one column per candidate service (blank = not a service of that application)."""
    wide = pd.DataFrame([r.reindex(classes) for r in rankings]).round(6)
    wide.index = df.index
    head = pd.DataFrame({"incident_id": df.incident_id, "application": df.application, "true_root_cause": df.root_cause,
                         "true_fault_type": df.fault_type})
    out = pd.concat([head, wide], axis=1)
    out.insert(4, "predicted_root_cause", [r.index[0] for r in rankings])
    out.insert(5, "predicted_probability", [round(float(r.iloc[0]), 6) for r in rankings])
    out["correct"] = out.true_root_cause == out.predicted_root_cause
    return out


def show_posteriors(log, df, rankings, n=10):
    rows = []
    for (_, r), rk in zip(df.head(n).iterrows(), rankings[:n]):
        top = ", ".join(f"{k} {v:.1%}" for k, v in rk.head(3).items())
        rows.append({"incident": r.incident_id, "true root cause": r.root_cause, "top-3 posterior": top,
                     "ok": "yes" if rk.index[0] == r.root_cause else "NO"})
    log.table(pd.DataFrame(rows), index=False)


def main(data_dir=ROOT / "data", out_dir=ROOT / "outputs", model_dir=ROOT / "models", step=False, all_cpts=False, full_individual=False):
    out_dir, model_dir = Path(out_dir), Path(model_dir)
    out_dir.mkdir(exist_ok=True), model_dir.mkdir(exist_ok=True)
    log, t0 = Log(step), time.time()

    # ------------------------------------------------------------------ 1. data
    log.section("LOAD AND CHECK THE DATASET", "Joins the metadata file and the metric file on the incident id.")
    full, inventory = load_data(Path(data_dir) / "rcaeval_cases.csv", Path(data_dir) / "rcaeval_features.csv")
    df = full[full["has_metrics"]].reset_index(drop=True)
    inventory["records_used_for_modelling"] = int(len(df))
    (out_dir / "dataset_inventory.json").write_text(json.dumps(inventory, indent=2))
    log.say(f"Incidents in metadata file : {inventory['metadata_records']}   (the file has {inventory['metadata_records'] + 1} lines incl. header)")
    log.say(f"Incidents with metrics     : {inventory['feature_records']}   -> used for modelling: {len(df)}")
    log.say(f"Incidents WITHOUT metrics  : {inventory['records_without_metrics']}  (kept missing, not invented, not modelled)")
    log.say(f"Metric columns             : {inventory['numeric_metric_columns']}   share of empty cells: {inventory['missing_metric_cell_fraction']:.1%}")
    log.say(f"Applications               : {inventory['applications']}")
    log.say("Root-cause services (class counts):")
    log.table(pd.Series(inventory["root_cause_counts"], name="incidents").to_frame())

    # ------------------------------------------------------------------ 2. split
    log.section("SPLIT INTO TRAINING AND UNSEEN TEST DATA",
                "Repeated runs of the same scenario (application|service|fault) stay together, so test incidents are really unseen.")
    tr_idx, te_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, df.root_cause, df.group))
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    assert not (set(train.group) & set(test.group)), "scenario group leaked across the split"
    assert not (set(train.incident_id) & set(test.incident_id))
    split_info = {
        "method": "StratifiedGroupKFold(n_splits=5, shuffle=True), first fold used as the 20% test set",
        "group_definition": "application | root-cause service | fault type", "random_seed": SEED,
        "train_records": len(train), "test_records": len(test),
        "train_groups": int(train.group.nunique()), "test_groups": int(test.group.nunique()),
        "classes_total": int(df.root_cause.nunique()), "classes_in_train": int(train.root_cause.nunique()),
        "classes_in_test": int(test.root_cause.nunique()),
        "test_classes_missing_from_train": sorted(set(test.root_cause) - set(train.root_cause)),
        "test_records_per_application": test.application.value_counts().to_dict(),
    }
    (out_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))
    log.say(f"Training incidents: {len(train)} ({split_info['train_groups']} scenarios)   Test incidents: {len(test)} ({split_info['test_groups']} scenarios)")
    log.say(f"Classes in training: {split_info['classes_in_train']}/18, in test: {split_info['classes_in_test']}/18; seed {SEED}")

    # ------------------------------------------------------------------ 3. representation selection
    log.section("CHOOSE THE FEATURE REPRESENTATION (cross-validation on TRAINING data only)",
                "Each line = one candidate network design; it is trained and scored on held-out training folds.")
    rows = []
    if full_individual:
        cfg = dict(representation="individual", change="standardized", quiet_quantile=0.5, n_individual=None)
        comp = pd.DataFrame([cfg | {"selection": "all usable individual metric nodes; no top-K feature selection"}])
        comp.to_csv(out_dir / "representation_comparison.csv", index=False)
        log.say("FULL INDIVIDUAL MODE: retaining every usable standardized metric as its own node.")
    else:
        for cfg_try in candidate_configs():
            cv1, cv3 = inner_cv(train, cfg_try)
            rows.append({**cfg_try, "cv_top1": cv1, "cv_top3": cv3})
            log.say(f"  {cfg_try['representation']:<18} {cfg_try['change']:<13} quiet_q={cfg_try['quiet_quantile']:<4}  CV top-1={cv1:.3f}  top-3={cv3:.3f}")
        comp = pd.DataFrame(rows)
        comp["n_individual"] = comp.get("n_individual")
        best = comp.sort_values(["cv_top1", "cv_top3"], ascending=False).iloc[0]
        comp.to_csv(out_dir / "representation_comparison.csv", index=False)
        cfg = dict(representation=best.representation, change=best.change, quiet_quantile=float(best.quiet_quantile))
        if best.representation == "individual":
            cfg["n_individual"] = 40
        log.say(f"\nSELECTED: {cfg}   (CV top-1 {best.cv_top1:.3f})")

    # ------------------------------------------------------------------ 4. structure
    model = RCAModel(**cfg)
    log.section("BAYESIAN NETWORK STRUCTURE: root, intermediate and leaf nodes",
                "root = no parents (needs a PRIOR); intermediate/leaf = have parents (need a CONDITIONAL PROBABILITY TABLE).")
    model.fit(train)
    roles = model.node_roles()
    order = {"root": 0, "intermediate": 1, "leaf": 2}
    struct = pd.DataFrame([{"node": n, "role": roles[n], "parents": ", ".join(model.model.get_parents(n)) or "-",
                            "n_states": len(model.state_names[n]),
                            "cpt_shape (states x parent combos)": "x".join(map(str, model.model.get_cpds(n).get_values().shape))}
                           for n in model.nodes()]).sort_values(["role", "node"], key=lambda s: s.map(order) if s.name == "role" else s)
    struct.to_csv(out_dir / "network_structure.csv", index=False)
    log.table(struct, index=False)
    log.say(f"\nNodes: {len(model.nodes())}   Edges: {len(model.edges())}   DAG and all CPTs valid: {model.validate_cpds()}")
    log.say("Edges (parent -> child):")
    log.say("  " + "; ".join(f"{a}->{b}" for a, b in model.edges()))

    # ------------------------------------------------------------------ 5. parameter learning
    log.section("PARAMETER LEARNING (BDeu prior, equivalent sample size %d)" % EQUIVALENT_SAMPLE_SIZE,
                "Counts from the training incidents + a small smoothing prior => no zero probabilities.")
    log.say("PRIOR of the root node  P(Application):")
    log.table(model.cpd_frame("Application").round(4))
    for node in [n for n in model.nodes() if roles[n] == "intermediate"]:
        log.say(f"\nCPT of intermediate node {node}  (rows = parent states, each row sums to 1):")
        log.table(model.cpd_frame(node).round(3))
    leaves = [n for n in model.nodes() if roles[n] == "leaf"]
    if all_cpts:
        for node in leaves:
            log.say(f"\nCPT of leaf node {node}:")
            log.table(model.cpd_frame(node).round(3))
    else:
        node = "Loc_cpu" if "Loc_cpu" in leaves else leaves[0]
        log.say(f"\nExample leaf CPT {node}  P({node} | RootCause) - first 6 rows (use --all-cpts to print all):")
        log.table(model.cpd_frame(node).round(3).head(6))
        log.say(f"({len(leaves)} leaf nodes in total; all are saved in outputs/conditional_probability_tables.csv)")
    assert model.validate_cpds()

    # ------------------------------------------------------------------ 6. training posteriors
    log.section("POSTERIOR PROBABILITIES FOR THE TRAINING INCIDENTS (sanity check)",
                "Variable Elimination gives P(RootCause | Application, symptoms). Training accuracy is optimistic by nature.")
    tr_rank = model.predict_frame(train, top_k=None)
    classes = model.state_names["RootCause"]
    tr_post = posterior_frame(train, tr_rank, classes)
    tr_post.to_csv(out_dir / "train_posteriors.csv", index=False)
    show_posteriors(log, train, tr_rank, 10)
    tr_metrics, _ = ev.score_rankings(train.root_cause, tr_rank)
    log.say(f"\nTRAINING set ({len(train)} incidents): top-1 {tr_metrics['top1_accuracy']:.1%}  top-3 {tr_metrics['top3_accuracy']:.1%}")

    # ------------------------------------------------------------------ 7. worked example
    log.section("WORKED EXAMPLE: one unseen incident, calculated by hand and by Variable Elimination",
                "posterior(RC) is proportional to prior P(RC | Application) x product of P(symptom | RC).")
    demo = test.iloc[0]
    feats = model.builder.transform(test.iloc[[0]]).iloc[0].to_dict()
    tab, ev_used, diff = model.explain(demo.application, feats)
    log.say(f"Incident {demo.incident_id}: application = {demo.application}")
    log.say("Evidence given to the network: " + ", ".join(f"{k}={v}" for k, v in ev_used.items() if k != "Application"))
    log.say("(Level_* symptoms are not shown below: with the application known they depend only on the fault type, so they cancel out of the root-cause posterior.)")
    log.table(tab.head(6).T.round(4))
    log.say(f"\nMax difference between the hand calculation and pgmpy Variable Elimination: {diff:.2e}")
    log.say(f"Predicted: {tab.index[0]}   Actual: {demo.root_cause}")

    # ------------------------------------------------------------------ 8. unseen test
    log.section("POSTERIOR PROBABILITIES FOR UNSEEN TEST INCIDENTS (the real accuracy)",
                "These incidents and their scenarios were never used for learning or for choosing the design.")
    rankings = model.predict_frame(test, top_k=None)
    te_post = posterior_frame(test, rankings, classes)
    te_post.to_csv(out_dir / "test_posteriors.csv", index=False)
    show_posteriors(log, test, rankings, 15)
    bn_metrics, y_pred = ev.score_rankings(test.root_cause, rankings)
    if "FaultType" in model.state_names:
        fr = [model.posterior(r.application, model.builder.transform(test).iloc[i].to_dict(), "FaultType")[0]
              for i, (_, r) in enumerate(test.iterrows())]
        bn_metrics["fault_type_top1_accuracy"] = ev.topk_accuracy(test.fault_type, fr, 1)
    log.say("\nBAYESIAN NETWORK on unseen data:")
    for k, v in bn_metrics.items():
        log.say(f"  {k:<26} {v:.3f}")
    log.say(f"  correct: {int(te_post.correct.sum())} of {len(te_post)}")
    log.say("\nPer-application top-1 on unseen data:")
    log.table(te_post.groupby("application").correct.agg(correct="sum", incidents="size").assign(
        accuracy=lambda d: (d.correct / d.incidents).round(3)))
    pc = ev.per_class_table(test.root_cause, y_pred)
    log.say("\nPer-class precision / recall / F1:")
    log.table(pc.round(2), index=False)

    # ------------------------------------------------------------------ 9. comparison
    log.section("COMPARISON WITH BASELINES AND A BENCHMARK", "Each row is a different algorithm; only the first row is the Bayesian Network.")
    maj = ev.score_rankings(test.root_cause, ev.majority_rankings(train, test, False))[0]
    maj_app = ev.score_rankings(test.root_cause, ev.majority_rankings(train, test, True))[0]
    et_metrics = ev.score_rankings(test.root_cause, ev.extra_trees_rankings(train, test, model, cfg["change"], SEED))[0]
    algos = [("Bayesian Network (pgmpy, Variable Elimination)", bn_metrics),
             ("Baseline: global majority class", maj),
             ("Baseline: majority class per application", maj_app),
             ("Benchmark: Extra Trees (500 trees)", et_metrics),
             ("Training-set check: Bayesian Network", tr_metrics)]
    metrics = [{"algorithm": a, "metric": k, "value": v} for a, d in algos for k, v in d.items()]
    pd.DataFrame(metrics).to_csv(out_dir / "evaluation_metrics.csv", index=False)
    log.table(pd.DataFrame({a: d for a, d in algos}).T[["top1_accuracy", "top3_accuracy", "top5_accuracy", "balanced_accuracy", "macro_f1"]].round(3))
    pc.to_csv(out_dir / "per_class_metrics.csv", index=False)
    ev.confusion_table(test.root_cause, y_pred).to_csv(out_dir / "confusion_matrix.csv")
    pred = pd.DataFrame({"incident_id": test.incident_id, "application": test.application,
                         "true_root_cause": test.root_cause, "true_fault_type": test.fault_type})
    for k in range(5):
        pred[f"pred_top{k+1}"] = [r.index[k] if len(r) > k else "" for r in rankings]
        pred[f"prob_top{k+1}"] = [round(float(r.iloc[k]), 6) if len(r) > k else np.nan for r in rankings]
    pred["correct_top1"] = pred.true_root_cause == pred.pred_top1
    pred.to_csv(out_dir / "test_predictions.csv", index=False)

    # ------------------------------------------------------------------ 10. robustness
    log.section("ROBUSTNESS CHECK: grouped 5-fold on all incidents (informational)")
    rob = []
    for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, df.root_cause, df.group)):
        m = RCAModel(**cfg).fit(df.iloc[a])
        d, _ = ev.score_rankings(df.iloc[b].root_cause, m.predict_frame(df.iloc[b]))
        rob.append({"fold": k, "n_test": len(b), **d})
        log.say(f"  fold {k}: top-1 {d['top1_accuracy']:.3f}  top-3 {d['top3_accuracy']:.3f}")
    rob = pd.DataFrame(rob)
    rob.to_csv(out_dir / "robustness_cv.csv", index=False)

    # ------------------------------------------------------------------ 11. exports
    log.section("SAVE THE GRAPH, TABLES AND MODEL")
    n_nodes, n_edges = draw_network(model.model, out_dir / "bayesian_network.png")
    pd.DataFrame(model.edges(), columns=["parent", "child"]).to_csv(out_dir / "network_edges.csv", index=False)
    model.all_cpds_long().to_csv(out_dir / "conditional_probability_tables.csv", index=False)
    model.builder.summary_rows().to_csv(out_dir / "discretization_thresholds.csv", index=False)
    model.save(model_dir / "rca_bn.joblib")
    (model_dir / "preprocessing_config.json").write_text(json.dumps({
        **cfg, "families_used": model.builder.families, "family_coverage_in_train": model.builder.family_coverage,
        "candidate_root_causes": model.builder.candidates, "metric_columns_considered": len(model.builder.columns),
        "bdeu_equivalent_sample_size": EQUIVALENT_SAMPLE_SIZE, "state_names": model.state_names}, indent=2))
    summary = {"selected_config": cfg, "nodes": n_nodes, "edges": n_edges, "cpds_valid": model.validate_cpds(),
               "train_records": len(train), "test_records": len(test), "bayesian_network": bn_metrics,
               "training_set_check": tr_metrics, "majority_global": maj, "majority_per_application": maj_app,
               "extra_trees_benchmark": et_metrics,
               "robustness_cv_mean_top1": float(rob.top1_accuracy.mean()), "robustness_cv_std_top1": float(rob.top1_accuracy.std()),
               "target_70pct_top1_reached": bool(bn_metrics["top1_accuracy"] >= 0.70),
               "runtime_seconds": round(time.time() - t0, 1)}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.say(f"Graph drawn from the fitted model: {n_nodes} nodes, {n_edges} edges.")
    log.say("\nWHAT EACH OUTPUT FILE CONTAINS:")
    for f, d in FILE_GUIDE:
        log.say(f"  {f:<46} {d}")
    log.say(f"\nDONE in {summary['runtime_seconds']} s.  Start the website with:  python -m streamlit run app.py")
    (out_dir / "run_log.txt").write_text("\n".join(log.lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and evaluate the RCAEval Bayesian Network.")
    ap.add_argument("--step", action="store_true", help="pause for Enter before each stage")
    ap.add_argument("--all-cpts", action="store_true", help="print every CPT in the terminal")
    a = ap.parse_args()
    main(step=a.step, all_cpts=a.all_cpts)
