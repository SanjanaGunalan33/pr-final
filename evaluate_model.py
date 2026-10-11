"""Metrics, baselines and the Extra-Trees benchmark. Each result is labelled by algorithm."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from preprocessing import benchmark_matrix, CHANGE_SUFFIX


def topk_accuracy(y_true, rankings, k):
    return float(np.mean([t in list(r.index[:k]) for t, r in zip(y_true, rankings)]))


def score_rankings(y_true, rankings):
    y_true = list(y_true)
    y_pred = [r.index[0] for r in rankings]
    return {
        "top1_accuracy": topk_accuracy(y_true, rankings, 1),
        "top3_accuracy": topk_accuracy(y_true, rankings, 3),
        "top5_accuracy": topk_accuracy(y_true, rankings, 5),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }, y_pred


def per_class_table(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"root_cause": labels, "precision": p, "recall": r, "f1": f, "support": s})


def confusion_table(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])


def majority_rankings(train_df, test_df, per_application):
    """Always predict the most frequent training class (optionally within the application)."""
    overall = train_df["root_cause"].value_counts()
    out = []
    for _, r in test_df.iterrows():
        if per_application:
            c = train_df.loc[train_df["application"] == r["application"], "root_cause"].value_counts()
            c = c if len(c) else overall
        else:
            c = overall
        out.append(c.astype(float))
    return out


def extra_trees_rankings(train_df, test_df, bn_model, change="standardized", seed=42):
    """Benchmark: Extra Trees on log-scaled raw metrics + application. NOT the Bayesian Network."""
    suffix = CHANGE_SUFFIX[change]
    cols = [c for c in train_df.columns if c.endswith(suffix)]
    apps = sorted(train_df["application"].unique())

    def mat(d):
        a = np.stack([(d["application"] == x).to_numpy(float) for x in apps], axis=1)
        return np.hstack([benchmark_matrix(d, cols), a])
    et = ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    et.fit(mat(train_df), train_df["root_cause"])
    proba = et.predict_proba(mat(test_df))
    out = []
    for p, (_, r) in zip(proba, test_df.iterrows()):
        s = pd.Series(p, index=et.classes_)
        s = s[s.index.isin(bn_model.app_services.get(r["application"], list(s.index)))]  # same app restriction as the BN
        s = s / s.sum() if s.sum() > 0 else s
        out.append(s.sort_values(ascending=False))
    return out
