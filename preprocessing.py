"""Data loading, validation and leakage-safe discretisation for the RCAEval project.

Everything that learns something from data (family discovery, thresholds, candidate
services, feature selection) lives in FeatureBuilder.fit() and is fitted on the
training rows only. FeatureBuilder.transform() just applies the stored values.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

UNKNOWN, QUIET, OTHER = "Unknown", "Quiet", "Other"
LEVELS = ["Low", "Medium", "High"]
CHANGE_SUFFIX = {"standardized": "__standardized_change", "relative": "__relative_change"}

# Columns of the metadata file that are used (labels + application only).
CASE_REQUIRED = ["Incident ID", "Application", "Known Root Cause Service", "Fault Type"]
# Columns of the feature file that duplicate labels -> never used as predictors.
FEATURE_LABEL_COLS = ["case", "application", "root_cause_service", "fault"]


# ----------------------------------------------------------------------------- loading
def load_data(cases_path, features_path):
    """Load, validate and join the two CSV files. Returns (merged_df, inventory_dict)."""
    cases_path, features_path = Path(cases_path), Path(features_path)
    missing = [str(p) for p in (cases_path, features_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required dataset file(s) not found: " + ", ".join(missing)
            + ". Place rcaeval_cases.csv and rcaeval_features.csv in the data/ folder."
        )
    cases = pd.read_csv(cases_path)
    feats = pd.read_csv(features_path)

    for col in CASE_REQUIRED:
        if col not in cases.columns:
            raise ValueError(f"Metadata file is missing required column '{col}'.")
    for col in FEATURE_LABEL_COLS:
        if col not in feats.columns:
            raise ValueError(f"Feature file is missing required column '{col}'.")
    if cases["Incident ID"].duplicated().any():
        raise ValueError("Duplicate Incident IDs in the metadata file.")
    if feats["case"].duplicated().any():
        raise ValueError("Duplicate case IDs in the feature file.")
    extra = set(feats["case"]) - set(cases["Incident ID"])
    if extra:
        raise ValueError(f"{len(extra)} feature records have no metadata record, e.g. {sorted(extra)[:3]}")

    # Consistency of the label columns duplicated in both files (check only, not used later).
    chk = cases.merge(feats[FEATURE_LABEL_COLS], left_on="Incident ID", right_on="case", how="inner")
    label_mismatch = int(
        (chk["Application"] != chk["application"]).sum()
        + (chk["Known Root Cause Service"] != chk["root_cause_service"]).sum()
        + (chk["Fault Type"] != chk["fault"]).sum()
    )

    metric_cols = [c for c in feats.columns if c not in FEATURE_LABEL_COLS]
    metrics = feats[["case"] + metric_cols]
    merged = cases.merge(metrics, left_on="Incident ID", right_on="case", how="left", validate="one_to_one")
    merged = merged.rename(columns={
        "Incident ID": "incident_id", "Application": "application",
        "Known Root Cause Service": "root_cause", "Fault Type": "fault_type"})
    merged = merged.assign(has_metrics=merged["case"].notna(),
                           group=merged["application"] + "|" + merged["root_cause"] + "|" + merged["fault_type"])
    merged = merged.drop(columns=["case"])

    num = merged[metric_cols].to_numpy(dtype=float)
    inventory = {
        "metadata_records": int(len(cases)),
        "feature_records": int(len(feats)),
        "joined_records": int(len(merged)),
        "records_without_metrics": merged.loc[~merged["has_metrics"], "incident_id"].tolist(),
        "numeric_metric_columns": int(len(metric_cols)),
        "label_mismatches_between_files": label_mismatch,
        "applications": merged["application"].value_counts().to_dict(),
        "root_cause_classes": int(merged["root_cause"].nunique()),
        "root_cause_counts": merged["root_cause"].value_counts().to_dict(),
        "fault_type_counts": merged["fault_type"].value_counts().to_dict(),
        "missing_metric_cell_fraction": float(np.isnan(num[merged["has_metrics"].to_numpy()]).mean()),
        "infinite_metric_cells": int(np.isinf(num[~np.isnan(num)]).sum()),
        "duplicate_metric_rows": int(merged.loc[merged["has_metrics"], metric_cols].duplicated().sum()),
        "scenario_groups": int(merged["group"].nunique()),
    }
    return merged, inventory


def parse_metric_column(col: str):
    """'ts-auth-service_cpu__standardized_change' -> ('ts-auth-service', 'cpu')."""
    base = col.split("__")[0]
    service, family = base.rsplit("_", 1)
    if family.startswith("latency"):      # latency, latency-50, latency-90 -> one family
        family = "latency"
    return service, family


def _as_matrix(df: pd.DataFrame, cols) -> np.ndarray:
    arr = df.reindex(columns=cols).to_numpy(dtype=float, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    return np.abs(arr)


def _family_score(arr: np.ndarray):
    """Max |change| over the columns of one family and which column achieved it."""
    filled = np.where(np.isnan(arr), -1.0, arr)
    if filled.shape[1] == 0:
        return np.full(len(arr), np.nan), np.zeros(len(arr), dtype=int)
    idx = filled.argmax(axis=1)
    score = filled.max(axis=1)
    score[score < 0] = np.nan
    return score, idx


# ----------------------------------------------------------------------------- features
class FeatureBuilder:
    """Turns raw metric columns into discrete Bayesian-Network variables.

    representation:
      family_levels      Level_<family>  (Low/Medium/High/Unknown) per metric family
      localized_nofault  Level_<family> + Loc_<family>  (which service is most anomalous)
      localized          same variables, different graph (adds a FaultType node)
      individual         top-K single metrics (tertile states), chosen by mutual information
    """

    def __init__(self, representation="localized", change="standardized", quiet_quantile=0.75,
                 n_individual=40, min_family_coverage=0.10):
        self.representation = representation
        self.change = change
        self.quiet_quantile = quiet_quantile
        self.n_individual = n_individual
        self.min_family_coverage = min_family_coverage
        self.fitted = False

    # -- fit
    def fit(self, df: pd.DataFrame, y_root: pd.Series):
        suffix = CHANGE_SUFFIX[self.change]
        self.columns = [c for c in df.columns if c.endswith(suffix)]
        if not self.columns:
            raise ValueError(f"No metric columns ending in '{suffix}' found.")
        parsed = {c: parse_metric_column(c) for c in self.columns}
        self.col_service = {c: parsed[c][0] for c in self.columns}
        self.candidates = sorted(pd.unique(y_root))
        self.applications = sorted(df["application"].unique())
        arr = _as_matrix(df, self.columns)

        self.family_cols, self.thresholds, self.quiet_thr = {}, {}, {}
        self.family_coverage = {}
        fam_all = sorted({p[1] for p in parsed.values()})
        for fam in fam_all:
            cols = [c for c in self.columns if parsed[c][1] == fam]
            score, _ = _family_score(arr[:, [self.columns.index(c) for c in cols]])
            cov = float(np.mean(~np.isnan(score)))
            self.family_coverage[fam] = cov
            if cov < self.min_family_coverage:
                continue                       # too rarely measured to be a useful variable
            valid = score[~np.isnan(score)]
            self.family_cols[fam] = cols
            self.thresholds[fam] = [float(np.quantile(valid, 1 / 3)), float(np.quantile(valid, 2 / 3))]
            self.quiet_thr[fam] = float(np.quantile(valid, self.quiet_quantile))
        self.families = list(self.family_cols)

        self.level_nodes = [f"Level_{f}" for f in self.families] if self.representation != "individual" else []
        self.loc_nodes = [f"Loc_{f}" for f in self.families] if self.representation in ("localized", "localized_nofault") else []

        self.metric_nodes, self.metric_cuts = [], {}
        if self.representation == "individual":
            self._fit_individual(arr, y_root)
        self.fitted = True
        return self

    def _fit_individual(self, arr, y_root):
        from sklearn.feature_selection import mutual_info_classif
        cuts = {}
        codes = np.full(arr.shape, 3, dtype=int)             # 3 = Unknown
        for j, c in enumerate(self.columns):
            v = arr[:, j]
            ok = ~np.isnan(v)
            if ok.sum() < 10 or np.unique(v[ok]).size < 3:
                continue
            q = np.quantile(v[ok], [1 / 3, 2 / 3])
            cuts[c] = [float(q[0]), float(q[1])]
            codes[ok, j] = np.digitize(v[ok], q)
        usable = list(cuts)
        idx = [self.columns.index(c) for c in usable]
        mi = mutual_info_classif(codes[:, idx], pd.factorize(y_root)[0], discrete_features=True, random_state=0)
        # n_individual=None or <=0 means retain EVERY usable distinct metric.
        # The final project uses this mode rather than a top-K subset.
        order = np.argsort(-mi)
        if self.n_individual is not None and self.n_individual > 0:
            order = order[: self.n_individual]
        chosen = [usable[i] for i in order]
        self.metric_cuts = {c: cuts[c] for c in chosen}
        self.metric_nodes = [self._metric_node(c) for c in chosen]
        self.metric_mi = {self._metric_node(usable[i]): float(mi[i]) for i in order}

    def _metric_node(self, col):
        s, f = parse_metric_column(col)
        return f"M[{s}|{f}]"

    # -- states
    def state_names(self):
        st = {}
        for n in self.level_nodes:
            st[n] = LEVELS + [UNKNOWN]
        for f, n in zip(self.families, self.loc_nodes):
            st[n] = [QUIET, OTHER] + [s for s in self.candidates] + [UNKNOWN]
        for n in self.metric_nodes:
            st[n] = LEVELS + [UNKNOWN]
        return st

    @property
    def feature_nodes(self):
        return self.level_nodes + self.loc_nodes + self.metric_nodes

    # -- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeatureBuilder must be fitted before transform().")
        out = pd.DataFrame(index=df.index)
        arr = _as_matrix(df, self.columns)
        for fam in self.families:
            cols = self.family_cols[fam]
            sub = arr[:, [self.columns.index(c) for c in cols]]
            score, idx = _family_score(sub)
            nan = np.isnan(score)
            if self.level_nodes:
                lv = np.array(LEVELS + [UNKNOWN], dtype=object)[np.digitize(np.nan_to_num(score), self.thresholds[fam])]
                lv[nan] = UNKNOWN
                out[f"Level_{fam}"] = lv
            if self.loc_nodes:
                svc = np.array([self.col_service[cols[i]] for i in idx], dtype=object)
                state = np.where(np.isin(svc, self.candidates), svc, OTHER).astype(object)
                state[np.nan_to_num(score) < self.quiet_thr[fam]] = QUIET
                state[nan] = UNKNOWN
                out[f"Loc_{fam}"] = state
        for c, q in self.metric_cuts.items():
            v = arr[:, self.columns.index(c)]
            lv = np.array(LEVELS, dtype=object)[np.digitize(np.nan_to_num(v), q)]
            lv[np.isnan(v)] = UNKNOWN
            out[self._metric_node(c)] = lv
        return out

    def summary_rows(self):
        rows = []
        for f in self.families:
            rows.append({"variable": f"Level_{f}", "family": f, "coverage_in_train": round(self.family_coverage[f], 3),
                         "low_medium_cut": self.thresholds[f][0], "medium_high_cut": self.thresholds[f][1],
                         "quiet_cut_for_Loc": self.quiet_thr[f]})
        for n, c in zip(self.metric_nodes, self.metric_cuts):
            rows.append({"variable": n, "family": parse_metric_column(c)[1], "low_medium_cut": self.metric_cuts[c][0],
                         "medium_high_cut": self.metric_cuts[c][1], "mutual_information": self.metric_mi[n]})
        return pd.DataFrame(rows)


def benchmark_matrix(df: pd.DataFrame, columns, clip_values=None):
    """Numeric matrix for the Extra-Trees benchmark (training-fitted clip values)."""
    x = df.reindex(columns=columns).to_numpy(dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan
    x = np.sign(x) * np.log1p(np.abs(x))     # tames the 1e19 outliers
    return np.nan_to_num(x, nan=0.0)
