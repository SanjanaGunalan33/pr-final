"""Discrete Bayesian Network for root-cause analysis (pgmpy).

Graph variants (all are DAGs, built from the fitted FeatureBuilder):
  localized          Application -> RootCause, Application -> FaultType,
                     RootCause -> Loc_<family>, FaultType -> Level_<family>, Application -> Level_<family>
  localized_nofault  Application -> RootCause, RootCause -> Level_<family>, RootCause -> Loc_<family>
  family_levels      Application -> RootCause, RootCause -> Level_<family>
  individual         Application -> RootCause, RootCause -> M[service|family]   (top-K metrics)
"""
from __future__ import annotations

import logging
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pgmpy").setLevel(logging.ERROR)

from pgmpy.estimators import BayesianEstimator          # noqa: E402
from pgmpy.inference import VariableElimination         # noqa: E402
from pgmpy.models import DiscreteBayesianNetwork        # noqa: E402

from preprocessing import UNKNOWN, FeatureBuilder        # noqa: E402

EQUIVALENT_SAMPLE_SIZE = 10      # BDeu prior strength (documented in README)


def _cause_node(label):
    import re
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(label)).strip("_")[:70]
    return "Cause__" + (safe or "Unknown")


def build_edges(builder: FeatureBuilder):
    rep = builder.representation
    edges = [("Application", "RootCause"), ("RootCause", "FaultType")]
    # Explicit one-node-per-root-cause indicator layer. RootCause remains the
    # mutually-exclusive latent class used for stable, tractable VE inference.
    edges.extend(("RootCause", _cause_node(c)) for c in builder.candidates)
    if rep == "localized":
        edges.append(("Application", "FaultType"))
        for n in builder.loc_nodes:
            edges.append(("RootCause", n))
        for n in builder.level_nodes:
            edges += [("FaultType", n), ("Application", n)]
    elif rep in ("localized_nofault", "family_levels"):
        for n in builder.loc_nodes + builder.level_nodes:
            edges.append(("RootCause", n))
    elif rep == "individual":
        for n in builder.metric_nodes:
            edges.append(("RootCause", n))
    else:
        raise ValueError(f"Unknown representation '{rep}'")
    return edges


class RCAModel:
    def __init__(self, representation="localized", change="standardized", quiet_quantile=0.75, n_individual=40):
        self.builder = FeatureBuilder(representation, change, quiet_quantile, n_individual)
        self.model = None
        self.engine = None

    # ------------------------------------------------------------------ training
    def fit(self, df: pd.DataFrame):
        """df: joined training rows (must contain application, root_cause, fault_type, metrics)."""
        df = df[df["has_metrics"]] if "has_metrics" in df.columns else df
        self.builder.fit(df, df["root_cause"])
        feats = self.builder.transform(df)
        data = pd.DataFrame({"Application": df["application"].values, "RootCause": df["root_cause"].values}, index=df.index)
        # FaultType is retained as an intermediate causal descriptor for every
        # representation. Separate binary nodes expose each root-cause class.
        data["FaultType"] = df["fault_type"].fillna("Unknown").astype(str).values
        for cause in self.builder.candidates:
            data[_cause_node(cause)] = (df["root_cause"].values == cause).astype(str)
        data = pd.concat([data, feats], axis=1)

        self.state_names = {
            "Application": sorted(df["application"].unique()),
            "RootCause": list(self.builder.candidates),
            "FaultType": sorted(data["FaultType"].unique()),
            **{_cause_node(c): ["False", "True"] for c in self.builder.candidates},
            **self.builder.state_names(),
        }
        # services that really occurred with each application in training
        self.app_services = df.groupby("application")["root_cause"].agg(lambda s: sorted(set(s))).to_dict()
        self.fault_by_app = df.groupby("application")["fault_type"].agg(lambda s: sorted(set(s))).to_dict()

        model = DiscreteBayesianNetwork(build_edges(self.builder))
        est = BayesianEstimator(model, data, state_names=self.state_names)
        cpds = [est.estimate_cpd(n, prior_type="BDeu", equivalent_sample_size=EQUIVALENT_SAMPLE_SIZE)
                for n in model.nodes()]
        model.add_cpds(*cpds)
        model.check_model()
        self.model = model
        self.engine = VariableElimination(model)
        self.n_train = int(len(df))
        return self

    # ------------------------------------------------------------------ inference
    def _evidence(self, application, symptoms):
        """Keep only valid, known evidence. Unknown / unsupported values are left unobserved."""
        notes, ev = [], {}
        if application in self.state_names["Application"]:
            ev["Application"] = application
        else:
            notes.append(f"Application '{application}' was not seen in training; it is treated as unobserved.")
        for node, state in (symptoms or {}).items():
            if node not in self.state_names or node in ("Application", "RootCause", "FaultType"):
                notes.append(f"Ignored unsupported variable '{node}'.")
            elif state == UNKNOWN or state is None or (isinstance(state, float) and np.isnan(state)):
                continue
            elif state not in self.state_names[node]:
                notes.append(f"Ignored unseen state '{state}' for {node}.")
            else:
                ev[node] = state
        return ev, notes

    def posterior(self, application, symptoms, target="RootCause"):
        """Posterior over `target` given evidence (Variable Elimination). Returns (Series, notes)."""
        ev, notes = self._evidence(application, symptoms)
        ev = {k: v for k, v in ev.items() if k != target}
        phi = self.engine.query([target], evidence=ev, joint=False, show_progress=False)
        phi = phi[target] if isinstance(phi, dict) else phi
        s = pd.Series(phi.values, index=phi.state_names[target])
        if target == "RootCause" and application in self.app_services:
            s = s[s.index.isin(self.app_services[application])]
        elif target == "FaultType" and application in self.fault_by_app:
            s = s[s.index.isin(self.fault_by_app[application])]
        s = s / s.sum()
        return s.sort_values(ascending=False), ev, notes

    def predict_incident(self, application, symptoms, top_k=5):
        post, ev, notes = self.posterior(application, symptoms)
        out = {"ranking": [(k, float(v)) for k, v in post.head(top_k).items()], "evidence_used": ev, "notes": notes}
        if "FaultType" in self.state_names:
            fp, _, _ = self.posterior(application, symptoms, "FaultType")
            out["fault_ranking"] = [(k, float(v)) for k, v in fp.head(3).items()]
        return out

    def predict_frame(self, df: pd.DataFrame, top_k=5):
        """Batch prediction from raw rows: preprocess with the fitted builder, then infer."""
        feats = self.builder.transform(df)
        rows = []
        for i, (_, r) in enumerate(df.iterrows()):
            sym = feats.iloc[i].to_dict() if r.get("has_metrics", True) else {}
            post, _, notes = self.posterior(r["application"], sym)
            rows.append(post if top_k is None else post.head(max(top_k, 5)))
        return rows

    def explain(self, application, symptoms):
        """Hand calculation  P(RC | app, evidence) ~ P(RC | app) * prod P(child | RC)  and its VE check.

        Valid because every observed child of RootCause has RootCause as its only parent; evidence
        nodes that are not children of RootCause (e.g. Level_* when Application is known) are
        d-separated from RootCause and cancel out. Returns (table, evidence_used, max_abs_diff_vs_VE).
        """
        ev, _ = self._evidence(application, symptoms)
        if "Application" not in ev:
            return None, ev, None
        # pgmpy's to_dataframe(): rows = parent-state combinations, columns = the node's own states
        prior_cpd = self.cpd_frame("RootCause")
        services = self.app_services[application]
        tab = pd.DataFrame({"prior P(RC|App)": prior_cpd.loc[application].reindex(services)})
        used = []
        for child in sorted(self.model.get_children("RootCause")):
            if child in ev:
                cpt = self.cpd_frame(child)
                tab[f"P({child}={ev[child]} | RC)"] = [cpt.loc[s, ev[child]] for s in services]
                used.append(child)
        tab["unnormalised"] = tab.prod(axis=1)
        tab["posterior (hand)"] = tab["unnormalised"] / tab["unnormalised"].sum()
        post, _, _ = self.posterior(application, symptoms)
        tab["posterior (Variable Elimination)"] = post.reindex(services)
        diff = float((tab["posterior (hand)"] - tab["posterior (Variable Elimination)"]).abs().max())
        return tab.sort_values("posterior (hand)", ascending=False), ev, diff

    def node_roles(self):
        """root = no parents, leaf = no children, intermediate = both."""
        roles = {}
        for n in self.model.nodes():
            p, c = self.model.get_parents(n), self.model.get_children(n)
            roles[n] = "root" if not p else ("leaf" if not c else "intermediate")
        return roles

    # ------------------------------------------------------------------ introspection
    def edges(self):
        return sorted(self.model.edges())

    def nodes(self):
        return sorted(self.model.nodes())

    def cpd_frame(self, node):
        """Rows = parent-state combinations, columns = the node's states (a root node has one 'prior' row)."""
        cpd = self.model.get_cpds(node)
        if len(cpd.variables) == 1:
            return pd.DataFrame([cpd.get_values().ravel()], columns=cpd.state_names[node], index=["prior"])
        return cpd.to_dataframe()

    def all_cpds_long(self):
        rows = []
        for node in self.nodes():
            cpd = self.model.get_cpds(node)
            parents = list(cpd.variables[1:])
            vals = cpd.get_values()
            states = cpd.state_names[node]
            combos = pd.MultiIndex.from_product([cpd.state_names[p] for p in parents]) if parents else [()]
            for j, combo in enumerate(combos):
                combo = combo if isinstance(combo, tuple) else (combo,)
                for i, st in enumerate(states):
                    rows.append({"node": node, "state": st,
                                 "parents": "; ".join(f"{p}={v}" for p, v in zip(parents, combo)) or "(none)",
                                 "probability": float(vals[i, j])})
        return pd.DataFrame(rows)

    def validate_cpds(self, tol=1e-6):
        """True if every CPT column sums to one and the graph is a DAG."""
        import networkx as nx
        ok = nx.is_directed_acyclic_graph(nx.DiGraph(self.model.edges()))
        for cpd in self.model.get_cpds():
            sums = cpd.get_values().sum(axis=0)
            ok &= bool(np.allclose(sums, 1.0, atol=tol))
        return bool(ok and self.model.check_model())

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def draw_network(model, path, title="Fitted Bayesian Network (RCAEval)"):
    """Draw the real fitted network (nodes/edges read from the model object) and save a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    G = nx.DiGraph(model.edges())
    G.add_nodes_from(model.nodes())
    assert set(G.nodes) == set(model.nodes()) and set(G.edges) == set(model.edges()), "plot/model mismatch"
    assert nx.is_directed_acyclic_graph(G)

    layers = list(nx.topological_generations(G))
    pos = {}
    for li, layer in enumerate(layers):
        layer = sorted(layer)
        for k, n in enumerate(layer):
            pos[n] = ((k - (len(layer) - 1) / 2) * 2.4, -li * 3.2)

    roles = {}
    for n in G.nodes:
        roles[n] = "Root node (prior)" if G.in_degree(n) == 0 else ("Leaf node (symptom)" if G.out_degree(n) == 0 else "Intermediate node")
    colors = {"Root node (prior)": "#4C72B0", "Intermediate node": "#DD8452", "Leaf node (symptom)": "#55A868"}
    kind = lambda n: roles[n]
    width = max(12, 1.5 * max(len(l) for l in layers))
    fig, ax = plt.subplots(figsize=(width, 3.2 * len(layers) + 1.5), dpi=200)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=14, edge_color="#777777", width=0.9,
                           node_size=2600, connectionstyle="arc3,rad=0.04")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[colors[kind(n)] for n in G.nodes], node_size=2600,
                           edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6.5, font_color="white", font_weight="bold",
                            labels={n: n.replace("_", chr(10), 1) if "_" in n else n for n in G.nodes})
    present = [k for k in colors if k in {kind(n) for n in G.nodes}]
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[k], markersize=11, label=k)
                       for k in present], loc="upper left", frameon=True)
    for li, layer in enumerate(layers):
        ax.text(min(x for x, _ in pos.values()) - 2.2, -li * 3.2, f"Level {li + 1}", fontsize=9, color="#555555", va="center")
    ax.set_title(f"{title}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return G.number_of_nodes(), G.number_of_edges()
