"""Streamlit inference UI for the Titanic survival classifier.

Pick a trained model (PyTorch MLP or XGBoost), point it at a CSV (by path or
upload), and view predictions, metrics and plots. The app is pure glue: the
feature pipeline (``TitanicPreprocessor``) and the models are loaded from the
artifacts written by ``mlp/train.py`` / ``xgb/train.py``, so what you see here
reproduces the held-out validation numbers exactly.

Run:
    streamlit run ds_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data import PROJECT_ROOT
from src.preprocessing import TitanicPreprocessor

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CSV = PROJECT_ROOT / "data" / "val_split.csv"

STRATEGIES = {
    "PyTorch MLP": "mlp",
    "XGBoost": "xgb",
}


# ── Model adapters ───────────────────────────────────────────────────────────
# A thin uniform interface (``.proba(X) -> P(survived)``) so the rest of the app
# is model-agnostic. Each adapter wraps the framework-specific load + predict.
class MLPAdapter:
    def __init__(self, model):
        self.model = model

    def proba(self, x: np.ndarray) -> np.ndarray:
        from mlp.model import predict_proba
        return predict_proba(self.model, x)


class XGBAdapter:
    def __init__(self, model):
        self.model = model

    def proba(self, x: np.ndarray) -> np.ndarray:
        from xgb.model import predict_proba
        return predict_proba(self.model, x)


@st.cache_resource(show_spinner=False)
def load_artifacts(strategy: str):
    """Load (model_adapter, preprocessor) for a strategy. Cached across reruns.

    Raises FileNotFoundError with an actionable message if artifacts are missing.
    """
    strat_dir = MODELS_DIR / strategy
    pre_path = strat_dir / "preprocessor.joblib"
    if not pre_path.exists():
        raise FileNotFoundError(
            f"No preprocessor at {pre_path}. Train first: `python -m {strategy}.train`."
        )
    pre = TitanicPreprocessor.load(pre_path)

    if strategy == "mlp":
        import torch
        from mlp.model import MLP

        weights = strat_dir / "mlp.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"No model at {weights}. Train first: `python -m mlp.train`."
            )
        bundle = torch.load(weights, map_location="cpu", weights_only=False)
        model = MLP(bundle["input_dim"], bundle["config"]["hidden_dims"],
                    bundle["config"]["dropout"])
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        return MLPAdapter(model), pre

    import xgboost as xgb

    weights = strat_dir / "xgb.json"
    if not weights.exists():
        raise FileNotFoundError(
            f"No model at {weights}. Train first: `python -m xgb.train`."
        )
    model = xgb.XGBClassifier()
    model.load_model(weights)
    return XGBAdapter(model), pre


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Died (0)", "Survived (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_roc(y_true: np.ndarray, proba: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_prob_hist(proba: np.ndarray, pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.hist(proba[pred == 0], bins=20, range=(0, 1), alpha=0.7,
            label="Predicted died", color="#d62728")
    ax.hist(proba[pred == 1], bins=20, range=(0, 1), alpha=0.7,
            label="Predicted survived", color="#2ca02c")
    ax.axvline(0.5, ls="--", color="black", lw=1)
    ax.set_xlabel("P(survived)")
    ax.set_ylabel("Count")
    ax.set_title("Predicted probabilities")
    ax.legend()
    fig.tight_layout()
    return fig


# ── Explainability ───────────────────────────────────────────────────────────
# Conceptual features the model engineers (vs. raw Kaggle columns). Used to
# colour/flag engineered features green in the preview + importance views.
# Names here cover both the readable preview labels (e.g. "Title") and the
# one-hot column prefixes (e.g. "Title_Mr").
ENGINEERED = {"FamilySize", "IsAlone", "TicketGroupSize", "LogFare", "FareRank", "Title"}

# The 12 features the model actually consumes, in a readable order: raw first,
# engineered second (Tab 1 colours them grey vs green accordingly).
RAW_PREVIEW_COLS = ["Sex", "Age", "Pclass", "SibSp", "Parch", "Embarked"]
ENG_PREVIEW_COLS_ONLY = ["Title", "FamilySize", "IsAlone", "TicketGroupSize", "LogFare", "FareRank"]
PREVIEW_COLS = RAW_PREVIEW_COLS + ENG_PREVIEW_COLS_ONLY

# Categorical features that get one-hot expanded; their columns are "<prefix>_<cat>".
CAT_PREFIXES = ("Sex", "Pclass", "Embarked", "Title")

# One-line "how it was built" notes shown as a (?) tooltip on each engineered
# column header in Tab 1.
ENG_HELP = {
    "Title": "Extracted from Name (Mr, Mrs, Miss, Master, …; rare ones → Other).",
    "FamilySize": "SibSp + Parch + 1 (counts the passenger themselves).",
    "IsAlone": "1 if travelling alone (FamilySize = 1), else 0.",
    "TicketGroupSize": "How many passengers share the same ticket number.",
    "LogFare": "log(1 + Fare) — compresses the heavy fare skew.",
    "FareRank": "Fare's percentile (0–1) within the same Pclass.",
}


def _is_engineered(name: str) -> bool:
    return name in ENGINEERED or name.startswith("Title_")


def feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    """Map each conceptual feature -> the column indices that encode it.

    One-hot columns (``Sex_female``, ``Title_Mr`` …) collapse back to their
    parent feature (``Sex``, ``Title``); plain numeric columns map to themselves.
    Lets us measure importance *per feature across the dataset* (12 groups), not
    per one-hot column (23).
    """
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        key = name.split("_")[0] if name.startswith(CAT_PREFIXES) and "_" in name else name
        groups.setdefault(key, []).append(i)
    return groups


def permutation_importance(adapter, x: np.ndarray, y: np.ndarray,
                           feature_names: list[str],
                           groups: dict[str, list[int]] | None = None,
                           n_repeats: int = 5, seed: int = 42) -> pd.Series:
    """Mean ROC-AUC drop when a feature is shuffled (model-agnostic).

    If ``groups`` is given, all columns of a feature are shuffled together with a
    *shared* row permutation — so a one-hot block stays valid and we get the
    importance of the whole feature (e.g. all of ``Title``) across the dataset.
    Otherwise each of the 23 columns is shuffled on its own.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, adapter.proba(x))
    if groups is None:
        groups = {n: [i] for i, n in enumerate(feature_names)}

    out: dict[str, float] = {}
    for name, cols in groups.items():
        total = 0.0
        for _ in range(n_repeats):
            xp = x.copy()
            perm = rng.permutation(len(x))  # one shared shuffle keeps one-hot valid
            for c in cols:
                xp[:, c] = x[perm, c]
            total += base - roc_auc_score(y, adapter.proba(xp))
        out[name] = total / n_repeats
    return pd.Series(out).sort_values()


def plot_importance(imp: pd.Series):
    """Horizontal bar of permutation importance (all rows); engineered in green."""
    from matplotlib.patches import Patch

    colors = ["#2ca02c" if _is_engineered(n) else "#7f7f7f" for n in imp.index]
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_xlabel("Mean ROC-AUC drop when shuffled")
    ax.set_title("Permutation importance")
    ax.legend(handles=[Patch(color="#2ca02c", label="Engineered"),
                       Patch(color="#7f7f7f", label="Raw")], loc="lower right")
    fig.tight_layout()
    return fig


def style_preview(df: pd.DataFrame):
    """Colour engineered columns green, raw columns grey (Tab 1)."""
    def col_color(col):
        bg = "#1b5e20" if _is_engineered(col.name) else "#3a3a3a"
        return [f"background-color: {bg}; color: white"] * len(col)
    return df.style.apply(col_color, axis=0)


def _group_shap(sv, feature_names: list[str], groups: dict[str, list[int]]):
    """Collapse a per-column SHAP Explanation into per-feature contributions.

    SHAP values are additive, so a feature's contribution is the sum over its
    one-hot columns. The displayed value is the raw number for plain features,
    or the active category label (e.g. ``Mr``) for one-hot groups.
    """
    import shap

    vals, data = sv.values[0], sv.data[0]
    names, gv, gd = [], [], []
    for name, cols in groups.items():
        names.append(name)
        gv.append(float(np.sum([vals[c] for c in cols])))
        if len(cols) == 1:
            gd.append(float(data[cols[0]]))
        else:
            active = [c for c in cols if data[c] == 1]
            gd.append(feature_names[active[0]].split("_", 1)[1] if active else "")
    return shap.Explanation(
        values=np.array(gv)[None, :],
        base_values=np.array([float(sv.base_values[0])]),
        data=np.array(gd, dtype=object)[None, :],
        feature_names=names,
    )


def shap_waterfall(adapter, x: np.ndarray, i: int, feature_names: list[str],
                   groups: dict[str, list[int]] | None = None,
                   max_background: int = 100):
    """SHAP waterfall for one row: how each feature moved P(survived) from the
    baseline to this prediction. ``groups`` collapses one-hot columns into their
    parent feature (12 vs 23). Raises ImportError if `shap` isn't installed."""
    import shap

    bg = shap.utils.sample(x, min(max_background, len(x)), random_state=42)
    explainer = shap.Explainer(adapter.proba, shap.maskers.Independent(bg),
                               feature_names=list(feature_names))
    sv = explainer(x[i:i + 1])
    if groups is not None:
        sv = _group_shap(sv, feature_names, groups)
    plt.figure()
    shap.plots.waterfall(sv[0], max_display=len(sv.feature_names), show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def compute_results(strategy: str, strategy_label: str, path_str: str, upload) -> dict | None:
    """Read the CSV, load the model, run inference. Returns a results dict to stash
    in session_state, or None after showing a user-facing error."""
    # Load the input frame (upload wins over path).
    try:
        if upload is not None:
            df = pd.read_csv(upload)
            source = f"upload: {upload.name}"
        else:
            p = Path(path_str.strip().strip('"'))
            if not p.exists():
                st.error(f"CSV not found: `{p}`")
                return None
            df = pd.read_csv(p)
            source = str(p)
    except Exception as e:  # noqa: BLE001 - surface any read error to the user
        st.error(f"Could not read the CSV: {e}")
        return None

    try:
        adapter, pre = load_artifacts(strategy)
    except FileNotFoundError as e:
        st.error(str(e))
        return None

    try:
        x, y = pre.transform(df)
        proba = adapter.proba(x)
    except Exception as e:  # noqa: BLE001 - malformed columns etc.
        st.error(f"Inference failed (check the CSV columns): {e}")
        return None

    return {
        "strategy": strategy, "strategy_label": strategy_label, "source": source,
        "df": df, "adapter": adapter, "pre": pre, "x": x, "y": y,
        "proba": proba, "pred": (proba >= 0.5).astype(int), "has_labels": y is not None,
    }


# ── App ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Titanic Survival Inference", page_icon="🚢",
                       layout="wide")
    st.title("🚢 Titanic Survival — Inference UI")
    st.caption("Load a trained model, run it on a CSV, inspect metrics and plots.")

    # Sidebar controls.
    with st.sidebar:
        st.header("Controls")
        strategy_label = st.radio("Model", list(STRATEGIES.keys()))
        strategy = STRATEGIES[strategy_label]

        st.subheader("Dataset")
        path_str = st.text_input("CSV path", value=str(DEFAULT_CSV))
        upload = st.file_uploader("…or upload a CSV", type="csv")
        st.caption("An uploaded file overrides the path above.")

        run = st.button("Run inference", type="primary", use_container_width=True)

    # `st.button` is True only on the click's rerun; widget toggles below trigger
    # fresh reruns. Stash results in session_state so the page survives them.
    if run:
        res = compute_results(strategy, strategy_label, path_str, upload)
        if res is not None:
            st.session_state["results"] = res

    if "results" not in st.session_state:
        st.info("Pick a model and a dataset in the sidebar, then **Run inference**.")
        return

    r = st.session_state["results"]
    strategy, strategy_label, source = r["strategy"], r["strategy_label"], r["source"]
    df, adapter, pre = r["df"], r["adapter"], r["pre"]
    x, y, proba, pred, has_labels = r["x"], r["y"], r["proba"], r["pred"], r["has_labels"]

    st.success(f"Ran **{strategy_label}** on {len(df)} rows from `{source}`.")

    # Data preview (collapsed by default to keep the page tight).
    with st.expander(f"Input preview — {len(df)} rows × {df.shape[1]} columns", expanded=False):
        st.dataframe(df, use_container_width=True)
    if has_labels:
        st.caption("`Survived` column present → full evaluation below.")
    else:
        st.info("No `Survived` column → inference-only mode (predictions, no metrics).")

    # Metrics + label-dependent plots.
    if has_labels:
        y = y.astype(int)
        st.subheader("Metrics (held-out evaluation)")
        c = st.columns(5)
        c[0].metric("Accuracy", f"{accuracy_score(y, pred):.3f}")
        c[1].metric("Precision", f"{precision_score(y, pred, zero_division=0):.3f}")
        c[2].metric("Recall", f"{recall_score(y, pred, zero_division=0):.3f}")
        c[3].metric("F1", f"{f1_score(y, pred, zero_division=0):.3f}")
        c[4].metric("ROC-AUC", f"{roc_auc_score(y, proba):.3f}")

        st.subheader("Plots")
        p1, p2, p3 = st.columns(3)
        p1.pyplot(plot_confusion(y, pred))
        p2.pyplot(plot_roc(y, proba))
        p3.pyplot(plot_prob_hist(proba, pred))
    else:
        st.subheader("Predicted probability distribution")
        _, mid, _ = st.columns([1, 2, 1])
        mid.pyplot(plot_prob_hist(proba, pred))

    # Predictions table + download. Prediction columns lead; the rest follow.
    st.subheader("Predictions")
    out = df.copy()
    out.insert(0, "proba_survived", np.round(proba, 4))
    out.insert(0, "prediction", np.where(pred == 1, "Survived", "Died"))

    if has_labels:
        st.caption("Rows shaded **green** where the prediction matched the actual "
                   "outcome, **red** where it missed.")
        correct = pred == y
        styler = out.style.apply(
            lambda row: [
                f"background-color: {'#1b5e20' if correct[row.name] else '#7f1d1d'}"
            ] * len(row),
            axis=1,
        )
        st.dataframe(styler, use_container_width=True)
    else:
        st.dataframe(out, use_container_width=True)

    st.download_button(
        "Download predictions CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{strategy}.csv",
        mime="text/csv",
    )

    # ── Feature influence (explainability) ───────────────────────────────────
    st.subheader("Feature influence")
    tab_eng, tab_global, tab_local = st.tabs(
        ["Engineered features", "Global importance", "Why this passenger?"]
    )

    with tab_eng:
        st.caption("The 12 features the model actually consumes (imputed, before "
                   "scaling / one-hot). **Green = engineered**, **grey = raw**. "
                   "Hover the **?** on a green column to see how it was built.")
        eng = pre._build_features(df)
        cols = [c for c in PREVIEW_COLS if c in eng.columns]
        col_config = {c: st.column_config.Column(help=h) for c, h in ENG_HELP.items() if c in cols}
        st.dataframe(style_preview(eng[cols]), use_container_width=True,
                     column_config=col_config)

    with tab_global:
        if has_labels:
            by_feature = st.checkbox(
                "Group one-hot columns into their feature (12 features)", value=True,
                help="On: importance of each whole feature (Title, Sex …) across the "
                     "dataset. Off: all 23 encoded columns separately.",
            )
            view = "per feature (12)" if by_feature else "per column (23)"
            st.caption(f"Permutation importance, {view} — drop in ROC-AUC when a feature "
                       "is shuffled (mean of 5 shuffles). Engineered features in green.")
            groups = feature_groups(pre.feature_names_) if by_feature else None
            with st.spinner("Computing permutation importance…"):
                imp = permutation_importance(adapter, x, y, pre.feature_names_, groups)
            _, mid, _ = st.columns([1, 3, 1])
            mid.pyplot(plot_importance(imp))
        else:
            st.info("Permutation importance needs the `Survived` labels to measure the "
                    "ROC-AUC drop — unavailable in inference-only mode.")

    with tab_local:
        st.caption("SHAP — how each feature pushed this passenger's predicted "
                   "P(survived) above or below the baseline.")
        c_idx, c_grp = st.columns([1, 2])
        idx = int(c_idx.number_input("Row to explain (0-based)", min_value=0,
                                     max_value=len(df) - 1, value=0, step=1))
        by_feature_l = c_grp.checkbox(
            "Group one-hot columns into their feature (12 features)", value=True,
            key="shap_group",
            help="On: contribution of each whole feature (Title, Sex …). "
                 "Off: all 23 encoded columns separately.",
        )
        summary = f"Prediction: **{out['prediction'].iloc[idx]}**  ·  P(survived) = {proba[idx]:.3f}"
        if has_labels:
            summary += f"  ·  actual: **{'Survived' if y[idx] == 1 else 'Died'}**"
        st.write(summary)
        groups_l = feature_groups(pre.feature_names_) if by_feature_l else None
        try:
            with st.spinner("Computing SHAP values…"):
                fig = shap_waterfall(adapter, x, idx, pre.feature_names_, groups_l)
            st.pyplot(fig)
        except ImportError:
            st.warning(
                "Install SHAP for per-row explanations:\n\n"
                "`pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org shap`"
            )
        except Exception as e:  # noqa: BLE001 - surface SHAP failures, don't crash
            st.error(f"SHAP explanation failed: {e}")


if __name__ == "__main__":
    main()
