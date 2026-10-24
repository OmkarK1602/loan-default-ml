"""Functions to measure and plot how good a model is."""

import matplotlib
matplotlib.use("Agg")  # draw plots without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, average_precision_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def get_metrics(y_true, y_proba, threshold=0.5):
    """Calculate all the metrics we track in MLflow."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def save_confusion_matrix(y_true, y_proba, path, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No default", "Default"], cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc_curve(y_true, y_proba, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_title("ROC curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_pr_curve(y_true, y_proba, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.axhline(np.mean(y_true), color="k", linestyle="--", label="Default rate")
    ax.set_title("Precision-Recall curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_classification_report(y_true, y_proba, path, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    text = classification_report(y_true, y_pred, target_names=["No default", "Default"])
    with open(path, "w") as f:
        f.write(text)


def get_feature_importance(pipeline):
    """Return a DataFrame of feature importances (or None if the model has none)."""
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])  # size of the coefficient
    else:
        return None
    df = pd.DataFrame({"feature": names, "importance": values})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_feature_importance(pipeline, png_path, csv_path, top=15):
    df = get_feature_importance(pipeline)
    if df is None:
        return False
    df.to_csv(csv_path, index=False)
    small = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(small["feature"], small["importance"], color="#3b6ea5")
    ax.set_title("Top feature importances")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return True


def compare_thresholds(y_true, y_proba, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    """Bonus: show how precision/recall change when we move the cut-off."""
    rows = []
    for t in thresholds:
        m = get_metrics(y_true, y_proba, threshold=t)
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
            "f1": round(m["f1"], 3),
            "false_positives": int(((y_pred == 1) & (np.asarray(y_true) == 0)).sum()),
            "false_negatives": int(((y_pred == 0) & (np.asarray(y_true) == 1)).sum()),
        })
    return pd.DataFrame(rows)
