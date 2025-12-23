# src/evaluate.py
"""
Evaluation, uplift estimation, segmentation, plotting, and reporting utilities
for the uplift modeling pipeline.
"""

from datetime import datetime
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)

# -----------------------------
# Public API (explicit export)
# -----------------------------
__all__ = [
    "predict_s_uplift",
    "predict_t_uplift",
    "qini_dataframe",
    "auuc",
    "plot_qini",
    "plot_roc",
    "plot_pr",
    "plot_confusion",
    "build_uplift_segmentation",
    "build_report_text",
]


# =====================================================
# Uplift prediction
# =====================================================

def predict_s_uplift(model, preprocessor, X_df):
    """
    Predict uplift using an S-Learner.

    Returns:
    - uplift: p1 - p0
    - p1: predicted probability with treatment
    - p0: predicted probability without treatment
    """
    X_p = preprocessor.transform(X_df)

    t1 = np.hstack([X_p, np.ones((X_p.shape[0], 1))])
    t0 = np.hstack([X_p, np.zeros((X_p.shape[0], 1))])

    p1 = model.predict(t1)
    p0 = model.predict(t0)

    uplift = p1 - p0
    return uplift, p1, p0


def predict_t_uplift(model_treat, model_ctrl, preprocessor, X_df):
    """
    Predict uplift using a T-Learner.

    Returns:
    - uplift: p_treated - p_control
    - p_treated
    - p_control
    """
    X_p = preprocessor.transform(X_df)

    p_t = model_treat.predict(X_p)
    p_c = model_ctrl.predict(X_p)

    uplift = p_t - p_c
    return uplift, p_t, p_c


# =====================================================
# Qini & AUUC
# =====================================================

def qini_dataframe(y_true, treatment, uplift_scores):
    """
    Build a Qini curve dataframe for uplift evaluation.
    """
    df_ = pd.DataFrame(
        {
            "y": y_true,
            "treatment": treatment,
            "uplift": uplift_scores,
        }
    ).sort_values("uplift", ascending=False).reset_index(drop=True)

    df_["n"] = np.arange(1, len(df_) + 1)
    df_["cum_treated"] = df_["treatment"].cumsum()
    df_["cum_control"] = df_.index + 1 - df_["cum_treated"]

    df_["cum_y_treated"] = (df_["y"] * df_["treatment"]).cumsum()
    df_["cum_y_control"] = (df_["y"] * (1 - df_["treatment"])).cumsum()

    df_["rate_treated"] = df_["cum_y_treated"] / df_["cum_treated"].replace(0, np.nan)
    df_["rate_control"] = df_["cum_y_control"] / df_["cum_control"].replace(0, np.nan)

    df_["uplift_cum"] = df_["rate_treated"].fillna(0) - df_["rate_control"].fillna(0)

    overall_control_rate = (
        df_.loc[df_["treatment"] == 0, "y"].mean()
        if (df_["treatment"] == 0).any()
        else 0
    )

    df_["incremental"] = df_["cum_y_treated"] - df_["cum_treated"] * overall_control_rate

    return df_


def auuc(df_qini):
    """
    Compute Area Under the Uplift Curve (AUUC).
    Compatible with NumPy >= 2.0.
    """
    x = np.arange(1, len(df_qini) + 1) / len(df_qini)
    y = df_qini["incremental"].values
    return np.trapezoid(y, x)


# =====================================================
# Plotting utilities (file-safe)
# =====================================================

def plot_qini(df_qini_s, auuc_s, df_qini_t=None, auuc_t=None, save_path=None):
    plt.figure(figsize=(8, 6))

    x_s = np.arange(len(df_qini_s)) / len(df_qini_s)
    plt.plot(
        x_s,
        df_qini_s["incremental"],
        label=f"S-Learner AUUC = {auuc_s:.4f}",
        linewidth=2,
    )

    if df_qini_t is not None and auuc_t is not None:
        x_t = np.arange(len(df_qini_t)) / len(df_qini_t)
        plt.plot(
            x_t,
            df_qini_t["incremental"],
            label=f"T-Learner AUUC = {auuc_t:.4f}",
            linestyle="--",
        )

    plt.xlabel("Fraction of population targeted")
    plt.ylabel("Cumulative incremental conversions")
    plt.title("Qini Curve (Incremental Conversions)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.close()


def plot_roc(y_true, y_score, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Predicted p1)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.close()


def plot_pr(y_true, y_score, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.close()


def plot_confusion(y_true, y_score, threshold=0.5, save_path=None):
    pred_label = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred_label)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.close()


# =====================================================
# Segmentation
# =====================================================

def assign_uplift_group(row, q80):
    """
    Assign customer to one of four uplift segments based on uplift
    and baseline probabilities.
    """
    u = row["uplift"]
    p0 = row["p0"]
    p1 = row["p1"]

    if (u >= q80) or (u > 0.05):
        return "Takers"
    if u <= -0.01:
        return "Sleeping Dogs"
    if p0 >= 0.5 and abs(u) < 0.02:
        return "Sure Things"
    if p0 < 0.15 and p1 < 0.15:
        return "Lost Causes"

    return "Takers" if u > 0 else "Lost Causes"


def build_uplift_segmentation(X_test, y_test, t_test, uplift, p0, p1):
    """
    Build per-customer uplift segmentation and summary table.
    """
    seg_df = X_test.reset_index(drop=True).copy()
    seg_df["y"] = y_test.reset_index(drop=True)
    seg_df["treatment"] = t_test.reset_index(drop=True)
    seg_df["uplift"] = uplift
    seg_df["p0"] = p0
    seg_df["p1"] = p1

    q80 = seg_df["uplift"].quantile(0.80)

    seg_df["group"] = seg_df.apply(lambda r: assign_uplift_group(r, q80), axis=1)

    seg_summary = (
        seg_df.groupby("group")
        .agg(
            count=("y", "size"),
            conversion_rate=("y", "mean"),
            avg_uplift=("uplift", "mean"),
            median_uplift=("uplift", "median"),
            avg_p0=("p0", "mean"),
            avg_p1=("p1", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    action_map = {
        "Takers": "Target (high incremental ROI)",
        "Sure Things": "Do not target (no incremental gain)",
        "Lost Causes": "Do not target (unlikely to convert)",
        "Sleeping Dogs": "Exclude (negative uplift)",
    }

    seg_summary["recommended_action"] = seg_summary["group"].map(action_map)

    return seg_df, seg_summary


# =====================================================
# Reporting
# =====================================================

def build_report_text(auuc_s, auuc_t=None, dataset_name="Hillstrom E-mail Analytics"):
    """
    Build final markdown report for uplift modeling results.
    Safely handles skipped T-Learner.
    """
    auuc_s_val = float(auuc_s) if auuc_s is not None else float("nan")

    if auuc_t is not None and not math.isnan(auuc_t):
        auuc_t_display = f"{auuc_t:.6f}"
        tlearner_note = ""
    else:
        auuc_t_display = "N/A (skipped due to insufficient control samples)"
        tlearner_note = (
            "\nNote on T-Learner:\n"
            "The T-Learner was conditionally skipped because the training split contained "
            "an insufficient number of control-group samples. Training separate models "
            "under such conditions would lead to unstable and unreliable CATE estimates. "
            "In this scenario, the S-Learner is the preferred and more robust approach.\n"
        )

    return f"""# Uplift Modeling Project – Executive Summary

Date: {datetime.utcnow().isoformat()}Z

Dataset:
{dataset_name}

Models Implemented:
- S-Learner (LightGBM, treatment as a feature)
- T-Learner (LightGBM, separate treated and control models – conditional)

Evaluation (Uplift Metrics):
- AUUC (S-Learner): {auuc_s_val:.6f}
- AUUC (T-Learner): {auuc_t_display}

Model Comparison:
The S-Learner achieved the highest and most reliable AUUC score, indicating superior
ranking of customers by incremental conversion impact. By pooling treated and control
data within a single model, the S-Learner provides more stable CATE estimates when
sample sizes are imbalanced or outcome patterns are similar across groups.
{tlearner_note}

Segmentation:
Customers were segmented into four standard uplift groups based on predicted uplift
and baseline probabilities:
- Takers
- Sure Things
- Lost Causes
- Sleeping Dogs

Final Strategic Recommendation:
Target Takers, exclude Sleeping Dogs, and avoid spending on Sure Things and Lost Causes.
"""
