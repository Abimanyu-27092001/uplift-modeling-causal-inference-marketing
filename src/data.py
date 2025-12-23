# src/data.py

import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_data(path) -> pd.DataFrame:
    """
    Load dataset from CSV.
    """
    return pd.read_csv(path)


# -------------------------------------------------------------------
# Feature / target / treatment construction (Hillstrom-specific)
# -------------------------------------------------------------------

def build_xy_treatment(df: pd.DataFrame):
    """
    Build feature matrix X, target y, and treatment indicator t
    from the Hillstrom email marketing dataset.

    Definitions:
    - treatment = 1 if customer received any email
    - treatment = 0 if control group (No E-Mail)
    - conversion = provided binary column
    """

    df = df.copy()

    # ---- Validate raw columns ----
    required = {"segment", "conversion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---- Target ----
    y = df["conversion"]

    # ---- Treatment (derived, NOT expected in raw data) ----
    t = (df["segment"] != "No E-Mail").astype(int)

    # ---- Features ----
    X = df.drop(columns=["segment", "conversion"])

    return X, y, t


# -------------------------------------------------------------------
# Train / test split
# -------------------------------------------------------------------

def stratified_uplift_split(X, y, t, test_size=0.25, seed=42):
    """
    Stratified split preserving treatment and outcome distribution.

    Strategy:
    1. Try joint stratification (treatment + outcome)
    2. Fallback to treatment-only stratification
    3. Final fallback: random split
    """

    try:
        stratify_col = t.astype(str) + "_" + y.astype(str)
        train_idx, test_idx = train_test_split(
            X.index,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_col
        )
    except ValueError:
        try:
            train_idx, test_idx = train_test_split(
                X.index,
                test_size=test_size,
                random_state=seed,
                stratify=t
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                X.index,
                test_size=test_size,
                random_state=seed,
                shuffle=True
            )

    return (
        X.loc[train_idx], X.loc[test_idx],
        y.loc[train_idx], y.loc[test_idx],
        t.loc[train_idx], t.loc[test_idx]
    )
