# src/explain.py

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Feature name extraction
# ------------------------------------------------------------------

def get_feature_names_from_preprocessor(preprocessor, include_treatment=True):
    """
    Safely extract feature names from a ColumnTransformer-based preprocessor.

    Assumptions:
    - Named transformers: 'num' and 'cat'
    - Categorical pipeline contains OneHotEncoder named 'ohe'
    - Treatment is appended manually for S-Learner (optional)
    """

    feature_names = []

    # Numeric features
    if 'num' in preprocessor.named_transformers_:
        num_cols = preprocessor.named_transformers_['num'].feature_names_in_
        feature_names.extend(list(num_cols))

    # Categorical features
    if 'cat' in preprocessor.named_transformers_:
        cat_pipe = preprocessor.named_transformers_['cat']
        ohe = cat_pipe.named_steps.get('ohe')

        if ohe is not None:
            cat_cols = cat_pipe.feature_names_in_
            try:
                ohe_names = ohe.get_feature_names_out(cat_cols)
                feature_names.extend(list(ohe_names))
            except Exception:
                # Fallback if sklearn version mismatch
                feature_names.extend([f"cat__{c}" for c in cat_cols])

    # Treatment column (S-Learner only)
    if include_treatment:
        feature_names.append("treatment")

    return feature_names


# ------------------------------------------------------------------
# Global feature importance
# ------------------------------------------------------------------

def compute_global_feature_importance(model, feature_names):
    """
    Compute global feature importance from a LightGBM model.

    Returns:
    - DataFrame sorted by gain importance
    """

    importance_gain = model.feature_importance(importance_type='gain')
    importance_split = model.feature_importance(importance_type='split')

    n = min(len(feature_names), len(importance_gain))
    feature_names = feature_names[:n]

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": importance_gain[:n],
        "importance_split": importance_split[:n]
    }).sort_values("importance_gain", ascending=False)

    return df_importance


# ------------------------------------------------------------------
# Local uplift explanation (S-Learner)
# ------------------------------------------------------------------

def local_uplift_explanation(
    idx,
    X_df,
    X_train_p,
    model,
    preprocessor,
    feature_names,
    top_k=8
):
    """
    Generate a local uplift explanation for a single instance using
    sensitivity-based approximation.

    Returns:
    - p1, p0, uplift
    - top contributing features
    """

    # ---- Input safety ----
    if idx not in X_df.index:
        raise ValueError(f"Index {idx} not found in X_df")

    # ---- Select and preprocess single row ----
    x_row = X_df.loc[[idx]]
    Xp = preprocessor.transform(x_row)

    row_p = Xp.flatten()

    # ---- Base predictions ----
    p1 = model.predict(
        np.hstack([Xp, np.ones((1, 1))])
    )[0]

    p0 = model.predict(
        np.hstack([Xp, np.zeros((1, 1))])
    )[0]

    uplift = p1 - p0

    base_uplift = uplift

    # ---- Feature sensitivity approximation ----
    contributions = []
    n_features = min(len(row_p), X_train_p.shape[1])

    for i in range(n_features):
        temp = row_p.copy()

        # Replace with training median
        med = np.median(X_train_p[:, i])
        temp[i] = med

        pt1 = model.predict(
            np.hstack([temp.reshape(1, -1), np.ones((1, 1))])
        )[0]

        pt0 = model.predict(
            np.hstack([temp.reshape(1, -1), np.zeros((1, 1))])
        )[0]

        eff = base_uplift - (pt1 - pt0)
        contributions.append((i, eff))

    # ---- Select top-k contributors ----
    contributions = sorted(contributions, key=lambda x: -abs(x[1]))[:top_k]

    mapped = []
    for i, eff in contributions:
        fname = feature_names[i] if i < len(feature_names) else f"f_{i}"
        mapped.append({
            "feature": fname,
            "effect_on_uplift": float(eff)
        })

    return {
        "index": int(idx),
        "p1": float(p1),
        "p0": float(p0),
        "uplift": float(uplift),
        "top_features": mapped
    }
