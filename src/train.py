# src/train.py

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split


def train_lgb_classifier(
    X,
    y,
    params=None,
    num_boost_round=300,
    random_state=42,
    valid_split=0.15
):
    """
    Generic LightGBM binary classifier trainer.

    Used by:
    - S-Learner
    - T-Learner (treated / control models)

    Early stopping is intentionally avoided to ensure
    compatibility across environments and stable CATE estimates.
    """

    # -------------------------
    # Optional validation split
    # -------------------------
    if valid_split and 0 < valid_split < 1.0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=valid_split,
            random_state=random_state,
            stratify=y
        )

        train_set = lgb.Dataset(X_tr, label=y_tr)
        valid_sets = [
            train_set,
            lgb.Dataset(X_val, label=y_val)
        ]
        valid_names = ["train", "valid"]

    else:
        train_set = lgb.Dataset(X, label=y)
        valid_sets = [train_set]
        valid_names = ["train"]

    # -------------------------
    # Base parameters
    # -------------------------
    base_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": random_state,
        "verbosity": -1
    }

    if params:
        base_params.update(params)

    # -------------------------
    # Train model
    # -------------------------
    model = lgb.train(
        params=base_params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names
    )

    return model


def build_s_learner_matrix(X_p, t):
    """
    Append treatment indicator as a feature for S-Learner.

    Parameters
    ----------
    X_p : np.ndarray
        Preprocessed feature matrix
    t : pandas.Series or np.ndarray
        Treatment indicator (0/1)

    Returns
    -------
    np.ndarray
        Feature matrix with treatment appended
    """

    if hasattr(t, "values"):
        t_arr = t.values
    else:
        t_arr = np.asarray(t)

    return np.hstack([X_p, t_arr.reshape(-1, 1)])


def train_t_learner(
    X_train_p,
    y_train,
    t_train,
    min_samples=100,
    num_boost_round=300,
    random_state=42
):
    """
    Train T-Learner models safely.

    Two separate models are trained:
    - One on treated samples
    - One on control samples

    If either group has insufficient samples, training is skipped.

    Returns
    -------
    (model_treat, model_ctrl) or (None, None)
    """

    # Split by treatment
    X_tr_treated = X_train_p[t_train == 1]
    y_tr_treated = y_train[t_train == 1]

    X_tr_control = X_train_p[t_train == 0]
    y_tr_control = y_train[t_train == 0]

    # Safety check
    if (X_tr_treated.shape[0] < min_samples) or (X_tr_control.shape[0] < min_samples):
        return None, None

    # Train treated model
    model_t_treat = train_lgb_classifier(
        X_tr_treated,
        y_tr_treated,
        num_boost_round=num_boost_round,
        valid_split=0.0,
        random_state=random_state
    )

    # Train control model
    model_t_ctrl = train_lgb_classifier(
        X_tr_control,
        y_tr_control,
        num_boost_round=num_boost_round,
        valid_split=0.0,
        random_state=random_state
    )

    return model_t_treat, model_t_ctrl
