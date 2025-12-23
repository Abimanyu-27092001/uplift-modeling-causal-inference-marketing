# src/features.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def infer_feature_types(X: pd.DataFrame):
    """
    Infer numerical and categorical columns from feature matrix X.
    """
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    """
    Build ColumnTransformer preprocessing pipeline.
    """

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names(preprocessor, num_cols, cat_cols):
    """
    Recover output feature names after preprocessing.
    Used for feature importance and explainability.
    """

    feature_names = []

    # Numerical features (unchanged names)
    feature_names.extend(num_cols)

    # Categorical features (OHE expanded)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        ohe_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names.extend(ohe_feature_names.tolist())

    return feature_names
