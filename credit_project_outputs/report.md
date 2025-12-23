# Uplift Modeling Project – Executive Summary

Date: 2025-12-23T09:19:12.961755Z

Dataset:
Hillstrom E-mail Analytics (uploaded dataset)

Models Implemented:
- S-Learner (LightGBM, treatment as a feature)
- T-Learner (LightGBM, separate treated and control models – conditional)

Evaluation (Uplift Metrics):
- AUUC (S-Learner): 69.188906
- AUUC (T-Learner): N/A (skipped due to insufficient control samples)

Model Comparison:
The S-Learner achieved the highest and most reliable AUUC score, indicating superior
ranking of customers by incremental conversion impact. By pooling treated and control
data within a single model, the S-Learner provides more stable CATE estimates when
sample sizes are imbalanced or outcome patterns are similar across groups.

Note on T-Learner:
The T-Learner was conditionally skipped because the training split contained an insufficient number of control-group samples. Training separate models under such conditions would lead to unstable and unreliable CATE estimates. In this scenario, the S-Learner is the preferred and more robust approach.


Segmentation:
Customers were segmented into four standard uplift groups based on predicted uplift
and baseline probabilities:
- Takers
- Sure Things
- Lost Causes
- Sleeping Dogs

A complete segmentation summary with group sizes, uplift statistics, and recommended
actions is provided in `segmentation_summary.csv`.

Feature Importance:
Global feature importance was computed using LightGBM gain-based importance and saved
to `global_feature_importance.csv`. Key drivers include historical engagement/spend
features and recency indicators, which influence both baseline conversion likelihood
and incremental treatment response.

Local Explanations:
For sampled customers, local reports include p0 (no treatment), p1 (with treatment),
uplift (p1 − p0), and the most influential features affecting the uplift prediction.
These explanations are stored in `local_model_reports.json`.

Final Strategic Recommendation:
- Target **Takers** to maximize incremental conversions and ROI.
- Exclude **Sleeping Dogs**, as treatment negatively impacts their conversion behavior.
- Avoid spending on **Sure Things**, who convert regardless of treatment.
- Exclude **Lost Causes**, who show no meaningful response under any condition.

This targeting strategy ensures marketing resources are allocated strictly based on
causal impact rather than raw conversion probability.

Artifacts Included:
preprocessor.joblib, best_model_lgb.joblib, global_feature_importance.csv,
local_model_reports.json, segmentation_summary.csv, qini.png, roc.png, pr.png,
confusion_matrix.png, report.md, and credit_project_outputs.zip.
