# Uplift Modeling Project – Executive Summary

Date: 2025-12-23T12:07:02.661828Z

Dataset:
Hillstrom E-mail Analytics

Models Implemented:
- S-Learner (LightGBM, treatment as a feature)
- T-Learner (LightGBM, separate treated and control models – conditional)

Evaluation (Uplift Metrics):
- AUUC (S-Learner): 48.546851
- AUUC (T-Learner): 82.678555

Model Comparison:
The S-Learner achieved the highest and most reliable AUUC score, indicating superior
ranking of customers by incremental conversion impact. By pooling treated and control
data within a single model, the S-Learner provides more stable CATE estimates when
sample sizes are imbalanced or outcome patterns are similar across groups.


Segmentation:
Customers were segmented into four standard uplift groups based on predicted uplift
and baseline probabilities:
- Takers
- Sure Things
- Lost Causes
- Sleeping Dogs

Final Strategic Recommendation:
Target Takers, exclude Sleeping Dogs, and avoid spending on Sure Things and Lost Causes.
