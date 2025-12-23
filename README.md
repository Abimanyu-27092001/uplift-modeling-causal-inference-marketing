# Uplift Modeling with Causal Inference for Marketing Interventions

## Overview

This project implements a **causal uplift modeling pipeline** to estimate **individual-level incremental treatment effects (CATE)** for a marketing intervention.  
Rather than predicting who will convert, the system identifies **who converts because of the intervention**, enabling optimal targeting decisions under limited budget constraints.

The pipeline uses **S-Learner and T-Learner uplift frameworks** with **LightGBM**, evaluated using **uplift-specific metrics (Qini / AUUC)**, and is structured following **production-grade ML engineering practices**.

---

## Problem Statement

Given historical marketing campaign data with randomized treatment assignment:

- **Treatment**: Receiving an email campaign  
- **Outcome**: Customer conversion  

The objective is to estimate the **incremental causal effect** of treatment at the individual level and translate it into an actionable targeting strategy.

Traditional classifiers fail in this setting because they optimize:

P(Y = 1 | X)

instead of the causal quantity:

Δ = P(Y = 1 | T = 1, X) − P(Y = 1 | T = 0, X)

---

## Dataset

**Hillstrom E-Mail Analytics Dataset**

A randomized marketing experiment containing:

- Customer demographics and behavioral features  
- Explicit treatment assignment (No E-Mail vs Email variants)  
- Binary conversion outcome  

The randomized design enables unbiased causal estimation when modeled correctly.

---

## Methodology

### 1. Causal Framing

- Target variable: binary conversion  
- Treatment indicator: email vs no email  
- Estimand: Conditional Average Treatment Effect (CATE)  

---

### 2. Uplift Modeling Approaches

#### S-Learner (Primary Model)

- Single model trained on pooled data  
- Treatment indicator appended as a feature  
- Estimates:
  - p1 = P(Y = 1 | T = 1, X)
  - p0 = P(Y = 1 | T = 0, X)
- Uplift = p1 − p0  

The S-Learner is selected as the **primary model** due to:
- Higher AUUC on this dataset  
- Lower variance under treatment imbalance  
- More stable CATE estimates  

---

#### T-Learner (Secondary / Conditional)

- Two independent models:
  - Treated group model
  - Control group model  
- Used only when both groups have sufficient samples  
- Automatically skipped when estimates would be unstable  

---

### 3. Feature Engineering & Preprocessing

- Explicit separation of numerical and categorical features  
- ColumnTransformer-based preprocessing:
  - Scaling for numerical features  
  - One-hot encoding for categorical features  
- All preprocessing artifacts are serialized and reusable  

---

### 4. Model Training

- LightGBM (GBDT)  
- No early stopping (intentional):
  - Prevents leakage across treatment groups  
  - Produces more stable causal estimates  
- Fixed random seeds ensure reproducibility  

---

### 5. Evaluation (Causal Metrics)

Standard classification metrics are **not used**.

Instead, evaluation relies on:

- **Qini Curve**: cumulative incremental gain  
- **AUUC**: area under the uplift curve  

These metrics measure how well the model ranks individuals by **incremental causal effect**, which is the correct optimization objective.

---

### 6. Customer Segmentation (Uplift-Based)

Customers are segmented into four standard causal groups:

| Segment | Description | Action |
|------|------------|--------|
| Takers | Positive uplift, low baseline | Target |
| Sure Things | High baseline, low uplift | Do not target |
| Lost Causes | Low baseline, low uplift | Do not target |
| Sleeping Dogs | Negative uplift | Exclude |

Segmentation is derived strictly from model outputs, not heuristics.

---

### 7. Local Causal Explanations

For sampled individuals, structured local reports are generated containing:

- p0 (no treatment probability)  
- p1 (treatment probability)  
- Individual uplift (p1 − p0)  
- Most influential features contributing to uplift  

These explanations are designed for **decision transparency**, not post-hoc storytelling.

---

## Project Structure
.
├── src/
│ ├── data.py # Data loading, treatment logic, causal splits
│ ├── features.py # Feature typing and preprocessing pipeline
│ ├── train.py # S-Learner & T-Learner training logic
│ ├── evaluate.py # Qini / AUUC computation
│ └── explain.py # Local uplift explanations
│
├── notebooks/
│ └── uplift-modeling-causal-inference-marketing.ipynb
│
├── credit_project_outputs/
│ ├── model_s_lgb.joblib
│ ├── preprocessor.joblib
│ ├── segmentation_summary.csv
│ ├── global_feature_importance.csv
│ ├── local_model_reports.json
│ ├── qini.png
│ └── report.md
│
├── requirements.txt
├── pipeline.md
└── README.md
