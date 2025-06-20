# 🏀 Scouting ML – Predicting Emerging Basketball Talent

A data-driven machine learning pipeline to identify emerging basketball players across seasons. This project combines unsupervised clustering, pseudo-labeling, and supervised prediction to support talent scouting and recruitment decisions.

## 🎯 Objective

Develop an automated system capable of predicting which players are most likely to "break out" in the next season, using historical data and advanced modeling techniques.

---

## 📁 Project Structure


---

## 🔁 Pipeline Overview

### Phase 0 – Data Ingestion
- Load raw Excel statistics.
- Harmonize formats and column names.
- Save cleaned datasets (`.xlsx` + `.pkl`).

### Phase 1 – Clustering & PII Computation
- Normalize player statistics.
- Compute Player Impact Index (PII), rewarding efficiency, usage, and youth.
- Apply KMeans clustering and visualize using PCA.
- Assign qualitative role-based labels to clusters.

### Phase 2 – Pseudo-Labeling (Career-Level)
- Analyze cluster progression across seasons.
- Detect emergent players based on cluster improvements, PII trend, and rookie standout performance.
- Assign `Is_Career_Emergent` label.

### Phase 3 – Supervised Dataset Construction
- Build a flat dataset (one row per player-season).
- Derive `Label_Emergent_Next_Season` for supervised learning (seasonal pseudo-label).
- Engineer features: raw stats, age, Δ trends, age flags.

### Phase 4 – Model Training
- Train models on historical data:
  - Logistic Regression (with regularization)
  - Random Forest
  - XGBoost
- Validate with metrics: precision, recall, F1-score.
- Extract feature importance for scouting interpretability.

### Phase 5 – Current Season Prediction
- Preprocess latest season data (e.g., 2024–25).
- Apply trained model to estimate `prob_emergent`.
- Apply dynamic thresholding (by age group) to flag `predicted_emergent` players.
- Visualize predictions with interactive plots.

---

## 🧠 Key Concepts

- **PII (Player Impact Index)**: Custom metric combining normalized stats and age bonus.
- **Pseudo-labels**: Generated automatically based on observed historical trajectories.
- **Dynamic thresholding**: Adjust emergence probability cutoff by age.
- **Explainability**: Outputs include probability, cluster, and role label.

---

## 📊 Outputs

- Cleaned and clustered player datasets (`.pkl`, `.xlsx`)
- HTML and PNG visualizations of:
  - Clusters (PCA)
  - Top performers by metric
  - Probabilistic predictions
- Prediction table with:
  - Player name, team, age
  - Predicted emergence probability
  - Binary prediction (Emerging / Not Emerging)

---

## 🛠 Dependencies

- Python ≥ 3.8
- pandas, numpy, scikit-learn
- plotly, matplotlib
- openpyxl, tqdm
- joblib

---

## ✍ Authors

Frullo & Pareschi  
University of Bologna – Optimization & Machine Learning Project

---

## 📌 License

MIT License – feel free to use and adapt for scouting, research, or analytics purposes.

---

