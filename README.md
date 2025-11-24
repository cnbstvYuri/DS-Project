# Heart Disease Streamlit Dashboard

Files:

- src/train_and_save.py : trains RandomForest and LogisticRegression including feature engineering (AgeGroup, CholCategory, AgeOver50, CSI, RiskFactorsCount).
- src/utils.py : helper functions (metrics, load test split).
- src/explainability.py : SHAP helpers.
- app.py : Streamlit dashboard with EDA, Model Performance, SHAP explainability, single.
- models/ : saved models and test split after training.
- data/ : place dataset.csv here (optional).
- requirements.txt
- Dockerfile

Quickstart:

1. Place your updated heart CSV (1025 rows) at:
   `/mnt/data/streamlit_ml_dashboard/data/dataset.csv`
   (or pass an explicit path to the training script).

2. Install dependencies:
