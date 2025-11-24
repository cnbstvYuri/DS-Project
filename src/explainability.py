"""Explainability helpers (SHAP)"""
import numpy as np
import pandas as pd

def compute_shap(model, X_sample):
    try:
        import shap
    except Exception as e:
        raise RuntimeError("SHAP is not installed. Install it to use explainability features.") from e

    # If pipeline, try to extract inner estimator
    try:
        clf = model.named_steps['model']
    except Exception:
        clf = model

    # Try TreeExplainer for tree models, otherwise Explainer
    try:
        explainer = shap.Explainer(clf, X_sample)
        shap_values = explainer(X_sample)
        return shap_values
    except Exception:
        # fallback to KernelExplainer (slower)
        explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X_sample, min(50, len(X_sample))))
        shap_values = explainer.shap_values(X_sample)
        return shap_values
