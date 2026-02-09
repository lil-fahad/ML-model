# -*- coding: utf-8 -*-
import os
import pathlib
import joblib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
model_path = ROOT / "models" / "hybrid_model.pkl"

m = joblib.load(model_path)
print("Loaded:", type(m))

est = m
if hasattr(m, "steps"):
    print("Pipeline steps:", [name for name, _ in m.steps])
    est = m.steps[-1][1]

print("Estimator:", type(est))
print("n_features_in_:", getattr(est, "n_features_in_", None))
print("feature_names_in_:", getattr(est, "feature_names_in_", None))
