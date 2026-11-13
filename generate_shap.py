"""
generate_shap.py
================
Run this ONCE to generate the missing shap_explainer.joblib.
Does NOT retrain any model -- just loads the saved XGBoost model
and builds the SHAP explainer from it.

Run:
    python generate_shap.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import joblib

MODEL_DIR   = "models"
FEAT_PATH   = os.path.join(MODEL_DIR, "feature_keys.json")
MODEL_PATH  = os.path.join(MODEL_DIR, "supervised_best.joblib")
SHAP_PATH   = os.path.join(MODEL_DIR, "shap_explainer.joblib")

# Load feature names
with open(FEAT_PATH) as f:
    feature_keys = json.load(f)["selected_features"]
print(f"[INFO] Features: {len(feature_keys)}")

# Load saved model
model = joblib.load(MODEL_PATH)
model_type = type(model).__name__
print(f"[INFO] Model loaded: {model_type}")

# Build SHAP
try:
    import shap
    print("[INFO] Building SHAP TreeExplainer...")

    if "XGB" in model_type:
        # XGBoost >= 2.0 fix: use native booster with explicit feature names
        booster = model.get_booster()
        booster.feature_names = [str(f) for f in feature_keys]
        explainer = shap.TreeExplainer(booster)
        print("[INFO] Used XGBoost native booster (XGB>=2.0 fix)")
    elif "LGBM" in model_type or "LightGBM" in model_type:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model)

    joblib.dump(explainer, SHAP_PATH)
    print(f"[INFO] Saved: {SHAP_PATH}")
    print("[INFO] Done. Restart app.py -- SHAP will now load successfully.")

except ImportError:
    print("[ERROR] SHAP not installed. Run: pip install shap")
except Exception as e:
    print(f"[ERROR] SHAP failed: {e}")
    print("  Try: pip install shap --upgrade")
