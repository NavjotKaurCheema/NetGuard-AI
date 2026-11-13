"""
inference.py  --  UPDATED FINAL VERSION
=========================================
Automatically loads whichever features were selected by train.py
from models/feature_keys.json.
Replaces your existing inference.py completely. No other file changes needed.
"""

import os, json, threading, warnings
import numpy as np
import joblib
warnings.filterwarnings("ignore", message="X does not have valid feature names")

MODEL_DIR   = "models"
SHAP_PATH   = os.path.join(MODEL_DIR, "shap_explainer.joblib")
FEAT_PATH   = os.path.join(MODEL_DIR, "feature_keys.json")

# Original 15-feature fallback (if feature_keys.json not found)
_FALLBACK = [
    "Protocol", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Flow IAT Mean", "Fwd IAT Mean", "Bwd IAT Mean",
    "Fwd PSH Flags", "Bwd URG Flags", "Bwd UGR Flags", "Fwd UGR Flags",
]

if os.path.exists(FEAT_PATH):
    with open(FEAT_PATH) as _f:
        _data = json.load(_f)
    FEATURE_KEYS = _data["selected_features"]
    print(f"[INFO] Loaded {len(FEATURE_KEYS)} research-selected features from feature_keys.json")
else:
    FEATURE_KEYS = _FALLBACK
    print(f"[WARN] feature_keys.json not found. Using fallback {len(FEATURE_KEYS)} features.")
    print("       Run train.py first to generate research-selected features.")

PROTO_MAP = {"tcp": 0, "udp": 1, "icmp": 2}

def _encode_proto(val):
    if isinstance(val, str):
        return PROTO_MAP.get(val.lower(), 3)
    try:
        return float(val)
    except:
        return 3.0

print("[INFO] Loading models...")
try:
    rf     = joblib.load(os.path.join(MODEL_DIR, "supervised_best.joblib"))
    iso    = joblib.load(os.path.join(MODEL_DIR, "unsupervised_iso.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    print("[INFO] Models loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Could not load models: {e}. Run train.py first!")

n_features = scaler.n_features_in_
print(f"[INFO] Scaler expects {n_features} features.")

try:
    explainer = joblib.load(SHAP_PATH)
    print("[INFO] SHAP explainer loaded.")
except:
    explainer = None
    print("[WARN] SHAP explainer not found.")

_lock   = threading.Lock()
_alpha  = 0.5
_beta   = 0.4
_thresh = 0.5

def set_detector_params(alpha, beta, thresh):
    global _alpha, _beta, _thresh
    with _lock:
        _alpha = alpha; _beta = beta; _thresh = thresh
    print(f"[INFO] Params updated: alpha={alpha} beta={beta} thresh={thresh}")

def get_detector_params():
    with _lock:
        return _alpha, _beta, _thresh

def predict_flow(flow: dict) -> dict:
    row = []
    for k in FEATURE_KEYS[:n_features]:
        v = flow.get(k, 0)
        if k == "Protocol":
            row.append(_encode_proto(v))
        else:
            try:
                val = float(v)
                row.append(0.0 if (np.isnan(val) or np.isinf(val)) else val)
            except:
                row.append(0.0)

    x        = np.array(row, dtype=np.float64).reshape(1, -1)
    # Pass as DataFrame with feature names to suppress sklearn warning:
    # "X does not have valid feature names, but StandardScaler was fitted with feature names"
    import pandas as _pd
    x_df     = _pd.DataFrame(x, columns=FEATURE_KEYS[:n_features])
    x_scaled = scaler.transform(x_df)

    rf_prob   = rf.predict_proba(x_scaled)[0][1]
    iso_raw   = iso.decision_function(x_scaled)[0]
    iso_score = 1.0 / (1.0 + np.exp(iso_raw * 10))

    alpha, beta, thresh = get_detector_params()
    attack_score = alpha * rf_prob + beta * iso_score
    label        = "Attack" if attack_score >= thresh else "Benign"

    shap_vals = []
    if explainer:
        try:
            sv = explainer.shap_values(x_scaled)
            # Handle all SHAP output formats:
            # sklearn TreeExplainer: returns list [class0, class1] or 2D array
            # booster TreeExplainer (XGBoost fix): returns single 2D array
            if isinstance(sv, list):
                sv = sv[1]           # list format: use class-1
            elif hasattr(sv, "ndim") and sv.ndim == 3:
                sv = sv[:, :, 1]     # 3D format: (samples, features, classes)
            sv = sv.flatten()
            top3 = np.argsort(np.abs(sv))[::-1][:3]
            shap_vals = [
                {"feature": FEATURE_KEYS[i] if i < len(FEATURE_KEYS) else f"f{i}",
                 "value": round(float(sv[i]), 4)}
                for i in top3 if i < len(sv)
            ]
        except:
            pass

    return {"score": float(attack_score), "label": label, "shap": shap_vals}
