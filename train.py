"""
Phases in train.py:

  Phase 0: Data loading + ALL preprocessing steps from Sakib & Tabassum (2025)
           - Remove duplicates
           - Handle missing values (imputation)
           - Correlation analysis (removes r>0.95 pairs)
           - Skewness correction (log1p transform)
           - Replace infinite values

  Phase 1: Feature selection
           RF Gini + Mutual Information on all 78 features
           (Hossain et al. in Rajathi & Rukmani 2025, Ramesh IJIRT 2024)

  Phase 2: Model comparison -- ALL ML + Deep Learning models
           ML:  Decision Tree, Random Forest, Extra Trees,
                Gradient Boosting, XGBoost, LightGBM, Logistic Regression
           DL:  MLP, 1D-CNN, LSTM, GRU, BiLSTM, CNN+BiLSTM
            # MULTI-CRITERIA MODEL SELECTION
    
    Weights chosen based on IDS domain priorities:
      Recall (DR)   30% -- missing an attack is catastrophic
      F1 Score      25% -- balances precision/recall on imbalanced data
      FAR           20% -- false alarms cause alert fatigue (lower=better)
      AUC-ROC       15% -- threshold-independent discrimination ability
      Infer Speed   10% -- real-time system needs fast per-flow prediction
    
    Each metric normalised to [0, 1] before weighting.
    FAR and infer_ms are inverted (lower is better -> higher score).

  Phase 3: Final training -- winner(XGBoost provided best) + Isolation Forest + SHAP + save all

Run:
    python train.py                        # full pipeline (recommended)
    python train.py --rows 300000          # quick test
    python train.py --skip-selection       # skip Phase 1 if done before
    python train.py --skip-dl             # skip deep learning models
    python train.py --model xgboost        # force specific model, skip comparison

Outputs (models/):
    supervised_best.joblib       <- best model
    unsupervised_iso.joblib    <- Isolation Forest
    scaler.joblib              <- StandardScaler
    shap_explainer.joblib      <- SHAP (if shap compatible)
    metrics.json               <- all metrics for dashboard
    feature_keys.json          <- selected features for inference.py
    comparison_results.json    <- all model results
    comparison_plot.png        <- bar chart
    feature_importance_plot.png

Fix applied: numpy>=2 incompatibility with shap -- handled gracefully.
DL models: use PyTorch on your RTX 3050 6GB (no Colab needed).
"""

import os, json, time, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, IsolationForest
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] LightGBM not installed: pip install lightgbm")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HAS_TORCH = True
    if DEVICE.type == "cuda":
        print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] No GPU found, DL models will use CPU (slower)")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("[WARN] PyTorch not installed: pip install torch")

# ================================================================
# CONSTANTS
# ================================================================
DEFAULT_CSV = "data/CICIDS2017_subset.csv"
OUTPUT_DIR  = "models"
LABEL_COL   = "Label"
PROTO_MAP   = {"tcp": 0, "udp": 1, "icmp": 2}

# Deep learning hyperparameters (tuned for RTX 3050 6GB)
DL_BATCH_MLP  = 4096   # MLP and CNN: large batch OK, no sequence dimension
DL_BATCH_SEQ  = 512    # LSTM/GRU/BiLSTM: sequence models need smaller batch
                        # (RTX 3050 6GB OOM at 4096 -- sequence hidden states)
DL_EPOCHS     = 50
DL_LR         = 0.001
DL_PATIENCE   = 5

ALL_78 = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


# ================================================================
# PHASE 0: LOAD & PREPROCESS (Sakib & Tabassum 2025 pipeline)
# ================================================================
def encode_protocol(val):
    if isinstance(val, str):
        return PROTO_MAP.get(val.lower().strip(), 3)
    try:
        return float(val)
    except:
        return 3.0


def load_and_clean(csv_path, max_rows=None):
    print("\n" + "="*65)
    print("  PHASE 0 -- LOAD & PREPROCESS")
    print("  Pipeline: Sakib & Tabassum (2025) + Rajathi & Rukmani (2025)")
    print("="*65)
    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_rows)
    df.columns = df.columns.str.strip()
    print(f"[INFO] Raw shape: {df.shape}")

    # Step 1: Remove duplicates (Sakib & Tabassum 2025)
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[P1] Removed {before - len(df):,} duplicate rows. Now: {len(df):,}")

    # Step 2: Replace infinite values (Sakib & Tabassum 2025)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"[P2] Replaced infinite values with NaN.")

    # Step 3: Handle missing values -- impute numeric with median
    # (Sakib & Tabassum 2025 -- imputation or removal)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_missing = df[num_cols].isna().sum().sum()
    if n_missing > 0:
        for col in num_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        print(f"[P3] Imputed {n_missing:,} missing values with column medians.")
    else:
        print(f"[P3] No missing values found.")

    # Step 4: Protocol encoding
    if "Protocol" in df.columns and pd.api.types.is_string_dtype(df["Protocol"]):
        df["Protocol"] = df["Protocol"].apply(encode_protocol)

    # Step 5: Binary label
    y = df[LABEL_COL].apply(
        lambda x: 0 if str(x).strip().upper() in ("BENIGN", "NORMAL", "0") else 1
    )
    print(f"[INFO] Benign: {(y==0).sum():,} | Attacks: {(y==1).sum():,}")

    # Step 6: Skewness correction on numeric features
    # (Sakib & Tabassum 2025 -- log1p applied to highly skewed features)
    # Only applied to non-negative features with skewness > 3
    avail_feats = [f for f in ALL_78 if f in df.columns]
    X_temp = df[avail_feats].astype(np.float64)
    skew_cols = []
    for col in avail_feats:
        try:
            sk = X_temp[col].skew()
            if sk > 3.0 and X_temp[col].min() >= 0:
                df[col] = np.log1p(df[col].astype(np.float64))
                skew_cols.append(col)
        except:
            pass
    print(f"[P6] Applied log1p skewness correction to {len(skew_cols)} features.")
    if skew_cols:
        print(f"     e.g.: {skew_cols[:5]}")

    print(f"[INFO] Preprocessing complete. Final shape: {df.shape}")
    return df, y


def make_X(df, feature_keys):
    for f in feature_keys:
        if f not in df.columns:
            df[f] = 0
    return df[feature_keys].astype(np.float64).fillna(0)


# ================================================================
# PHASE 1: FEATURE SELECTION
# ================================================================
def phase1_select(df, y, top_n):
    print("\n" + "="*65)
    print(f"  PHASE 1 -- FEATURE SELECTION (top {top_n} features)")
    print("  RF Gini + Mutual Information (Hossain et al. 2025, Ramesh 2024)")
    print("="*65)

    avail = [f for f in ALL_78 if f in df.columns]
    print(f"[INFO] Starting with {len(avail)} available features")
    X_all = make_X(df, avail)

    # Stage 1: near-constant filter (std < 0.01)
    stds    = X_all.std()
    low_var = stds[stds < 0.01].index.tolist()
    s1      = [f for f in avail if f not in low_var]
    print(f"\n[S1] Near-constant removed ({len(low_var)}): {low_var}")

    # Stage 2: Pearson correlation > 0.95 (Sakib & Tabassum 2025)
    corr  = X_all[s1].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropc = [c for c in upper.columns if any(upper[c] > 0.95)]
    s2    = [f for f in s1 if f not in dropc]
    print(f"[S2] Correlated removed ({len(dropc)}): {dropc}")
    print(f"     Remaining: {len(s2)} features")

    # Stage 3: RF Gini importance
    print(f"\n[S3] Random Forest Gini importance (150 trees, 150k sample)...")
    n_s   = min(len(X_all), 150000)
    idx   = np.random.choice(len(X_all), n_s, replace=False)
    rf_s  = RandomForestClassifier(n_estimators=150, max_depth=12,
                                    random_state=42, n_jobs=-1)
    rf_s.fit(X_all[s2].iloc[idx].values, y.iloc[idx].values)
    imp   = pd.Series(rf_s.feature_importances_, index=s2).sort_values(ascending=False)
    rf_top = imp.head(top_n + 5).index.tolist()
    print("  Top 15 by RF Gini:")
    for i,(f,v) in enumerate(imp.head(15).items(), 1):
        print(f"    {i:2d}. {f:<45} {v:.5f}")

    # Stage 4: Mutual Information
    print(f"\n[S4] Mutual Information (80k sample)...")
    n_mi  = min(len(X_all), 80000)
    idx2  = np.random.choice(len(X_all), n_mi, replace=False)
    mi_sc = mutual_info_classif(X_all[s2].iloc[idx2].values,
                                 y.iloc[idx2].values, random_state=42)
    mi    = pd.Series(mi_sc, index=s2).sort_values(ascending=False)
    mi_top = mi.head(top_n + 5).index.tolist()
    print("  Top 15 by Mutual Information:")
    for i,(f,v) in enumerate(mi.head(15).items(), 1):
        c = "checkRF" if f in rf_top else "      "
        print(f"    {i:2d}. [{c}] {f:<40} {v:.4f}")

    # Final: confirmed by BOTH, then fill from MI if needed
    both  = [f for f in rf_top if f in mi_top]
    extra = [f for f in mi_top if f not in both]
    final = (both + extra)[:top_n]

    print(f"\n[INFO] SELECTED {len(final)} features:")
    for i,f in enumerate(final, 1):
        rr = list(imp.index).index(f)+1 if f in imp.index else "-"
        mr = list(mi.index).index(f)+1  if f in mi.index  else "-"
        print(f"  {i:2d}. {f:<45} [RF#{rr} MI#{mr}]")

    _plot_features(imp, mi, final)
    data = {"selected_features": final, "n_features": len(final),
            "removed_constant": low_var, "removed_correlated": dropc,
            "method": "RF_Gini+MI",
            "papers": ["Hossain et al. in Rajathi Rukmani (2025)",
                       "Ramesh IJIRT (2024)", "Sakib & Tabassum (2025)"]}
    with open(os.path.join(OUTPUT_DIR,"feature_keys.json"),"w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved: models/feature_keys.json")
    return final, imp, mi


def _plot_features(imp, mi, final):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Feature Selection -- RF Gini + Mutual Information\n"
                 "Coloured = selected", fontsize=12, fontweight="bold")
    for ax, series, title, sc, oc in [
        (axes[0], imp.head(30), "RF Gini Importance", "#2563EB", "#CBD5E1"),
        (axes[1], mi.head(30),  "Mutual Information",  "#16A34A", "#CBD5E1"),
    ]:
        cols = [sc if f in final else oc for f in series.index]
        ax.barh(range(len(series)), series.values[::-1], color=cols[::-1])
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels([f[:38] for f in series.index[::-1]], fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR,"feature_importance_plot.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[INFO] Feature plot saved: {p}")


# ================================================================
# DEEP LEARNING MODELS (PyTorch)
# RTX 3050 6GB handles all these -- no Colab needed
# Architecture from Sakib & Tabassum (2025)
# ================================================================
class MLPNet(nn.Module):
    """MLP -- best DL performer in Sakib & Tabassum (2025): 99.97% accuracy"""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)


class CNN1DNet(nn.Module):
    """1D CNN -- treats features as sequence"""
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*4, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        return self.fc(x.view(x.size(0), -1)).squeeze(1)


class LSTMNet(nn.Module):
    """LSTM -- each feature treated as a timestep.
    hidden=64 (reduced from 128) to fit RTX 3050 6GB with 30-feature sequences."""
    def __init__(self, d):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=False)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(2))
        return self.fc(out[:,-1,:]).squeeze(1)


class GRUNet(nn.Module):
    """GRU -- lighter than LSTM, similar performance.
    hidden=64 (reduced from 128) to fit RTX 3050 6GB."""
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(1, 64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc  = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(2))
        return self.fc(out[:,-1,:]).squeeze(1)


class BiLSTMNet(nn.Module):
    """Bidirectional LSTM -- best for DoS in Sakib & Tabassum (2025).
    hidden=64 per direction (128 total) to fit RTX 3050 6GB."""
    def __init__(self, d):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.fc   = nn.Linear(128, 1)   # 64*2 bidirectional
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(2))
        return self.fc(out[:,-1,:]).squeeze(1)


class CNNBiLSTMNet(nn.Module):
    """CNN + BiLSTM -- best for Web Attacks in Sakib & Tabassum (2025).
    CNN channels=32, LSTM hidden=64 per direction to fit RTX 3050 6GB."""
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU()
        )
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.fc   = nn.Linear(128, 1)   # 64*2 bidirectional
    def forward(self, x):
        c = self.conv(x.unsqueeze(1)).permute(0,2,1)
        out, _ = self.lstm(c)
        return self.fc(out[:,-1,:]).squeeze(1)


def _train_torch(model, X_tr, y_tr, batch_size=None):
    """Train any PyTorch model with class-weighted BCE + early stopping.
    batch_size: use DL_BATCH_MLP for MLP/CNN, DL_BATCH_SEQ for LSTM/GRU/BiLSTM.
    """
    if batch_size is None:
        batch_size = DL_BATCH_MLP
    model = model.to(DEVICE)
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt       = optim.Adam(model.parameters(), lr=DL_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)

    split     = int(len(X_tr) * 0.9)
    X_t, X_v  = X_tr[:split], X_tr[split:]
    y_t, y_v  = y_tr[:split], y_tr[split:]
    loader    = DataLoader(
        TensorDataset(torch.FloatTensor(X_t), torch.FloatTensor(y_t)),
        batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type=="cuda")
    )
    # Validation DataLoader -- NEVER load full val set to GPU at once.
    # For sequence models (LSTM/GRU/BiLSTM) on 200k rows, this causes OOM.
    # Solution: run validation in the same batch_size as training.
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_v), torch.FloatTensor(y_v)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    best_loss, pat, best_state = float("inf"), 0, None
    for epoch in range(DL_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # Batched validation -- no OOM regardless of model or dataset size
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xvb, yvb in val_loader:
                xvb, yvb = xvb.to(DEVICE), yvb.to(DEVICE)
                val_losses.append(criterion(model(xvb), yvb).item())
        vl = float(np.mean(val_losses))
        scheduler.step(vl)
        if vl < best_loss:
            best_loss = vl; pat = 0
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            pat += 1
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{DL_EPOCHS}  val_loss={vl:.4f}")
        if pat >= DL_PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def _predict_torch(model, X_te, threshold=0.5):
    model.eval()
    probs = []
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_te)),
        batch_size=8192, shuffle=False
    )
    with torch.no_grad():
        for (xb,) in loader:
            p = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            probs.extend(p)
    probs = np.array(probs)
    return (probs >= threshold).astype(int), probs


class TorchWrapper:
    """Sklearn-style wrapper for PyTorch models."""
    def __init__(self, cls, d, is_seq=False):
        self.cls    = cls
        self.d      = d
        self.model  = None
        self.is_seq = is_seq   # True for LSTM/GRU/BiLSTM -- uses smaller batch
    def fit(self, X, y):
        self.model = self.cls(self.d)
        batch = DL_BATCH_SEQ if self.is_seq else DL_BATCH_MLP
        self.model = _train_torch(self.model, X, y, batch)
        return self
    def predict(self, X):
        preds, _ = _predict_torch(self.model, X); return preds
    def predict_proba(self, X):
        _, p = _predict_torch(self.model, X)
        return np.column_stack([1-p, p])


# ================================================================
# PHASE 2: MODEL COMPARISON
# ================================================================
def phase2_compare(X_tr, X_te, y_tr, y_te, feature_keys, skip_dl=False):
    print("\n" + "="*65)
    print(f"  PHASE 2 -- MODEL COMPARISON ({len(feature_keys)} features)")
    print("="*65)
    d = len(feature_keys)

    # ML models
    catalogue = {
        "Decision Tree":       DecisionTreeClassifier(random_state=42, max_depth=20),
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1,
                                                   random_state=42, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42,
                                                       n_jobs=-1, class_weight="balanced"),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=200, random_state=42,
                                                     n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                           learning_rate=0.1, subsample=0.8,
                                                           random_state=42),
    }
    if HAS_XGB:
        catalogue["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=4, random_state=42, n_jobs=-1,
            eval_metric="logloss", verbosity=0)
    if HAS_LGBM:
        catalogue["LightGBM"] = LGBMClassifier(n_estimators=300, max_depth=6,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1)

    # DL models (RTX 3050 6GB handles all of these)
    if HAS_TORCH and not skip_dl:
        print(f"[INFO] Adding DL models. Device: {DEVICE}")
        catalogue["MLP (GPU)"]         = TorchWrapper(MLPNet,      d, is_seq=False)
        catalogue["1D-CNN (GPU)"]      = TorchWrapper(CNN1DNet,   d, is_seq=False)
        catalogue["LSTM (GPU)"]        = TorchWrapper(LSTMNet,    d, is_seq=True)
        catalogue["GRU (GPU)"]         = TorchWrapper(GRUNet,     d, is_seq=True)
        catalogue["BiLSTM (GPU)"]      = TorchWrapper(BiLSTMNet,  d, is_seq=True)
        catalogue["CNN+BiLSTM (GPU)"]  = TorchWrapper(CNNBiLSTMNet, d, is_seq=True)
    elif skip_dl:
        print("[INFO] DL models skipped (--skip-dl flag).")
    else:
        print("[WARN] PyTorch not found. DL models skipped.")

    results, fitted = [], {}
    for name, model in catalogue.items():
        print(f"\n[TRAIN] {name}...")
        t0 = time.time()
        try:
            model.fit(X_tr, y_tr)
            t_train = time.time() - t0
            t1      = time.time()
            y_pred  = model.predict(X_te)
            t_inf   = (time.time()-t1) / len(y_te) * 1000
            y_sc    = (model.predict_proba(X_te)[:,1]
                       if hasattr(model,"predict_proba")
                       else model.decision_function(X_te))
            acc  = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec  = recall_score(y_te, y_pred, zero_division=0)
            f1   = f1_score(y_te, y_pred, zero_division=0)
            auc  = float(roc_auc_score(y_te, y_sc))
            cm   = confusion_matrix(y_te, y_pred)
            tn,fp,fn,tp = cm.ravel()
            far  = fp/(fp+tn) if (fp+tn)>0 else 0
            print(f"  Acc:{acc*100:.3f}%  Prec:{prec:.4f}  Rec:{rec:.4f}  "
                  f"F1:{f1:.4f}  FAR:{far:.4f}  AUC:{auc:.4f}  "
                  f"Train:{t_train:.0f}s  Inf:{t_inf:.4f}ms")
            r = {"model":name,"accuracy":round(acc*100,4),
                 "precision":round(prec,4),"recall":round(rec,4),
                 "f1":round(f1,4),"far":round(far,4),"auc_roc":round(auc,4),
                 "train_time_s":round(t_train,1),"infer_ms":round(t_inf,5),
                 "confusion_matrix":cm.tolist(),
                 "tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}
            results.append(r); fitted[name] = model
        except Exception as e:
            import traceback
            print(f"  [ERROR] {name}: {e}")
            traceback.print_exc()
            # Free GPU memory so subsequent models can still run
            if HAS_TORCH and DEVICE is not None and DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                import gc; gc.collect()

    # ================================================================
    # MULTI-CRITERIA MODEL SELECTION
    # ================================================================
    # Research basis:
    #   "The core challenge is to maintain high detection rate while
    #    significantly reducing the false alarm rate." -- PMC 2025
    #   "No single metric should be used in isolation." -- Surgery 2023
    #   "F1 does not account for true negatives." -- Deepchecks 2025
    #   Frontiers IDS 2024: precision, recall, FAR, F1, AUC all needed
    #
    # Weights chosen based on IDS domain priorities:
    #   Recall (DR)   30% -- missing an attack is catastrophic
    #   F1 Score      25% -- balances precision/recall on imbalanced data
    #   FAR           20% -- false alarms cause alert fatigue (lower=better)
    #   AUC-ROC       15% -- threshold-independent discrimination ability
    #   Infer Speed   10% -- real-time system needs fast per-flow prediction
    #
    # Each metric normalised to [0, 1] before weighting.
    # FAR and infer_ms are inverted (lower is better -> higher score).

    W_RECALL  = 0.30
    W_F1      = 0.25
    W_FAR     = 0.20   # inverted: 1 - norm(far)
    W_AUC     = 0.15
    W_SPEED   = 0.10   # inverted: 1 - norm(infer_ms)

    # Collect raw values for normalisation
    recalls    = [r["recall"]   for r in results]
    f1s        = [r["f1"]       for r in results]
    fars       = [r["far"]      for r in results]
    aucs       = [r["auc_roc"]  for r in results]
    infer_ms   = [r["infer_ms"] for r in results]

    def norm(val, vals):
        lo, hi = min(vals), max(vals)
        return (val - lo) / (hi - lo) if hi > lo else 1.0

    for r in results:
        score = (
            W_RECALL * norm(r["recall"],   recalls)
          + W_F1     * norm(r["f1"],       f1s)
          + W_FAR    * (1 - norm(r["far"], fars))        # lower FAR is better
          + W_AUC    * norm(r["auc_roc"],  aucs)
          + W_SPEED  * (1 - norm(r["infer_ms"], infer_ms))  # faster is better
        )
        r["composite_score"] = round(score, 4)

    # Sort by composite score (descending)
    sr_composite = sorted(results, key=lambda x: x["composite_score"], reverse=True)
    # Also keep F1-sorted for the printed table
    sr = sorted(results, key=lambda x: x["f1"], reverse=True)

    # ── Print table sorted by composite score ──────────────────
    print("\n" + "="*115)
    print("  MULTI-CRITERIA MODEL SELECTION RANKING")
    print("  Weights: Recall=30% | F1=25% | FAR=20% | AUC=15% | InferSpeed=10%")
    print("="*115)
    print(f"{'#':<4}{'MODEL':<22}{'ACC%':>9}{'PREC':>8}{'REC':>8}"
          f"{'F1':>8}{'FAR%':>8}{'AUC':>8}{'INF(ms)':>9}{'COMPOSITE':>11}")
    print("="*115)
    for i,r in enumerate(sr_composite, 1):
        marker = " <<< WINNER" if i == 1 else ""
        print(f"{i:<4}{r['model']:<22}{r['accuracy']:>9.3f}"
              f"{r['precision']:>8.4f}{r['recall']:>8.4f}"
              f"{r['f1']:>8.4f}{r['far']*100:>8.4f}"
              f"{r['auc_roc']:>8.4f}{r['infer_ms']:>9.4f}"
              f"{r['composite_score']:>11.4f}{marker}")
    print("="*115)

    # ── Explain the winner ─────────────────────────────────────
    winner = sr_composite[0]
    f1_winner = sr[0]
    print(f"\n COMPOSITE WINNER : {winner['model']}")
    print(f"   Composite Score : {winner['composite_score']:.4f}")
    print(f"   Recall          : {winner['recall']:.4f}  (attacks caught)")
    print(f"   F1 Score        : {winner['f1']:.4f}")
    print(f"   FAR             : {winner['far']*100:.4f}%  (false alarms)")
    print(f"   AUC-ROC         : {winner['auc_roc']:.4f}")
    print(f"   Inference Speed : {winner['infer_ms']:.4f} ms/flow")

    if f1_winner['model'] != winner['model']:
        print(f"\n  NOTE: F1-only winner would be '{f1_winner['model']}'")
        print(f"  But composite scoring chose '{winner['model']}' because:")
        if winner['recall'] > f1_winner['recall']:
            delta = winner['recall'] - f1_winner['recall']
            print(f"    - Better recall by {delta:.4f} (catches {delta*100:.2f}% more attacks)")
        if winner['far'] < f1_winner['far']:
            delta = f1_winner['far'] - winner['far']
            print(f"    - Lower FAR by {delta*100:.4f}% (fewer false alarms)")
        if winner['infer_ms'] < f1_winner['infer_ms']:
            ratio = f1_winner['infer_ms'] / max(winner['infer_ms'], 0.0001)
            print(f"    - {ratio:.1f}x faster inference (better for real-time use)")

    # Attach composite score to all results for JSON
    for r in sr:
        cs = next((x["composite_score"] for x in results if x["model"]==r["model"]), 0)
        r["composite_score"] = cs

    with open(os.path.join(OUTPUT_DIR,"comparison_results.json"),"w") as f:
        json.dump(sr_composite, f, indent=2)
    _plot_comparison(sr_composite)

    best_name = winner["model"]
    return best_name, fitted[best_name], sr_composite


def _plot_comparison(sr):
    n  = len(sr)
    names = [r["model"] for r in sr]
    acc   = [r["accuracy"] for r in sr]
    prec  = [r["precision"]*100 for r in sr]
    rec   = [r["recall"]*100 for r in sr]
    f1    = [r["f1"]*100 for r in sr]
    times = [r["train_time_s"] for r in sr]
    x, w  = np.arange(n), 0.2

    fig, axes = plt.subplots(1, 2, figsize=(max(18, n*1.8), 8))
    fig.suptitle("Model Comparison (ML + DL) -- CIC-IDS-2017\n"
                 "Research-selected 30 features | Sorted by F1",
                 fontsize=12, fontweight="bold")
    ax = axes[0]
    for i,(dat,lbl,col) in enumerate([
        (acc,"Accuracy","#2563EB"),(prec,"Precision","#16A34A"),
        (rec,"Recall","#D97706"),(f1,"F1","#7C3AED")]):
        ax.bar(x+(i-1.5)*w, dat, w, label=lbl, color=col, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Score (%)"); ax.set_title("Classification Metrics (sorted by F1)")
    lo = min(min(acc),min(prec),min(rec),min(f1))-3
    ax.set_ylim(max(lo,0),101); ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3)

    ax2 = axes[1]
    ct  = ["#22C55E" if t<120 else "#F59E0B" if t<600 else "#EF4444" for t in times]
    b2  = ax2.bar(names, times, color=ct, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(n)); ax2.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("Training Time (s)"); ax2.set_title("Training Time (Green<2min)")
    ax2.bar_label(b2, fmt="%.0fs", fontsize=8, padding=2); ax2.grid(axis="y",alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR,"comparison_plot.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[INFO] Comparison plot saved: {p}")


# ================================================================
# PHASE 3: FINAL TRAINING & SAVE
# ================================================================
def phase3_final(best_model, best_name, X_tr, X_te, y_tr, y_te,
                  scaler, feature_keys, comp_results):
    print("\n" + "="*65)
    print(f"  PHASE 3 -- FINAL EVALUATION & SAVING")
    print(f"  Winner: {best_name}")
    print("="*65)

    y_pred  = best_model.predict(X_te)
    y_score = (best_model.predict_proba(X_te)[:,1]
               if hasattr(best_model,"predict_proba")
               else best_model.decision_function(X_te))

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_score)
    cm   = confusion_matrix(y_te, y_pred)
    tn,fp,fn,tp = cm.ravel()
    far  = fp/(fp+tn) if (fp+tn)>0 else 0

    print(f"\n  Model    : {best_name}")
    print(f"  Features : {len(feature_keys)}")
    print(f"  Accuracy : {acc*100:.4f}%")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  FAR      : {far:.4f}")
    print(f"\n{classification_report(y_te, y_pred, zero_division=0)}")

    fpr,tpr,_  = roc_curve(y_te, y_score)
    pre,rec2,_ = precision_recall_curve(y_te, y_score)

    top_features = []
    if hasattr(best_model,"feature_importances_"):
        pairs = sorted(zip(feature_keys, best_model.feature_importances_),
                       key=lambda x:x[1], reverse=True)
        top_features = [{"feature":f,"importance":float(v)} for f,v in pairs[:15]]
        print("  Top 10 feature importances:")
        for f,v in pairs[:10]:
            print(f"    {f:<45}: {v:.5f}")
    elif hasattr(best_model, "model") and best_model.model is not None:
        # DL model -- no feature importance, use zeros
        top_features = [{"feature": f, "importance": 0.0} for f in feature_keys[:10]]

    # Isolation Forest (unsupervised zero-day detector)
    print("\n[INFO] Training Isolation Forest (zero-day anomaly detection)...")
    iso = IsolationForest(n_estimators=100, contamination=0.1,
                          random_state=42, n_jobs=-1)
    iso.fit(X_tr)
    print("[INFO] Isolation Forest trained.", flush=True)

    # SHAP -- fixed for numpy>=2 compatibility
    print("\n[INFO] Attempting SHAP TreeExplainer...")
    _try_shap(best_model, feature_keys)

    # metrics.json
    metrics = {
        "model_name":  best_name,
        "n_features":  len(feature_keys),
        "accuracy":    round(acc*100, 4),
        "precision":   round(prec*100, 4),
        "recall":      round(rec*100, 4),
        "f1":          round(f1*100, 4),
        "auc_roc":     round(auc, 4),
        "far":         round(far, 4),
        "confusion_matrix": cm.tolist(),
        "roc":         {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr":          {"precision": pre.tolist(), "recall": rec2.tolist()},
        "top_features": top_features,
        "all_model_comparison": comp_results,
        "preprocessing": [
            "Removed duplicates (Sakib & Tabassum 2025)",
            "Replaced infinite values",
            "Missing value imputation with median",
            "Log1p skewness correction on highly skewed features",
            "Pearson correlation filter r>0.95"
        ]
    }
    with open(os.path.join(OUTPUT_DIR,"metrics.json"),"w") as f:
        json.dump(metrics, f)
    print("[INFO] metrics.json saved.")

    joblib.dump(scaler, os.path.join(OUTPUT_DIR,"scaler.joblib"))
    joblib.dump(iso,    os.path.join(OUTPUT_DIR,"unsupervised_iso.joblib"))

    # Save best model -- handle both sklearn and torch
    if hasattr(best_model, "model") and best_model.model is not None:
        # It is a PyTorch model -- save via joblib wrapper
        joblib.dump(best_model, os.path.join(OUTPUT_DIR,"supervised_best.joblib"))
        print(f"[INFO] PyTorch model ({best_name}) saved.")
    else:
        joblib.dump(best_model, os.path.join(OUTPUT_DIR,"supervised_best.joblib"))
        print(f"[INFO] sklearn model ({best_name}) saved.")

    print(f"[INFO] All artifacts saved to {OUTPUT_DIR}/")
    print("\n Training complete! Now run: python app.py")


def _try_shap(best_model, feature_keys):
    """
    Try to build SHAP explainer. Handles numpy>=2 incompatibility gracefully.
    Fix: pip install numpy<2 --force-reinstall  OR  pip install shap --upgrade
    """
    # Skip SHAP for DL models (TreeExplainer does not support them)
    if hasattr(best_model, "model") and best_model.model is not None:
        print("[INFO] SHAP skipped for deep learning models (use LIME instead).")
        return

    try:
        import shap

        # XGBoost-specific fix: 'could not convert string to float' error
        # occurs when XGBoost internal booster format conflicts with SHAP.
        # Solution: pass feature_perturbation="tree_path_dependent" which
        # avoids the background data conversion that triggers the string error.
        model_type = type(best_model).__name__

        if "XGB" in model_type:
            # XGBoost >= 2.0 fix: extract native booster, pass feature names.
            # shap.TreeExplainer wrapping the sklearn API breaks in newer versions
            # because the booster stores importances as formatted strings.
            # Documented fix: https://github.com/shap/shap/issues/3527
            booster = best_model.get_booster()
            booster.feature_names = [str(f) for f in feature_keys]
            explainer = shap.TreeExplainer(booster)
        else:
            explainer = shap.TreeExplainer(best_model)

        joblib.dump(explainer, os.path.join(OUTPUT_DIR, "shap_explainer.joblib"))
        print(f"[INFO] SHAP TreeExplainer saved for {model_type}.")

    except ImportError:
        print("[INFO] SHAP not installed. pip install shap")
    except Exception as e:
        msg = str(e)
        if "np.obj2sctype" in msg or "NumPy 2" in msg or "numpy" in msg.lower():
            print("[WARN] SHAP incompatible with NumPy 2.x.")
            print("  Fix: pip install numpy<2 --force-reinstall")
        else:
            print(f"[WARN] SHAP failed: {e}")
            print("  SHAP saved as None -- predictions will work without it.")


# ================================================================
# MAIN
# ================================================================
def _build_forced(name, d):
    n = name.lower()
    if n in ("xgb","xgboost"):
        return XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,
                             scale_pos_weight=4,random_state=42,n_jobs=-1,
                             eval_metric="logloss",verbosity=0), "XGBoost"
    if n in ("lgbm","lightgbm"):
        return LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,
                              is_unbalance=True,random_state=42,n_jobs=-1,
                              verbose=-1), "LightGBM"
    if n in ("et","extratrees"):
        return ExtraTreesClassifier(n_estimators=200,random_state=42,
                                    n_jobs=-1,class_weight="balanced"), "ExtraTrees"
    if n in ("mlp","mlp_gpu") and HAS_TORCH:
        return TorchWrapper(MLPNet, d), "MLP (GPU)"
    if n in ("bilstm") and HAS_TORCH:
        return TorchWrapper(BiLSTMNet, d), "BiLSTM (GPU)"
    return RandomForestClassifier(n_estimators=200,random_state=42,
                                   n_jobs=-1,class_weight="balanced"), "RandomForest"


def main(csv_path, max_rows, skip_sel, force_model, top_n, skip_dl):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)

    df, y = load_and_clean(csv_path, max_rows)

    feat_file = os.path.join(OUTPUT_DIR,"feature_keys.json")
    if skip_sel and os.path.exists(feat_file):
        with open(feat_file) as f:
            feature_keys = json.load(f)["selected_features"]
        print(f"[INFO] Loaded {len(feature_keys)} saved features.")
    else:
        feature_keys, _, _ = phase1_select(df, y, top_n)

    X = make_X(df, feature_keys)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n[INFO] Train:{len(y_tr):,}  Test:{len(y_te):,}")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    y_tr_n = np.array(y_tr)
    y_te_n = np.array(y_te)

    if force_model:
        print(f"\n[INFO] Forcing model: {force_model}. Skipping comparison.")
        m, mn = _build_forced(force_model, len(feature_keys))
        m.fit(X_tr_s, y_tr_n)
        best_name, best_model, comp_results = mn, m, []
    else:
        best_name, best_model, comp_results = phase2_compare(
            X_tr_s, X_te_s, y_tr_n, y_te_n, feature_keys, skip_dl)

    phase3_final(best_model, best_name, X_tr_s, X_te_s,
                 y_tr_n, y_te_n, scaler, feature_keys, comp_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",            default=DEFAULT_CSV)
    parser.add_argument("--rows",           type=int, default=0,
                        help="Max rows (0=all). 300000 for quick test.")
    parser.add_argument("--top-n",          type=int, default=30)
    parser.add_argument("--skip-selection", action="store_true",
                        help="Skip Phase 1 if feature_keys.json exists.")
    parser.add_argument("--skip-dl",        action="store_true",
                        help="Skip deep learning models (ML only, much faster).")
    parser.add_argument("--model",          default=None,
                        help="Force model: rf/extratrees/xgboost/lightgbm/mlp/bilstm")
    args = parser.parse_args()
    main(args.csv, args.rows if args.rows>0 else None,
         args.skip_selection, args.model, args.top_n, args.skip_dl)
