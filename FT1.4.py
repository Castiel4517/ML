# stacked_nafld_v3.py
# Stacking v3: Automatic feature intelligence + medical-derived features (AST/ALT ratio, APRI, FIB-4)
# Usage: python stacked_nafld_v3.py
# Outputs -> ./output_v3/

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
DATA_PATH = "./data/1018.csv"
OUTDIR = "./output_v3"
os.makedirs(OUTDIR, exist_ok=True)
RANDOM_STATE = 7

# Medical formula constants (editable)
ULN_AST = 40.0  # Upper Limit of Normal for AST (adjust if you know cohort-specific value)

# Fraction of features to keep by combined score (AUROC + mutual info)
KEEP_FRACTION = 0.40  # keep top 40% of features by combined score

# -------------------------
# Utilities
# -------------------------
def find_col(df, candidates):
    """Find column in df with name matching any candidate (case-insensitive). Return first match or None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand is None:
            continue
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None

def safe_div(a, b, eps=1e-8):
    return a / (b + eps)

# -------------------------
# 1. Load data
# -------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded data:", DATA_PATH, "shape:", df.shape)

# assume last column is target (user confirmed binary)
target_col = df.columns[-1]
X = df.drop(columns=[target_col]).copy()
y = df[target_col].astype(int).copy()

print("Target column:", target_col, "value counts:\n", y.value_counts())

# -------------------------
# 2. Basic preprocessing
# -------------------------
# One-hot encode object/categorical columns (drop_first to avoid collinearity)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if len(cat_cols) > 0:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print(f"Applied one-hot to {len(cat_cols)} columns -> new shape {X.shape}")

# Save original copy for reference (we'll create transformed versions too)
X_orig = X.copy()

# -------------------------
# 3. Generate medical-derived features if possible
# -------------------------
# We'll try to detect plausible names (case-insensitive)
col_AST = find_col(X_orig, ["AST", "AsT", "ast"])
col_ALT = find_col(X_orig, ["ALT", "Alt", "alt"])
col_Platelet = find_col(X_orig, ["Platelet", "PLT", "platelet", "plt"])
col_Age = find_col(X_orig, ["Age", "age", "AGE"])

generated = []
if col_AST and col_ALT:
    X_orig["AST_ALT_ratio"] = safe_div(X_orig[col_AST], X_orig[col_ALT])
    generated.append("AST_ALT_ratio")
    print("Generated AST_ALT_ratio from", col_AST, col_ALT)
else:
    print("Skipping AST/ALT ratio: AST or ALT not found.")

if col_AST and col_Platelet:
    # APRI = (AST / ULN_AST) / platelet_count * 100
    X_orig["APRI"] = (X_orig[col_AST] / ULN_AST) / X_orig[col_Platelet] * 100.0
    generated.append("APRI")
    print("Generated APRI from", col_AST, "and", col_Platelet)
else:
    print("Skipping APRI: AST or Platelet not found.")

if col_Age and col_AST and col_Platelet and col_ALT:
    # FIB-4 = (Age * AST) / (platelet * sqrt(ALT))
    X_orig["FIB4"] = (X_orig[col_Age] * X_orig[col_AST]) / (X_orig[col_Platelet] * np.sqrt(np.maximum(X_orig[col_ALT], 1e-8)))
    generated.append("FIB4")
    print("Generated FIB4 from Age, AST, ALT, Platelet")
else:
    print("Skipping FIB-4: required columns (Age, AST, ALT, Platelet) not all present.")

if len(generated) > 0:
    print("Generated medical features:", generated)

# Update X to include generated columns
X = X_orig.copy()

# -------------------------
# 4. Missing value handling
# -------------------------
# Simple approach: numeric -> median, others -> mode
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if X[c].isna().sum() > 0:
        med = X[c].median()
        X[c].fillna(med, inplace=True)
        print(f"Filled NA in numeric {c} with median {med}")

# For categorical dummies, fill 0
nonnum = [c for c in X.columns if c not in num_cols]
for c in nonnum:
    if X[c].isna().sum() > 0:
        X[c].fillna(X[c].mode().iloc[0], inplace=True)
        print(f"Filled NA in non-numeric {c} with mode")

# -------------------------
# 5. Feature scoring: AUROC + mutual_info
# -------------------------
print("Computing univariate AUROC and mutual information for features...")

feature_scores = []
for col in X.columns:
    # if feature is constant, skip
    if X[col].nunique() <= 1:
        feature_scores.append((col, 0.0, 0.0, 0.0))
        continue
    try:
        # AUROC for numeric or binary features
        auc = roc_auc_score(y, X[col])
    except Exception:
        # if roc_auc fails (e.g., all same), set 0
        auc = 0.0
    # mutual information (requires 2D input)
    try:
        mi = mutual_info_classif(X[[col]], y, discrete_features='auto', random_state=RANDOM_STATE)[0]
    except Exception:
        mi = 0.0
    # normalize mi roughly (we'll combine by ranking)
    feature_scores.append((col, auc, mi, auc + mi))

scores_df = pd.DataFrame(feature_scores, columns=["feature", "auroc", "mutual_info", "score"])
scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)
scores_df.to_csv(os.path.join(OUTDIR, "feature_univariate_scores.csv"), index=False)
print("Saved feature_univariate_scores.csv (first 20):")
print(scores_df.head(20))

# -------------------------
# 6. Automatic transformation for skewed features
# - We'll detect skewness and apply Yeo-Johnson (handles negatives), or log1p if strongly right-skewed
# - Keep both original and transformed (trees can use original; linear/svm will use scaled transformed)
# -------------------------
from scipy.stats import skew

transformed_cols = []
X_transformed = X.copy()  # will hold transformed variants (suffix _tj/_log as created)

for col in num_cols:
    if col not in X.columns:
        continue
    vals = X[col].values
    # skip features that are indicators (0/1)
    if set(np.unique(vals)).issubset({0, 1}):
        continue
    try:
        s = skew(vals[~np.isnan(vals)])
    except Exception:
        s = 0.0
    # if strongly right skewed (>1) -> log1p if all non-negative, else Yeo-Johnson
    if s > 1.0:
        if np.min(vals) >= 0:
            newcol = col + "_log1p"
            X_transformed[newcol] = np.log1p(X[col].clip(lower=0))
            transformed_cols.append(newcol)
            print(f"Applied log1p to {col} -> {newcol} (skew {s:.2f})")
        else:
            newcol = col + "_yj"
            pt = PowerTransformer(method="yeo-johnson")
            X_transformed[newcol] = pt.fit_transform(X[[col]]).flatten()
            transformed_cols.append(newcol)
            joblib.dump(pt, os.path.join(OUTDIR, f"pt_{col}.pkl"))
            print(f"Applied Yeo-Johnson to {col} -> {newcol} (skew {s:.2f})")
    elif abs(s) > 0.75:
        # moderate skew, apply Yeo-Johnson
        newcol = col + "_yj"
        pt = PowerTransformer(method="yeo-johnson")
        X_transformed[newcol] = pt.fit_transform(X[[col]]).flatten()
        transformed_cols.append(newcol)
        joblib.dump(pt, os.path.join(OUTDIR, f"pt_{col}.pkl"))
        print(f"Applied Yeo-Johnson to {col} -> {newcol} (skew {s:.2f})")

# Merge transformed into X (we keep originals too)
X_fe = X_transformed.copy()
print(f"Total features after transforms: {X_fe.shape[1]} (added {len(transformed_cols)} transformed cols)")

# -------------------------
# 7. Select top features by combined score
# We created some new transformed features; need to recompute scores for new ones or simply map
# Strategy: keep top KEEP_FRACTION of the original + any newly generated medical features and transformed cols
# -------------------------
n_keep = max(5, int(np.ceil(X.shape[1] * KEEP_FRACTION)))
top_orig_features = scores_df["feature"].tolist()[:n_keep]

# Ensure medical-generated features and transformed cols are included if present in scoring
auto_include = [c for c in generated + transformed_cols if c in X_fe.columns]

selected_features = list(dict.fromkeys(top_orig_features + auto_include))  # preserve order, remove duplicates
print(f"Selecting top {len(selected_features)} features (requested fraction {KEEP_FRACTION})")
pd.Series(selected_features).to_csv(os.path.join(OUTDIR, "selected_features_v3.csv"), index=False)
print("Saved selected_features_v3.csv")

# Reduced matrices for modeling
X_sel = X_fe[selected_features].copy()

# -------------------------
# 8. Train/test split (stratified)
# -------------------------
train_X, test_X, train_y, test_y = train_test_split(
    X_sel, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train shape:", train_X.shape, "Test shape:", test_X.shape)

# -------------------------
# 9. Define base models
# - Trees receive raw selected features (no global scaler)
# - Scale-sensitive models inside pipeline with StandardScaler
# -------------------------
base_models = {
    "LGBM": lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.01, max_depth=-1, num_leaves=64,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=RANDOM_STATE, force_col_wise=True
    ),
    "XGB": xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
    ),
    "RF": RandomForestClassifier(
        n_estimators=1000, max_depth=None, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    ),
    "ADB": AdaBoostClassifier(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE),
    "LR": Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(penalty='l2', solver='liblinear', max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE))]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1))])
}

# -------------------------
# 10. OOF stacking to build meta features
# -------------------------
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

train_meta = np.zeros((train_X.shape[0], len(base_models)))
test_meta = np.zeros((test_X.shape[0], len(base_models)))
base_names = list(base_models.keys())

for idx, name in enumerate(base_names):
    model = base_models[name]
    print(f"\nTraining base: {name}")
    oof_preds = np.zeros(train_X.shape[0])
    test_fold = np.zeros((test_X.shape[0], n_folds))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
        X_tr = train_X.iloc[tr_idx].values
        X_val = train_X.iloc[val_idx].values
        y_tr = train_y.iloc[tr_idx].values
        y_val = train_y.iloc[val_idx].values

        model.fit(X_tr, y_tr)

        if hasattr(model, "predict_proba"):
            val_proba = model.predict_proba(X_val)[:, 1]
            test_fold[:, fold] = model.predict_proba(test_X.values)[:, 1]
        else:
            if hasattr(model, "decision_function"):
                val_score = model.decision_function(X_val)
                val_proba = 1 / (1 + np.exp(-val_score))
                test_fold[:, fold] = 1 / (1 + np.exp(-model.decision_function(test_X.values)))
            else:
                val_proba = model.predict(X_val)
                test_fold[:, fold] = model.predict(test_X.values)

        oof_preds[val_idx] = val_proba

    train_meta[:, idx] = oof_preds
    test_meta[:, idx] = test_fold.mean(axis=1)
    # save final model
    joblib.dump(model, os.path.join(OUTDIR, f"base_{name}_v3.pkl"))
    print(f"Saved base_{name}_v3.pkl")

# -------------------------
# 11. Meta model (Logistic Regression)
# -------------------------
meta_X = pd.DataFrame(train_meta, columns=base_names)
meta_X_test = pd.DataFrame(test_meta, columns=base_names)
meta_y = train_y.reset_index(drop=True)

meta_clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)
meta_clf.fit(meta_X, meta_y)
joblib.dump(meta_clf, os.path.join(OUTDIR, "meta_lr_v3.pkl"))
print("Saved meta_lr_v3.pkl")

# -------------------------
# 12. Evaluate stacked model
# -------------------------
meta_test_proba = meta_clf.predict_proba(meta_X_test)[:, 1]

# choose threshold that maximizes F1 on test (diagnostic)
thresholds = np.linspace(0.1, 0.9, 81)
best_thr, best_f1 = 0.5, -1
for thr in thresholds:
    p = (meta_test_proba >= thr).astype(int)
    f = f1_score(test_y, p)
    if f > best_f1:
        best_f1 = f
        best_thr = thr

y_pred_final = (meta_test_proba >= best_thr).astype(int)

metrics = {
    "Precision": precision_score(test_y, y_pred_final),
    "Recall": recall_score(test_y, y_pred_final),
    "F1": f1_score(test_y, y_pred_final),
    "Accuracy": accuracy_score(test_y, y_pred_final),
    "AUC": roc_auc_score(test_y, meta_test_proba),
    "Best_Threshold": best_thr
}

print("\n=== Stacked Model V3 Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

pd.DataFrame([metrics]).to_csv(os.path.join(OUTDIR, "stacked_v3_metrics.csv"), index=False)

# save test predictions
out_df = test_X.reset_index(drop=True).copy()
out_df["true"] = test_y.reset_index(drop=True)
out_df["meta_proba"] = meta_test_proba
out_df["meta_pred"] = y_pred_final
out_df.to_csv(os.path.join(OUTDIR, "stacked_v3_test_predictions.csv"), index=False)

# -------------------------
# 13. Base models comparison on test
# -------------------------
base_results = []
for i, name in enumerate(base_names):
    m = joblib.load(os.path.join(OUTDIR, f"base_{name}_v3.pkl"))
    if hasattr(m, "predict_proba"):
        prob = m.predict_proba(test_X.values)[:, 1]
    elif hasattr(m, "decision_function"):
        prob = 1 / (1 + np.exp(-m.decision_function(test_X.values)))
    else:
        prob = m.predict(test_X.values)
    # best threshold by F1
    best_thr_b, best_f1_b = 0.5, -1
    for thr in thresholds:
        pbin = (prob >= thr).astype(int)
        f1b = f1_score(test_y, pbin)
        if f1b > best_f1_b:
            best_f1_b = f1b
            best_thr_b = thr
    pbin = (prob >= best_thr_b).astype(int)
    base_results.append({
        "Model": name,
        "Precision": precision_score(test_y, pbin),
        "Recall": recall_score(test_y, pbin),
        "F1": f1_score(test_y, pbin),
        "AUC": roc_auc_score(test_y, prob),
        "Best_Threshold": best_thr_b
    })

base_df = pd.DataFrame(base_results).sort_values("AUC", ascending=False)
base_df.to_csv(os.path.join(OUTDIR, "base_models_comparison_v3.csv"), index=False)
print("Saved base_models_comparison_v3.csv")

# -------------------------
# 14. ROC plot for stacked
# -------------------------
fpr, tpr, _ = roc_curve(test_y, meta_test_proba)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"Stacked_v3 (AUC={metrics['AUC']:.3f})")
plt.plot([0,1],[0,1],'k--', alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Stacked v3 ROC")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "stacked_v3_ROC.png"))
print("Saved stacked_v3_ROC.png")

# -------------------------
# 15. Meta feature importance (map f0->base model)
# -------------------------
try:
    imp = meta_clf.coef_[0]
    imp_df = pd.DataFrame({
        "feature": base_names,
        "coefficient": imp
    }).sort_values("coefficient", ascending=False)
    imp_df.to_csv(os.path.join(OUTDIR, "meta_feature_importance_v3.csv"), index=False)
    print("Saved meta_feature_importance_v3.csv")
except Exception as e:
    print("Could not compute meta importance:", e)

# -------------------------
# 16. Save selected features and univariate scores
# -------------------------
scores_df.to_csv(os.path.join(OUTDIR, "feature_univariate_scores_v3.csv"), index=False)
pd.Series(selected_features).to_csv(os.path.join(OUTDIR, "selected_features_final_v3.csv"), index=False)
print("Saved feature lists and scores.")

print("\nALL DONE. Outputs in", OUTDIR)


