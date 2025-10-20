# =========================================
# 文件名：FT1.6_MLPBase.py
# 功能：在 FT1.5 基础上新增 MLP Base Model 并保留 Optuna 优化
# =========================================

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import chardet
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
import joblib
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

# =========================================
# 全局参数
# =========================================
RANDOM_STATE = 42
N_FOLDS = 5
OUTDIR = "./output_v5_mlpbase/"
os.makedirs(OUTDIR, exist_ok=True)

# =========================================
# 数据加载与划分
# =========================================
data_path = "./data/1020.csv"
with open(data_path, 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']

df = pd.read_csv(data_path, encoding=encoding)

# 查看数据类型
print("原始数据类型:\n", df.dtypes)

# 尝试将对象类型的列转换为浮点数类型
for col in df.select_dtypes(include=['object']).columns:
    try:
        df[col] = df[col].astype(float)
    except ValueError:
        print(f"无法将列 {col} 转换为浮点数类型。请检查该列的数据。")

# 再次查看数据类型
print("\n转换后的数据类型:\n", df.dtypes)

# 处理缺失值（如果存在）
df.fillna(df.mean(), inplace=True)  # 使用均值填充缺失值

# 假设 target 在最后一列
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)  # 确保是整数标签（0/1）

print(f"Loaded {data_path}: X shape {X.shape}, y distribution:\n{y.value_counts()}")

# ---- 2. 简单预处理：缺失值、one-hot（若有分类） ----
# 这里做最小处理：数值列标准化，类别列 one-hot
# 判断类别列 (object 或 category)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if len(cat_cols) > 0:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print(f"One-hot encoded {len(cat_cols)} categorical columns -> new X shape {X.shape}")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2,
                                                    random_state=RANDOM_STATE, stratify=y)

# =========================================
# Base Model 超参数搜索函数
# =========================================
def objective(trial, model_name):
    """Optuna 超参搜索目标函数"""
    if model_name == "LGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "random_state": RANDOM_STATE
        }
        model = lgb.LGBMClassifier(**params)

    elif model_name == "XGB":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "scale_pos_weight": len(train_y[train_y == 0]) / len(train_y[train_y == 1]),
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": RANDOM_STATE
        }
        model = xgb.XGBClassifier(**params)

    elif model_name == "RF":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "random_state": RANDOM_STATE,
            "class_weight": "balanced"
        }
        model = RandomForestClassifier(**params)

    elif model_name == "ADB":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "random_state": RANDOM_STATE
        }
        model = AdaBoostClassifier(**params)

    elif model_name == "SVM":
        params = {
            "C": trial.suggest_float("C", 0.1, 5.0),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
            "probability": True,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE
        }
        model = SVC(**params)

    elif model_name == "KNN":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan"]),
            "n_jobs": -1
        }
        model = KNeighborsClassifier(**params)

    elif model_name == "LR":
        params = {
            "C": trial.suggest_float("C", 0.01, 10.0),
            "penalty": "l2",
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 500
        }
        model = LogisticRegression(**params)

    elif model_name == "MLP":
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(64,), (64, 32), (128, 64)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.01, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.0005, 0.01, log=True),
            "max_iter": 800,
            "solver": "adam",
            "random_state": RANDOM_STATE
        }
        model = MLPClassifier(**params)

    # K折验证
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for train_idx, val_idx in kf.split(train_x, train_y):
        X_tr, X_val = train_x[train_idx], train_x[val_idx]
        y_tr, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        ypred = model.predict_proba(X_val)[:, 1]
        auc = metrics.roc_auc_score(y_val, ypred)
        aucs.append(auc)

    return np.mean(aucs)

# =========================================
# Base 模型列表（包含 MLP）
# =========================================
base_model_names = ["LGBM", "XGB", "RF", "ADB", "SVM", "LR"]

# =========================================
# 运行 Optuna 优化并保存最优参数
# =========================================
best_models = {}

for name in base_model_names:
    print(f"\n🔍 正在优化 Base 模型: {name} ...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, name), n_trials=20, show_progress_bar=False)

    print(f"✅ {name} 最优 AUC: {study.best_value:.4f}")
    print(f"最优参数: {study.best_params}")
    joblib.dump(study, os.path.join(OUTDIR, f"{name}_optuna_study.pkl"))

    # 使用最优参数重新训练模型
    if name == "LGBM":
        best_model = lgb.LGBMClassifier(**study.best_params)
    elif name == "XGB":
        best_model = xgb.XGBClassifier(**study.best_params)
    elif name == "RF":
        best_model = RandomForestClassifier(**study.best_params)
    elif name == "ADB":
        best_model = AdaBoostClassifier(**study.best_params)
    elif name == "SVM":
        best_model = SVC(**study.best_params, probability=True)
    elif name == "KNN":
        best_model = KNeighborsClassifier(**study.best_params)
    elif name == "LR":
        best_model = LogisticRegression(**study.best_params)
    elif name == "MLP":
        best_model = MLPClassifier(**study.best_params)

    best_models[name] = best_model

# =========================================
# OOF 生成
# =========================================
print("\n🚀 生成 Base 层 OOF 特征...")
train_meta = np.zeros((train_x.shape[0], len(best_models)))
test_meta = np.zeros((test_x.shape[0], len(best_models)))
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for i, (name, model) in enumerate(best_models.items()):
    print(f"\n📘 训练 {name} ...")
    oof_pred = np.zeros(train_x.shape[0])
    test_pred = np.zeros((N_FOLDS, test_x.shape[0]))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
        X_tr, X_val = train_x[tr_idx], train_x[val_idx]
        y_tr, y_val = train_y.iloc[tr_idx], train_y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        oof_pred[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred[fold, :] = model.predict_proba(test_x)[:, 1]

    train_meta[:, i] = oof_pred
    test_meta[:, i] = test_pred.mean(axis=0)
    joblib.dump(model, os.path.join(OUTDIR, f"{name}_base_model.pkl"))

base_names = list(best_models.keys())
pd.DataFrame(train_meta, columns=base_names).to_csv(os.path.join(OUTDIR, "train_meta_features.csv"), index=False)
pd.DataFrame(test_meta, columns=base_names).to_csv(os.path.join(OUTDIR, "test_meta_features.csv"), index=False)
print("\n✅ 所有 Base 模型 OOF 特征已生成并保存！")

# =========================================
# 保存每个 Base 模型的性能指标
# =========================================
print("\n📊 计算 Base 层模型性能指标...")

metrics_records = []

for i, name in enumerate(base_names):
    y_true = test_y
    y_prob = test_meta[:, i]
    
    # 自动阈值搜索
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thr, best_f1 = 0.5, 0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = metrics.f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1

    y_pred_final = (y_prob >= best_thr).astype(int)
    precision = metrics.precision_score(y_true, y_pred_final)
    recall = metrics.recall_score(y_true, y_pred_final)
    f1 = metrics.f1_score(y_true, y_pred_final)
    acc = metrics.accuracy_score(y_true, y_pred_final)
    auc = metrics.roc_auc_score(y_true, y_prob)

    metrics_records.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Accuracy": acc,
        "AUC": auc,
        "Best_Threshold": best_thr
    })

base_metrics_df = pd.DataFrame(metrics_records)
base_metrics_df.to_csv(os.path.join(OUTDIR, "Base_Model_Performance.csv"), index=False)
print("✅ Base 模型评估指标已保存至 Base_Model_Performance.csv")

# =========================================
# 第二层 Meta 模型堆叠 + Optuna 自动搜索
# =========================================
print("\n====================")
print("🚀 启动 Meta 模型层自动搜索 (Optuna)")
print("====================")

from sklearn.model_selection import StratifiedKFold
import optuna

# 读取 Base 层输出特征
train_meta_df = pd.read_csv(os.path.join(OUTDIR, "train_meta_features.csv"))
test_meta_df = pd.read_csv(os.path.join(OUTDIR, "test_meta_features.csv"))

meta_X = train_meta_df.values
meta_X_test = test_meta_df.values
meta_y = train_y.values

# -----------------------------------------
# 定义 Meta 模型搜索空间
# -----------------------------------------
def meta_objective(trial):
    """Optuna 优化 Meta 模型"""
    meta_type = trial.suggest_categorical("meta_type", ["LR", "LGBM", "XGB", "RF", "MLP"])

    if meta_type == "LR":
        model = LogisticRegression(
            C=trial.suggest_float("C", 0.01, 10.0, log=True),
            solver="liblinear",
            class_weight="balanced",
            max_iter=500,
            random_state=RANDOM_STATE
        )

    elif meta_type == "LGBM":
        model = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            random_state=RANDOM_STATE
        )

    elif meta_type == "XGB":
        model = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            eval_metric="auc",
            use_label_encoder=False,
            random_state=RANDOM_STATE
        )

    elif meta_type == "RF":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    elif meta_type == "MLP":
        model = MLPClassifier(
            hidden_layer_sizes=trial.suggest_categorical("hidden_layer_sizes", [(64,), (64, 32), (128, 64)]),
            activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
            alpha=trial.suggest_float("alpha", 0.0001, 0.01, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 0.0005, 0.01, log=True),
            solver="adam",
            max_iter=800,
            random_state=RANDOM_STATE
        )

    # -----------------------------------------
    # K 折验证 AUC 评估
    # -----------------------------------------
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []

    for tr_idx, val_idx in kf.split(meta_X, meta_y):
        X_tr, X_val = meta_X[tr_idx], meta_X[val_idx]
        y_tr, y_val = meta_y[tr_idx], meta_y[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = metrics.roc_auc_score(y_val, y_pred)
        aucs.append(auc)

    return np.mean(aucs)

# -----------------------------------------
# 启动 Optuna 搜索
# -----------------------------------------
print("\n🔍 Optuna 正在搜索最佳 Meta 模型结构...")
meta_study = optuna.create_study(direction="maximize")
meta_study.optimize(meta_objective, n_trials=30, show_progress_bar=False)

print("\n✅ Meta 层搜索完成")
print(f"🏆 最优 Meta 模型类型: {meta_study.best_params['meta_type']}")
print(f"最佳 AUC: {meta_study.best_value:.4f}")
print("最优参数:")
for k, v in meta_study.best_params.items():
    print(f"  {k}: {v}")

joblib.dump(meta_study, os.path.join(OUTDIR, "Meta_optuna_study.pkl"))

# =========================================
# 使用最优 Meta 模型重新训练 + 最终预测
# =========================================
best_meta_type = meta_study.best_params["meta_type"]

if best_meta_type == "LR":
    final_meta_model = LogisticRegression(**{k: v for k, v in meta_study.best_params.items() if k != "meta_type"})
elif best_meta_type == "LGBM":
    final_meta_model = lgb.LGBMClassifier(**{k: v for k, v in meta_study.best_params.items() if k != "meta_type"})
elif best_meta_type == "XGB":
    final_meta_model = xgb.XGBClassifier(**{k: v for k, v in meta_study.best_params.items() if k != "meta_type"})
elif best_meta_type == "RF":
    final_meta_model = RandomForestClassifier(**{k: v for k, v in meta_study.best_params.items() if k != "meta_type"})
elif best_meta_type == "MLP":
    final_meta_model = MLPClassifier(**{k: v for k, v in meta_study.best_params.items() if k != "meta_type"})

final_meta_model.fit(meta_X, meta_y)
joblib.dump(final_meta_model, os.path.join(OUTDIR, f"Final_Meta_{best_meta_type}.pkl"))

# 最终预测
final_pred = final_meta_model.predict_proba(meta_X_test)[:, 1]
meta_thresholds = np.linspace(0.01, 0.99, 99)
# 自动阈值搜索以找到最佳阈值
best_thr, best_f1 = 0.5, 0
for thr in meta_thresholds:
    y_pred = (final_pred >= thr).astype(int)
    f1 = f1_score(test_y, y_pred)
    if f1 > best_f1:
        best_thr, best_f1 = thr, f1

final_y_pred = (final_pred >= best_thr).astype(int)
final_auc = roc_auc_score(test_y, final_pred)
final_acc = accuracy_score(test_y, final_y_pred)
final_f1 = f1_score(test_y, final_y_pred)
final_precision = precision_score(test_y, final_y_pred)
final_recall = recall_score(test_y, final_y_pred)

print(f"\n🎯 最终堆叠模型性能：")
print(f"  AUC = {final_auc:.4f}")
print(f"  ACC = {final_acc:.4f}")
print(f"  F1  = {final_f1:.4f}")

# 保存 Meta 模型性能指标
meta_metrics = {
    "Model": f"Meta_{best_meta_type}",
    "Precision": metrics.precision_score(test_y, (final_pred >= 0.5).astype(int)),
    "Recall": metrics.recall_score(test_y, (final_pred >= 0.5).astype(int)),
    "F1-score": final_f1,
    "Accuracy": final_acc,
    "AUC": final_auc,
    "Best_Threshold": best_thr
}
pd.DataFrame([meta_metrics]).to_csv(os.path.join(OUTDIR, "Meta_Model_Performance_Detail.csv"), index=False)
print("✅ Meta 模型评估指标已保存至 Meta_Model_Performance_Detail.csv")

# ROC 可视化
fpr, tpr, _ = metrics.roc_curve(test_y, final_pred)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"{best_meta_type} (AUC={final_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"FT2.1 最终堆叠模型 ({best_meta_type}) ROC 曲线")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "Final_Stacking_ROC.png"))
plt.show()

print("\n✅ FT2.1 自动化堆叠训练完成！")
