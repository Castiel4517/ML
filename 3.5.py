# ================================================================
# FT3.3 版本：自动化机器学习管线（含完整EDA、调参、堆叠Meta、过拟合检测）
# ================================================================
import os
import warnings
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from math import ceil
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

# Base estimators
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


# -----------------------
# Global constants
# -----------------------
RANDOM_STATE = 42
N_FOLDS = 5
OUTDIR = "./1103/"
os.makedirs(OUTDIR, exist_ok=True)

N_TRIALS_PER_MODEL = 50          # optuna trials for base models (keeps same as your previous)
N_TRIALS_META = 30               # optuna trials for meta model
TOP_K_IMPORTANCE = 10
N_JOBS_PI = 8

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import logging
import warnings

# 日志配置
LOG_PATH = os.path.join(OUTDIR, "training_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler()  # 同时在控制台显示
    ]
)

# 抑制不必要的warnings输出
warnings.filterwarnings("ignore")

# 替代 print()
log = logging.getLogger(__name__)
log.info("=== 启动 FT3.4 全流程自动建模 ===")

# -----------------------
# 1. Load & prepare data
# -----------------------
def load_and_prepare_data():
    log.info("=" * 80)
    log.info("1. 数据加载与准备")
    log.info("=" * 80)
    DATA_PATH = 'D:\\20251018ML\\1023ML\\1024d.csv'  # 按你要求保留原路径

    # try to detect encoding
    import chardet
    with open(DATA_PATH, "rb") as f:
        raw = f.read()
        enc = chardet.detect(raw)["encoding"] or "utf-8"
    df = pd.read_csv(DATA_PATH, encoding=enc)
    data = df.apply(pd.to_numeric, errors='coerce')

    feature_cols = list(data.columns[:-1])
    target = data.columns[-1]
    log.info(f"数据形状: {data.shape}")
    log.info(f"特征数量: {len(feature_cols)}, 目标变量: {target}")
    return data, feature_cols, target


# -----------------------
# 2. Full EDA (kept)
# -----------------------
def perform_full_eda(data, feature_cols, target):
    log.info("\n" + "=" * 80)
    log.info("2. EDA 探索性数据分析（完整）")
    log.info("=" * 80)

    # data overview
    log.info("数据基本信息：")
    log.info(data.describe().round(4))
    log.info("\n缺失值及类型：")
    info_df = pd.DataFrame({
        'dtype': data[feature_cols + [target]].dtypes,
        'missing': data[feature_cols + [target]].isnull().sum(),
        'missing_pct': (data[feature_cols + [target]].isnull().sum() / len(data) * 100).round(2),
        'n_unique': data[feature_cols + [target]].nunique()
    })
    log.info(info_df)

    # target distribution plots (bar, pie, box)
    target_counts = data[target].value_counts()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(target_counts.index, target_counts.values)
    axes[0].set_title(f'{target} 分布')
    axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'{target} 比例分布')
    sns.boxplot(x=data[target], y=data[feature_cols[0]], ax=axes[2])
    axes[2].set_title(f'{feature_cols[0]} vs {target} 箱线图')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'target_variable_analysis.png'), dpi=300)
    plt.close()

    # features distribution (hist + kde)
    n_features = len(feature_cols)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.ravel()
    for i, col in enumerate(feature_cols):
        try:
            sns.histplot(data[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'{col} 分布')
        except Exception:
            axes[i].set_visible(False)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'feature_distribution_analysis.png'), dpi=300)
    plt.close()

    # boxplots for outliers
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.ravel()
    outlier_summary = {}
    for i, col in enumerate(feature_cols):
        try:
            box_plot = axes[i].boxplot(data[col].dropna(), patch_artist=True)
            axes[i].set_title(f'{col} 箱线图')
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower) | (data[col] > upper)][col]
            outlier_summary[col] = {'count': len(outliers), 'pct': len(outliers) / len(data) * 100}
            axes[i].text(0.02, 0.98, f'异常值: {len(outliers)} ({outlier_summary[col]["pct"]:.1f}%)',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except Exception:
            axes[i].set_visible(False)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'boxplot_outlier_analysis.png'), dpi=300)
    plt.close()

    # correlation matrix
    corr = data[feature_cols].corr()
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', cbar_kws={"shrink": 0.5})
    plt.title('特征相关性矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'feature_correlation_matrix.png'), dpi=300)
    plt.close()

    # PCA variance explained
    scaler = StandardScaler()
    X_tmp = scaler.fit_transform(data[feature_cols].fillna(0))
    pca_full = PCA()
    pca_full.fit(X_tmp)
    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.bar(range(1, len(evr) + 1), evr)
    ax1.set_title('各主成分解释方差比例')
    ax2.plot(range(1, len(cum) + 1), cum, 'bo-')
    ax2.axhline(0.8, color='r', linestyle='--', label='80%')
    ax2.axhline(0.9, color='g', linestyle='--', label='90%')
    ax2.set_title('累积解释方差比例')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'pca_variance_analysis.png'), dpi=300)
    plt.close()

    # save outlier summary
    outlier_df = pd.DataFrame(outlier_summary).T
    outlier_df.columns = ['count', 'pct']
    outlier_df.to_csv(os.path.join(OUTDIR, 'outlier_summary.csv'), index=True, encoding='utf-8-sig')
    log.info("EDA 完成，图片与摘要已保存至", OUTDIR)


# -----------------------
# 3. Preprocess (impute, outlier handling, scaling)
# -----------------------
def preprocess_data(data, feature_cols, target):
    log.info("\n" + "=" * 80)
    log.info("3. 数据预处理（缺失值、异常值、标准化）")
    log.info("=" * 80)
    X = data[feature_cols].copy()
    y = data[target].copy()

    # impute median
    imputer = SimpleImputer(strategy='median')
    X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # outlier detection by IQR. If >5% samples flagged, do winsorize (clip at 5/95), else drop
    def detect_outliers_iqr(col):
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (col < lower) | (col > upper)

    total_outliers = np.zeros(X_filled.shape[0], dtype=bool)
    for c in X_filled.columns:
        total_outliers = total_outliers | detect_outliers_iqr(X_filled[c])

    outlier_ratio = total_outliers.mean()
    log.info(f"检测到异常值比例: {outlier_ratio:.4f}")
    if outlier_ratio > 0.05:
        # winsorize at 5% and 95%
        lower_q = X_filled.quantile(0.05)
        upper_q = X_filled.quantile(0.95)
        X_clean = X_filled.clip(lower=lower_q, upper=upper_q, axis=1)
        y_clean = y.copy()
        log.info("异常值较多，已进行Winsorizing（5%-95%）")
    else:
        X_clean = X_filled.loc[~total_outliers].reset_index(drop=True)
        y_clean = y.loc[~total_outliers].reset_index(drop=True)
        log.info(f"异常值较少，已移除异常样本，移除后样本数: {len(X_clean)}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)
    return X_scaled, y_clean, scaler


# -----------------------
# Helper: pipeline wrapper for scale-sensitive models
# -----------------------
def make_scaled_pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


# -----------------------
# 4. Optuna objective for base models
# -----------------------
def create_objective(model_name, Xtr, ytr, random_state=RANDOM_STATE):
    def objective(trial):
        # build model by sampled params
        if model_name == "LGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-6, 10.0),
                "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-6, 10.0),
                "random_state": random_state
            }
            model = lgb.LGBMClassifier(**params)

        elif model_name == "XGB":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-6, 10.0),
                "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-6, 10.0),
                "use_label_encoder": False,
                "random_state": random_state,
                "verbosity": 0
            }
            model = xgb.XGBClassifier(**params)

        elif model_name == "RF":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
                "class_weight": "balanced",
                "random_state": random_state,
                "n_jobs": 1
            }
            model = RandomForestClassifier(**params)

        elif model_name == "ADB":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
                "random_state": random_state
            }
            model = AdaBoostClassifier(**params)

        elif model_name == "SVM":
            C = trial.suggest_loguniform("C", 1e-2, 10.0)
            kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            svc = SVC(C=C, kernel=kernel, gamma=gamma, probability=True,
                      class_weight="balanced", random_state=random_state)
            model = make_scaled_pipeline(svc)

        elif model_name == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 3, 15)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            model = make_scaled_pipeline(KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, n_jobs=1))

        elif model_name == "LR":
            C = trial.suggest_loguniform("C", 1e-3, 10.0)
            lr = LogisticRegression(C=C, penalty="l2", solver="liblinear",
                                    class_weight="balanced", max_iter=1000,
                                    random_state=random_state)
            model = make_scaled_pipeline(lr)

        elif model_name == "MLP":
            hidden = trial.suggest_categorical("hidden_layer_sizes", [(64,), (64, 32), (128, 64)])
            act = trial.suggest_categorical("activation", ["relu", "tanh"])
            alpha = trial.suggest_loguniform("alpha", 1e-6, 1e-2)
            lr_init = trial.suggest_loguniform("learning_rate_init", 1e-4, 1e-2)
            mlp = MLPClassifier(hidden_layer_sizes=hidden, activation=act, alpha=alpha,
                                learning_rate_init=lr_init, max_iter=800,
                                solver="adam", random_state=random_state)
            model = make_scaled_pipeline(mlp)
        else:
            raise ValueError("Unknown model name")

        # CV evaluation (AUC)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
        aucs = []
        for tr_idx, val_idx in skf.split(Xtr, ytr):
            X_tr_fold, X_val_fold = Xtr[tr_idx], Xtr[val_idx]
            y_tr_fold, y_val_fold = ytr.iloc[tr_idx], ytr.iloc[val_idx]
            model.fit(X_tr_fold, y_tr_fold)
            try:
                prob = model.predict_proba(X_val_fold)[:, 1]
            except Exception:
                if hasattr(model, "decision_function"):
                    scores = model.decision_function(X_val_fold)
                    prob = 1 / (1 + np.exp(-scores))
                else:
                    prob = model.predict(X_val_fold)
            aucs.append(roc_auc_score(y_val_fold, prob))
        return np.mean(aucs)
    return objective


# -----------------------
# 5. Tune base models with Optuna & build best_models dict
# -----------------------
def tune_and_build_base_models(X_train_np, y_train):
    base_model_names = ["LGBM", "XGB", "RF", "ADB", "SVM", "KNN", "LR", "MLP"]
    best_models = {}
    optuna_studies = {}

    for name in base_model_names:
        log.info(f"\n=== Tuning {name} (n_trials={N_TRIALS_PER_MODEL}) ===")
        study = optuna.create_study(direction="maximize")
        obj = create_objective(name, X_train_np, y_train)
        study.optimize(obj, n_trials=N_TRIALS_PER_MODEL, show_progress_bar=False)

        log.info(f"Best AUC for {name}: {study.best_value:.4f}")
        log.info(f"Best params: {study.best_params}")
        joblib.dump(study, os.path.join(OUTDIR, f"optuna_{name}_study.pkl"))
        optuna_studies[name] = study

        # build final model with best params
        bp = study.best_params
        if name == "LGBM":
            model = lgb.LGBMClassifier(**{**bp, "random_state": RANDOM_STATE})
        elif name == "XGB":
            model = xgb.XGBClassifier(**{**bp, "use_label_encoder": False,
                                         "random_state": RANDOM_STATE, "verbosity": 0})
        elif name == "RF":
            model = RandomForestClassifier(**{**bp, "random_state": RANDOM_STATE, "n_jobs": 1})
        elif name == "ADB":
            model = AdaBoostClassifier(**{**bp, "random_state": RANDOM_STATE})
        elif name == "SVM":
            svc = SVC(**{**bp, "probability": True,
                         "class_weight": "balanced", "random_state": RANDOM_STATE})
            model = make_scaled_pipeline(svc)
        elif name == "KNN":
            model = make_scaled_pipeline(KNeighborsClassifier(**bp, n_jobs=1))
        elif name == "LR":
            lr = LogisticRegression(**bp, penalty="l2", solver="liblinear",
                                    class_weight="balanced", max_iter=1000,
                                    random_state=RANDOM_STATE)
            model = make_scaled_pipeline(lr)
        elif name == "MLP":
            model = make_scaled_pipeline(MLPClassifier(**bp, solver="adam",
                                                       max_iter=800, random_state=RANDOM_STATE))
        best_models[name] = model

    joblib.dump(list(best_models.keys()), os.path.join(OUTDIR, "base_model_names.pkl"))
    return best_models, optuna_studies


# -----------------------
# 6. Generate OOF meta features (train_meta, test_meta)
# -----------------------
def generate_meta_features(best_models, X_train_np, y_train, X_test_np, test_X_df, train_X_df):
    log.info("\n=== Generating OOF meta features ===")
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_train = X_train_np.shape[0]
    n_test = X_test_np.shape[0]
    n_models = len(best_models)
    train_meta = np.zeros((n_train, n_models))
    test_meta = np.zeros((n_test, n_models))

    for i, (name, model) in enumerate(best_models.items()):
        log.info(f"\n-- Base model: {name}")
        oof_preds = np.zeros(n_train)
        test_fold_preds = np.zeros((N_FOLDS, n_test))
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_np, y_train)):
            X_tr_fold, X_val_fold = X_train_np[tr_idx], X_train_np[val_idx]
            y_tr_fold, y_val_fold = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model.fit(X_tr_fold, y_tr_fold)
            try:
                oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
                test_fold_preds[fold, :] = model.predict_proba(X_test_np)[:, 1]
            except Exception:
                if hasattr(model, "decision_function"):
                    scr_val = model.decision_function(X_val_fold)
                    oof_preds[val_idx] = 1 / (1 + np.exp(-scr_val))
                    test_fold_preds[fold, :] = 1 / (1 + np.exp(-model.decision_function(X_test_np)))
                else:
                    oof_preds[val_idx] = model.predict(X_val_fold)
                    test_fold_preds[fold, :] = model.predict(X_test_np)
        train_meta[:, i] = oof_preds
        test_meta[:, i] = test_fold_preds.mean(axis=0)

        # retrain model on full train for later use & save
        model.fit(X_train_np, y_train)
        joblib.dump(model, os.path.join(OUTDIR, f"{name}_base_model.pkl"))
        log.info(f"Saved {name}_base_model.pkl")

    # persist meta features
    train_meta_df = pd.DataFrame(train_meta, columns=list(best_models.keys()))
    test_meta_df = pd.DataFrame(test_meta, columns=list(best_models.keys()))
    train_meta_df.to_csv(os.path.join(OUTDIR, "train_meta_features.csv"), index=False)
    test_meta_df.to_csv(os.path.join(OUTDIR, "test_meta_features.csv"), index=False)
    log.info("Saved meta features CSVs.")
    return train_meta_df, test_meta_df


# -----------------------
# 7. Evaluate base models: compute train/test metrics for each base model
# -----------------------
def evaluate_base_models_from_meta(test_meta_df, train_meta_df, best_models, train_X_df, test_X_df, y_train, y_test):
    log.info("\n=== Evaluate base models on train & test (from meta features) ===")
    base_metrics = []
    thresholds = np.linspace(0.1, 0.9, 81)

    def best_threshold(y_true, prob):
        best_t, best_f1 = 0.5, -1
        for t in thresholds:
            f1 = f1_score(y_true, (prob >= t).astype(int))
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        return best_t

    # For each base model, load saved model and evaluate on both train and test (use saved models to ensure same behavior)
    for name in best_models.keys():
        model = joblib.load(os.path.join(OUTDIR, f"{name}_base_model.pkl"))
        # predictions using original features (train_X_df/test_X_df) for train/test metrics
        try:
            train_probs = model.predict_proba(train_X_df.values)[:, 1]
            test_probs = model.predict_proba(test_X_df.values)[:, 1]
        except Exception:
            if hasattr(model, "decision_function"):
                train_probs = 1 / (1 + np.exp(-model.decision_function(train_X_df.values)))
                test_probs = 1 / (1 + np.exp(-model.decision_function(test_X_df.values)))
            else:
                train_probs = model.predict(train_X_df.values)
                test_probs = model.predict(test_X_df.values)

        thr_train = best_threshold(y_train, train_probs)
        thr_test = best_threshold(y_test, test_probs)
        for ds_name, y_true, probs, thr in [
            ("Train", y_train, train_probs, thr_train),
            ("Test", y_test, test_probs, thr_test)
        ]:
            pred = (probs >= thr).astype(int)
            base_metrics.append({
                "Model": name,
                "Dataset": ds_name,
                "Accuracy": accuracy_score(y_true, pred),
                "Precision": precision_score(y_true, pred),
                "Recall": recall_score(y_true, pred),
                "F1": f1_score(y_true, pred),
                "AUC": roc_auc_score(y_true, probs)
            })

    df_base_metrics = pd.DataFrame(base_metrics)
    # add overfitting flag based on AUC difference
    df_base_metrics["Overfitting_Warning"] = ""
    for name in df_base_metrics["Model"].unique():
        auc_train = df_base_metrics.query("Model==@name & Dataset=='Train'")["AUC"].values[0]
        auc_test = df_base_metrics.query("Model==@name & Dataset=='Test'")["AUC"].values[0]
        if auc_train - auc_test > 0.1:
            df_base_metrics.loc[(df_base_metrics["Model"] == name) & (df_base_metrics["Dataset"] == "Test"),
                                "Overfitting_Warning"] = "⚠️可能过拟合"
    df_base_metrics.to_csv(os.path.join(OUTDIR, "Base_Model_Performance_TrainTest.csv"), index=False, encoding="utf-8-sig")
    log.info("Saved Base_Model_Performance_TrainTest.csv")
    return df_base_metrics


# -----------------------
# 8. Meta model Optuna search & final evaluation
# -----------------------
def meta_objective_factory(meta_X, meta_y, random_state=RANDOM_STATE):
    def meta_objective(trial):
        meta_type = trial.suggest_categorical("meta_type", ["LR", "LGBM", "XGB", "RF", "MLP"])
        if meta_type == "LR":
            C = trial.suggest_float("C", 0.01, 10.0, log=True)
            model = LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=500, random_state=random_state)
        elif meta_type == "LGBM":
            model = lgb.LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                random_state=random_state
            )
        elif meta_type == "XGB":
            model = xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                use_label_encoder=False,
                eval_metric="auc",
                random_state=random_state
            )
        elif meta_type == "RF":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                random_state=random_state
            )
        elif meta_type == "MLP":
            model = MLPClassifier(
                hidden_layer_sizes=trial.suggest_categorical("hidden_layer_sizes", [(64,), (64, 32), (128, 64)]),
                activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
                alpha=trial.suggest_float("alpha", 1e-4, 0.01, log=True),
                learning_rate_init=trial.suggest_float("learning_rate_init", 5e-4, 0.01, log=True),
                max_iter=800,
                random_state=random_state
            )

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
        aucs = []
        for tr_idx, val_idx in skf.split(meta_X, meta_y):
            X_tr, X_val = meta_X[tr_idx], meta_X[val_idx]
            y_tr, y_val = meta_y[tr_idx], meta_y[val_idx]
            model.fit(X_tr, y_tr)
            try:
                probs = model.predict_proba(X_val)[:, 1]
            except Exception:
                if hasattr(model, "decision_function"):
                    probs = 1 / (1 + np.exp(-model.decision_function(X_val)))
                else:
                    probs = model.predict(X_val)
            aucs.append(roc_auc_score(y_val, probs))
        return np.mean(aucs)
    return meta_objective


def tune_and_evaluate_meta(train_meta_df, test_meta_df, y_train, y_test):
    log.info("\n=== Meta 层 Optuna 搜索 ===")
    meta_X = train_meta_df.values
    meta_y = y_train.values

    study = optuna.create_study(direction="maximize")
    obj = meta_objective_factory(meta_X, meta_y)
    study.optimize(obj, n_trials=N_TRIALS_META, show_progress_bar=False)
    log.info("Meta best params:", study.best_params)
    joblib.dump(study, os.path.join(OUTDIR, "Meta_optuna_study.pkl"))

    # build final meta model
    best_meta_type = study.best_params["meta_type"]
    params = {k: v for k, v in study.best_params.items() if k != "meta_type"}
    if best_meta_type == "LR":
        final_meta_model = LogisticRegression(**params, solver="liblinear", class_weight="balanced", max_iter=800, random_state=RANDOM_STATE)
    elif best_meta_type == "LGBM":
        final_meta_model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE)
    elif best_meta_type == "XGB":
        final_meta_model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="auc", random_state=RANDOM_STATE)
    elif best_meta_type == "RF":
        final_meta_model = RandomForestClassifier(**params, random_state=RANDOM_STATE)
    elif best_meta_type == "MLP":
        final_meta_model = MLPClassifier(**params, max_iter=800, random_state=RANDOM_STATE)

    # train on full meta_X
    final_meta_model.fit(meta_X, meta_y)
    joblib.dump(final_meta_model, os.path.join(OUTDIR, f"Final_Meta_{best_meta_type}.pkl"))

    # predict on test_meta_df
    final_pred_prob = final_meta_model.predict_proba(test_meta_df.values)[:, 1]
    # auto threshold by F1 on test set (though one might prefer validation for thresholding)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1
    for thr in thresholds:
        pr = (final_pred_prob >= thr).astype(int)
        f1 = f1_score(y_test, pr)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    final_pred_class = (final_pred_prob >= best_thr).astype(int)

    final_auc = roc_auc_score(y_test, final_pred_prob)
    final_acc = accuracy_score(y_test, final_pred_class)
    final_f1 = f1_score(y_test, final_pred_class)
    final_precision = precision_score(y_test, final_pred_class)
    final_recall = recall_score(y_test, final_pred_class)

    meta_metrics = {
        "Model": f"Meta_{best_meta_type}",
        "Precision": final_precision,
        "Recall": final_recall,
        "F1-score": final_f1,
        "Accuracy": final_acc,
        "AUC": final_auc,
        "Best_Threshold": best_thr
    }
    pd.DataFrame([meta_metrics]).to_csv(os.path.join(OUTDIR, "Meta_Model_Performance_Detail.csv"), index=False, encoding="utf-8-sig")
    log.info("Saved Meta_Model_Performance_Detail.csv")

    # ROC plot
    fpr, tpr, _ = roc_curve(y_test, final_pred_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{best_meta_type} (AUC={final_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"最终堆叠模型 ({best_meta_type}) ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Final_Stacking_ROC.png"), dpi=300)
    plt.close()

    return meta_metrics, final_pred_prob, final_pred_class, final_meta_model


# -----------------------
# 9. Save final predictions appended to original test data (best-effort)
# -----------------------
def save_test_predictions_with_original(test_df_original, y_test, pred_prob, pred_class, OUTDIR, best_meta_type):
    # try to align and save
    if test_df_original is None:
        # fallback to building a DataFrame with meta features names
        test_df_original = pd.DataFrame(test_meta_df.values, columns=test_meta_df.columns)

    # ensure lengths match
    min_len = min(len(test_df_original), len(y_test), len(pred_prob), len(pred_class))
    if len(test_df_original) != min_len:
        test_df_original = test_df_original.iloc[:min_len].reset_index(drop=True)
        y_test = y_test[:min_len]
        pred_prob = pred_prob[:min_len]
        pred_class = pred_class[:min_len]

    out_df = test_df_original.copy().reset_index(drop=True)
    out_df["True_Label"] = y_test.reset_index(drop=True)
    out_df["Pred_Prob"] = pred_prob
    out_df["Pred_Class"] = pred_class
    out_df["Error"] = (out_df["True_Label"] != out_df["Pred_Class"]).astype(int)

    out_path = os.path.join(OUTDIR, f"Meta_{best_meta_type}_Predictions_Full.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"Saved meta predictions with original test data to {out_path}")
    return out_path


# -----------------------
# Main flow
# -----------------------
if __name__ == "__main__":
    data, feature_cols, target = load_and_prepare_data()
    perform_full_eda(data, feature_cols, target)
    X_scaled, y_clean, scaler = preprocess_data(data, feature_cols, target)

    # use stratified split
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.2,
                                                              random_state=RANDOM_STATE, stratify=y_clean)
    log.info(f"训练集形状: {X_train_df.shape}, 测试集形状: {X_test_df.shape}")
    # keep numpy arrays for speed where needed
    X_train_np = X_train_df.values
    X_test_np = X_test_df.values

    # 1) Tune & build base models
    best_models, optuna_studies = tune_and_build_base_models(X_train_np, y_train)

    # 2) Generate OOF meta features & save base models
    train_meta_df, test_meta_df = generate_meta_features(best_models, X_train_np, y_train, X_test_np, X_test_df, X_train_df)

    # 3) Evaluate base models using original features (train vs test) -> saves Base_Model_Performance_TrainTest.csv
    df_base_metrics = evaluate_base_models_from_meta(test_meta_df, train_meta_df, best_models, X_train_df, X_test_df, y_train, y_test)

    # 4) Meta model Optuna & evaluation (second-layer stacking)
    meta_metrics, meta_pred_prob, meta_pred_class, final_meta_model = tune_and_evaluate_meta(train_meta_df, test_meta_df, y_train, y_test)

    # 5) Save meta predictions appended to original test data (best-effort)
    # Try to find an original test file in ./data/test.csv as in your previous script; if not found, fallback to using test features
    test_data_path = os.path.join("./data", "test.csv")
    if os.path.exists(test_data_path):
        test_df_original = pd.read_csv(test_data_path)
    else:
        # use test features as representation
        test_df_original = X_test_df.reset_index(drop=True)

    # best_meta_type extraction
    try:
        best_meta_type = joblib.load(os.path.join(OUTDIR, "Meta_optuna_study.pkl")).best_params["meta_type"]
    except Exception:
        best_meta_type = meta_metrics["Model"].replace("Meta_", "")

    save_test_predictions_with_original(test_df_original, y_test.reset_index(drop=True), meta_pred_prob, meta_pred_class, OUTDIR, best_meta_type)

    print("\n====================")
    print("FT3.3 完成，所有输出保存在:", OUTDIR)
    log.info("====================")
