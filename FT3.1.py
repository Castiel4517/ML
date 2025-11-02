import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score

# 机器学习模型
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#mlines

import warnings
import os
import warnings
warnings.filterwarnings("ignore")

import chardet
import json
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
)
from sklearn.inspection import permutation_importance
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

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5
OUTDIR = "./TEST/"
os.makedirs(OUTDIR, exist_ok=True)

N_TRIALS_PER_MODEL = 50         # as requested
N_JOBS_PI = 8                  # for permutation_importance (use -1 to use all cores)
TOP_K_IMPORTANCE = 10          # when too many features, plot top K

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示中文
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行

# 设置随机种子
np.random.seed(42)

print("=" * 80)
print("机器学习完整Pipeline: EDA → 预处理 → 模型对比 → 最优选择")
print("=" * 80)
      
# 1. 数据加载和准备
print("\n1. 数据加载和准备")
print("-" * 50)

# 读取数据
DATA_PATH = 'D:\\20251018ML\\1023ML\\1024d.csv'
with open(DATA_PATH, "rb") as f:
    raw = f.read()
    import chardet as _ch
    enc = _ch.detect(raw)["encoding"] or "utf-8"

df = pd.read_csv(DATA_PATH, encoding=enc)

# data = pd.read_csv(r'D:\20251018ML\1023ML\1024d.csv')
data = df.apply(pd.to_numeric, errors='coerce')

print(f"数据形状: {data.shape}")
print(data.columns)
# 定义特征和目标变量
feature_cols = list(data.columns[:-1])

#grade目标变量
main_target = data.columns[-1]
print(f"特征变量: {feature_cols}")
print(f"目标变量: {main_target}")

# ==================== EDA探索性数据分析部分 ====================
print("\n" + "=" * 80)
print("EDA 探索性数据分析")
print("=" * 80)

# 2. 数据概览
print("\n2. 数据基本信息")
print("-" * 50)

print("数据集基本信息:")
print(f"数据形状: {data.shape}")
print(f"特征数量: {len(feature_cols)}")
print(f"样本数量: {len(data)}")

# 数据类型和缺失值信息
print("\n数据类型和缺失值:")
info_df = pd.DataFrame({
'数据类型': data[feature_cols + [main_target]].dtypes,
'缺失值数量': data[feature_cols + [main_target]].isnull().sum(),
'缺失值比例(%)': (data[feature_cols + [main_target]].isnull().sum() / len(data) * 100).round(2),
'唯一值数量': data[feature_cols + [main_target]].nunique()
})
print(info_df)

# 基本统计信息
print("\n特征变量描述性统计:")
desc_stats = data[feature_cols].describe()
print(desc_stats.round(4))

# 3. 目标变量分析
print("\n3. 目标变量分析")
print("-" * 50)

# 目标变量分布
target_counts = data[main_target].value_counts()
target_props = data[main_target].value_counts(normalize=True)

print("目标变量分布:")
target_summary = pd.DataFrame({
'数量': target_counts,
'比例(%)': (target_props * 100).round(2)
})
print(target_summary)

# # 目标变量可视化
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # 柱状图
# axes[0].bar(target_counts.index, target_counts.values)
# axes[0].set_title(f'{main_target} 分布')
# axes[0].set_xlabel('类别')
# axes[0].set_ylabel('样本数量')
# for i, v in enumerate(target_counts.values):
#     axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom')

# # 饼图
# axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
# axes[1].set_title(f'{main_target} 比例分布')
# # 箱线图
# sns.boxplot(x=data[main_target], y=data[feature_cols[0]], ax=axes[2])
# axes[2].set_title(f'{feature_cols[0]} vs {main_target} 箱线图')
# axes[2].set_xlabel(main_target)
# axes[2].set_ylabel(feature_cols[0])
# plt.tight_layout()
# plt.savefig('target_variable_analysis.png', dpi=300)
#  #plt.show()

# 4. 特征变量分布分析
print("\n4. 特征变量分布分析")
print("-" * 50)

# 计算需要的子图数量
n_features = len(feature_cols)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

# # 特征分布直方图
# print("绘制特征分布直方图...")
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

# # 保证 axes 一维化
# if n_rows > 1 or n_cols > 1:
#     axes = axes.ravel()
# else:
#     axes = [axes]

# for i, col in enumerate(feature_cols):
#     if i < len(axes):
#         # 直方图
#         axes[i].hist(data[col].dropna(), bins=30, density=True,
#                      alpha=0.7, edgecolor='black')

#         # 添加 KDE 曲线
#         try:
#             data[col].dropna().plot.density(ax=axes[i], linewidth=2)
#         except Exception:
#             pass

#         axes[i].set_title(f'{col} 分布')
#         axes[i].set_xlabel(col)
#         axes[i].set_ylabel('密度')
#         axes[i].grid(True, alpha=0.3)

#         # 添加统计信息
#         mean_val = data[col].mean()
#         median_val = data[col].median()
#         axes[i].axvline(mean_val, linestyle='--', alpha=0.7,
#                         label=f'均值: {mean_val:.2f}')
#         axes[i].axvline(median_val, linestyle='--', alpha=0.7,
#                         label=f'中位数: {median_val:.2f}')
#         axes[i].legend(fontsize=8)

# # 隐藏多余的子图
# for j in range(i + 1, len(axes)):
#     axes[j].set_visible(False)

# plt.tight_layout()
# plt.savefig('feature_distribution_analysis.png', dpi=300)
#  #plt.show()

# 5. 箱线图分析
print("\n5. 箱线图分析（异常值检测）")
print("-" * 50)

print("绘制箱线图分析异常值...")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

# 展平 axes 为一维
if n_rows > 1 or n_cols > 1:
    axes = axes.ravel()
else:
    axes = [axes]

outlier_summary = {}

for i, col in enumerate(feature_cols):
    if i < len(axes):
        # 箱线图
        box_plot = axes[i].boxplot(data[col].dropna(), patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')

        axes[i].set_title(f'{col} 箱线图')
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)

        # 计算异常值
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        outlier_count = len(outliers)
        outlier_percent = (outlier_count / len(data)) * 100

        outlier_summary[col] = {
            'count': outlier_count,
            'percentage': outlier_percent,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

        # 添加异常值信息
        axes[i].text(
            0.02, 0.98,
            f'异常值: {outlier_count} ({outlier_percent:.1f}%)',
            transform=axes[i].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

# 隐藏多余的子图
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('boxplot_outlier_analysis.png', dpi=300)
 #plt.show()

# 异常值统计
print("\n异常值统计总结:")
outlier_df = pd.DataFrame(outlier_summary).T
outlier_df.columns = ['异常值数量', '异常值比例(%)', '下界', '上界']
outlier_df['异常值比例(%)'] = outlier_df['异常值比例(%)'].round(2)
print(outlier_df)


# 6. 特征相关性分析
print("\n6. 特征相关性分析")
print("-" * 50)

print("计算特征间相关性...")
correlation_matrix = data[feature_cols].corr()

# # 相关性热力图
# plt.figure(figsize=(12, 10))
# mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
# sns.heatmap(
#     correlation_matrix,
#     mask=mask,
#     annot=True,
#     cmap='coolwarm',
#     center=0,
#     square=True,
#     linewidths=0.5,
#     cbar_kws={"shrink": 0.5},
#     fmt='.3f'
# )
# plt.title('特征相关性矩阵')
# plt.tight_layout()
# plt.savefig('feature_correlation_matrix.png', dpi=300)
#  #plt.show()

# 高相关性特征对
print("\n高相关性特征对 (|r| > 0.8):")
high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
            high_corr_pairs.append({
                'feature1': correlation_matrix.columns[i],
                'feature2': correlation_matrix.columns[j],
                'correlation': corr_val
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
    print(high_corr_df)
else:
    print("没有发现高相关性特征对")

# 若有高相关特征，则对高相关特征进行处理
if high_corr_pairs:
    to_remove = set()
    for pair in high_corr_pairs:
        # 简单策略：移除相关性较高对中的第二个特征
        to_remove.add(pair['feature2'])
    print(f"\n建议移除以下高相关性特征以减少多重共线性: {to_remove}")
    feature_cols = [col for col in feature_cols if col not in to_remove]
    print(f"更新后的特征列表: {feature_cols}")
else:
    print("无需移除任何特征")


# 7. 特征与目标变量关系分析
print("\n7. 特征与目标变量关系分析")
print("-" * 50)

# 不同类别下的特征分布对比
unique_targets = data[main_target].unique()
n_targets = len(unique_targets)

# print(f"绘制不同 {main_target} 类别下的特征分布对比...")

# # 为每个特征创建分类对比图
# for idx, col in enumerate(feature_cols[:6]):  # 只显示前6个特征避免图太多
#     plt.figure(figsize=(15, 5))

#     # 小提琴图
#     plt.subplot(1, 3, 1)
#     sns.violinplot(data=data, x=main_target, y=col)
#     plt.title(f'{col} - 小提琴图')
#     plt.xticks(rotation=45)

#     # 箱线图
#     plt.subplot(1, 3, 2)
#     sns.boxplot(data=data, x=main_target, y=col)
#     plt.title(f'{col} - 箱线图对比')
#     plt.xticks(rotation=45)

#     # 直方图叠加
#     plt.subplot(1, 3, 3)
#     for target in unique_targets:
#         subset = data[data[main_target] == target][col].dropna()
#         plt.hist(subset, alpha=0.6, label=f'{target} (n={len(subset)})', bins=20)
#     plt.xlabel(col)
#     plt.ylabel('频次')
#     plt.title(f'{col} - 分布对比')
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig(f'feature_{col}_by_{main_target}.png', dpi=300)
#      #plt.show()

# 8. 特征与目标变量相关性
print("\n8. 特征与目标变量相关性")
print("-" * 50)

# 自动判断目标变量类型：若为数值型且唯一值多 -> 回归相关性；否则 -> 分类方差分析
is_numeric = data[main_target].dtype in ['int64', 'float64']
is_classification = (not is_numeric) or (len(unique_targets) <= 5)

if is_numeric and not is_classification:
    # 数值型目标变量：计算 Pearson 相关性
    target_correlation = (
        data[feature_cols + [main_target]]
        .corr()[main_target]
        .drop(main_target)
        .sort_values(key=abs, ascending=False)
    )

    print("特征与目标变量相关性:")
    print(target_correlation)

    # 可视化特征与目标变量相关性
    plt.figure(figsize=(10, 8))
    target_correlation.plot(kind='barh')
    plt.title(f'特征与 {main_target} 的相关性')
    plt.xlabel('相关系数')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'feature_target_correlation_{main_target}.png', dpi=300)
     #plt.show()

else:
    # 分类变量目标：使用方差分析 ANOVA
    from scipy import stats

    print("特征与目标变量关联性分析 (F-统计量):")
    f_stats = []
    p_values = []

    for col in feature_cols:
        groups = [data[data[main_target] == target][col].dropna() for target in unique_targets]
        # 跳过类别样本过少的情况
        if any(len(g) < 2 for g in groups):
            f_stats.append(np.nan)
            p_values.append(np.nan)
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        f_stats.append(f_stat)
        p_values.append(p_val)

    anova_results = pd.DataFrame({
        'Feature': feature_cols,
        'F_statistic': f_stats,
        'p_value': p_values,
        'significant': ['是' if p < 0.05 else '否' for p in p_values]
    }).sort_values('F_statistic', ascending=False)

    print(anova_results)

# ==================== 预处理和建模部分 ====================

# 2. 数据预处理
print("\n" + "=" * 80)
print("数据预处理")
print("=" * 80)

# 提取特征和目标
X = data[feature_cols].copy()
y = data[main_target].copy()

# 处理缺失值
print("处理缺失值...")
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"处理前缺失值: {X.isnull().sum().sum()}")
print(f"处理后缺失值: {X_filled.isnull().sum().sum()}")

# 3. 异常值处理
print("\n3. 异常值检测和处理")
print("-" * 50)


def detect_outliers_iqr(df, column):
    """使用IQR方法检测异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers, lower_bound, upper_bound


# 检测异常值
outlier_info = {}
total_outliers = pd.Series([False] * len(X_filled))

for col in feature_cols:
    outliers, lower, upper = detect_outliers_iqr(X_filled, col)
    outlier_count = outliers.sum()
    outlier_percent = (outlier_count / len(X_filled)) * 100

    outlier_info[col] = {
        'count': outlier_count,
        'percentage': outlier_percent,
        'lower_bound': lower,
        'upper_bound': upper
    }

    total_outliers = total_outliers | outliers

print("异常值统计:")
for col, info in outlier_info.items():
    print(f"{col}: {info['count']} ({info['percentage']:.2f}%)")

print(f"\n总异常值样本数: {total_outliers.sum()} ({(total_outliers.sum() / len(X_filled)) * 100:.2f}%)")

# 异常值处理策略
outlier_threshold = 0.05  # 5%阈值
if (total_outliers.sum() / len(X_filled)) > outlier_threshold:
    print("\n异常值比例较高，使用Winsorizing方法处理...")
    # Winsorizing: 将异常值替换为分位数值
    X_clean = X_filled.copy()
    y_clean = y.copy()  # 保证 y 与 X_clean 对齐
    for col in feature_cols:
        outliers, lower, upper = detect_outliers_iqr(X_filled, col)
        X_clean.loc[X_clean[col] < lower, col] = lower
        X_clean.loc[X_clean[col] > upper, col] = upper
else:
    print("\n异常值比例较低，直接移除异常值...")
    # 移除异常值
    X_clean = X_filled[~total_outliers].copy()
    y_clean = y[~total_outliers].copy()

print(f"处理后数据形状: {X_clean.shape}")

# 4. 数据标准化
print("\n4. 数据标准化")
print("-" * 50)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)

print("标准化前后对比:")
comparison_df = pd.DataFrame({
    '原始均值': X_clean.mean(),
    '原始标准差': X_clean.std(),
    '标准化后均值': X_scaled.mean(),
    '标准化后标准差': X_scaled.std()
})
print(comparison_df.round(4))


# 5. PCA降维分析
print("\n5. PCA降维分析")
print("-" * 50)

from sklearn.decomposition import PCA

# 执行PCA分析
pca_full = PCA()
pca_result = pca_full.fit_transform(X_scaled)

# 计算累积解释方差
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# 可视化PCA结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 个体解释方差
ax1.bar(range(1, len(explained_variance) + 1), explained_variance)
ax1.set_xlabel('主成分')
ax1.set_ylabel('解释方差比例')
ax1.set_title('各主成分解释方差比例')
ax1.grid(True, alpha=0.3)

# 累积解释方差
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
ax2.axhline(y=0.8, color='r', linestyle='--', label='80%')
ax2.axhline(y=0.9, color='g', linestyle='--', label='90%')
ax2.axhline(y=0.95, color='orange', linestyle='--', label='95%')
ax2.set_xlabel('主成分数量')
ax2.set_ylabel('累积解释方差比例')
ax2.set_title('累积解释方差比例')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance_analysis.png', dpi=300)
 #plt.show()

# PCA降维决策
n_features = len(feature_cols)
n_components_80 = np.where(cumulative_variance >= 0.8)[0][0] + 1
n_components_90 = np.where(cumulative_variance >= 0.9)[0][0] + 1
n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1

print(f"原始特征数: {n_features}")
print(f"解释80%方差需要: {n_components_80} 个主成分")
print(f"解释90%方差需要: {n_components_90} 个主成分")
print(f"解释95%方差需要: {n_components_95} 个主成分")

# 降维决策
use_pca = False
if n_features > 10 and n_components_90 < n_features * 0.7:
    use_pca = True
    optimal_components = n_components_90
    print(f"\n✓ 建议使用PCA降维，保留{optimal_components}个主成分")
    
    pca = PCA(n_components=optimal_components)
    X_final = pd.DataFrame(
        pca.fit_transform(X_scaled),
        columns=[f'PC{i + 1}' for i in range(optimal_components)]
    )
else:
    print(f"\n✓ 不建议使用PCA降维，保持原始特征")
    X_final = X_scaled

print(f"最终特征维度: {X_final.shape}")


# 6. 数据集划分
print("\n6. 数据集划分")
print("-" * 50)

# 确保 X_final 和 y_final 定义一致
if 'X_clean' in locals() and 'y_clean' in locals():
    X_final = X_clean
    y_final = y_clean
else:
    X_final = X
    y_final = y

# 使用分层抽样确保类别分布一致，然后再进行划分
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.2,
    random_state=42,
    stratify=y_final
)

# ✅ 锁定副本，防止后续被覆盖
X_train_main, X_test_main = X_train.copy(), X_test.copy()
y_train_main, y_test_main = y_train.copy(), y_test.copy()

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
print(f"训练集标签分布: {pd.Series(y_train).value_counts().to_dict()}")
print(f"测试集标签分布: {pd.Series(y_test).value_counts().to_dict()}")



# Split (keep as DataFrame to preserve names for permutation importance)
train_X_df, test_X_df, train_y, test_y = X_train, X_test, y_train, y_test
print("Train shape:", train_X_df.shape, "Test shape:", test_X_df.shape)

# For speed, convert to numpy where models expect it, but keep DataFrame copies for PI
train_X = train_X_df.values
test_X = test_X_df.values

# ------------------------
# Helper: build pipeline-wrapped models for scale-sensitive ones
# ------------------------
def make_scaled_pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

# ------------------------
# Optuna objective generator per model
# ------------------------
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
            svc = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, class_weight="balanced", random_state=random_state)
            model = make_scaled_pipeline(svc)
        elif model_name == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 3, 15)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            model = make_scaled_pipeline(KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=1))
        elif model_name == "LR":
            C = trial.suggest_loguniform("C", 1e-3, 10.0)
            lr = LogisticRegression(C=C, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=1000, random_state=random_state)
            model = make_scaled_pipeline(lr)
        elif model_name == "MLP":
            hidden = trial.suggest_categorical("hidden_layer_sizes", [(64,), (64,32), (128,64)])
            act = trial.suggest_categorical("activation", ["relu", "tanh"])
            alpha = trial.suggest_loguniform("alpha", 1e-6, 1e-2)
            lr_init = trial.suggest_loguniform("learning_rate_init", 1e-4, 1e-2)
            mlp = MLPClassifier(hidden_layer_sizes=hidden, activation=act, alpha=alpha, learning_rate_init=lr_init,
                                max_iter=800, solver="adam", random_state=random_state)
            model = make_scaled_pipeline(mlp)
        else:
            raise ValueError("Unknown model name")

        # CV (Stratified)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
        aucs = []
        for tr_idx, val_idx in skf.split(Xtr, ytr):
            X_tr_fold, X_val_fold = Xtr[tr_idx], Xtr[val_idx]
            y_tr_fold, y_val_fold = ytr.iloc[tr_idx], ytr.iloc[val_idx]
            model.fit(X_tr_fold, y_tr_fold)
            # predict_proba may be on pipeline
            try:
                prob = model.predict_proba(X_val_fold)[:, 1]
            except Exception:
                # fallback: decision_function -> sigmoid
                if hasattr(model, "decision_function"):
                    scores = model.decision_function(X_val_fold)
                    prob = 1 / (1 + np.exp(-scores))
                else:
                    prob = model.predict(X_val_fold)
            aucs.append(roc_auc_score(y_val_fold, prob))
        return np.mean(aucs)
    return objective

# ------------------------
# Run Optuna tuning per base model
# ------------------------
base_model_names = ["LGBM", "XGB", "RF", "ADB", "SVM", "KNN", "LR", "MLP"]
best_models = {}
optuna_studies = {}

for name in base_model_names:
    print(f"\n=== Tuning {name} (n_trials={N_TRIALS_PER_MODEL}) ===")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    obj = create_objective(name, train_X, train_y)
    study.optimize(obj, n_trials=N_TRIALS_PER_MODEL, show_progress_bar=True, n_jobs=1)
    print(f"Best AUC for {name}: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    # Save study
    joblib.dump(study, os.path.join(OUTDIR, f"optuna_{name}_study.pkl"))
    optuna_studies[name] = study

    # Build final model with best params (wrap scaled ones properly)
    best_p = study.best_params
    if name == "LGBM":
        model = lgb.LGBMClassifier(**{**best_p, "random_state": RANDOM_STATE})
    elif name == "XGB":
        # xgboost expects use_label_encoder param handled inside optuna choices; ensure it is set to False
        model = xgb.XGBClassifier(**{**best_p, "use_label_encoder": False, "random_state": RANDOM_STATE, "verbosity": 0})
    elif name == "RF":
        model = RandomForestClassifier(**{**best_p, "class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": 1})
    elif name == "ADB":
        model = AdaBoostClassifier(**{**best_p, "random_state": RANDOM_STATE})
    elif name == "SVM":
        svc = SVC(C=best_p.get("C",1.0), kernel=best_p.get("kernel","rbf"), gamma=best_p.get("gamma","scale"),
                  probability=True, class_weight="balanced", random_state=RANDOM_STATE)
        model = make_scaled_pipeline(svc)
    elif name == "KNN":
        knn = KNeighborsClassifier(n_neighbors=best_p.get("n_neighbors",5), weights=best_p.get("weights","distance"), n_jobs=1)
        model = make_scaled_pipeline(knn)
    elif name == "LR":
        lr = LogisticRegression(C=best_p.get("C",1.0), penalty="l2", solver="liblinear", class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        model = make_scaled_pipeline(lr)
    elif name == "MLP":
        mlp = MLPClassifier(hidden_layer_sizes=best_p.get("hidden_layer_sizes",(64,)), activation=best_p.get("activation","relu"),
                            alpha=best_p.get("alpha",1e-4), learning_rate_init=best_p.get("learning_rate_init",1e-3),
                            max_iter=800, solver="adam", random_state=RANDOM_STATE)
        model = make_scaled_pipeline(mlp)
    else:
        raise ValueError("Unknown model name")

    best_models[name] = model

# Save best_models dict keys
joblib.dump(list(best_models.keys()), os.path.join(OUTDIR, "base_model_names.pkl"))

# ------------------------
# Generate OOF meta features (retrain each base on folds and produce oof/test preds)
# ------------------------
print("\n=== Generating OOF meta features ===")
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
train_meta = np.zeros((train_X.shape[0], len(best_models)))
test_meta = np.zeros((test_X.shape[0], len(best_models)))
oof_idx_map = np.zeros(train_X.shape[0], dtype=int)

for i, (name, model) in enumerate(best_models.items()):
    print(f"\n-- Base model: {name}")
    oof_preds = np.zeros(train_X.shape[0])
    test_fold_preds = np.zeros((N_FOLDS, test_X.shape[0]))
    fold_i = 0
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_X, train_y)):
        X_tr_fold, X_val_fold = train_X[tr_idx], train_X[val_idx]
        y_tr_fold, y_val_fold = train_y.iloc[tr_idx], train_y.iloc[val_idx]
        # fit model (pipeline accepts numpy)
        model.fit(X_tr_fold, y_tr_fold)
        # predict prob
        try:
            oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
            test_fold_preds[fold_i, :] = model.predict_proba(test_X)[:, 1]
        except Exception:
            if hasattr(model, "decision_function"):
                scr = model.decision_function(X_val_fold)
                oof_preds[val_idx] = 1 / (1 + np.exp(-scr))
                test_fold_preds[fold_i, :] = 1 / (1 + np.exp(-model.decision_function(test_X)))
            else:
                oof_preds[val_idx] = model.predict(X_val_fold)
                test_fold_preds[fold_i, :] = model.predict(test_X)
        fold_i += 1

    train_meta[:, i] = oof_preds
    test_meta[:, i] = test_fold_preds.mean(axis=0)

    # Save the final model trained on entire train set
    model.fit(train_X, train_y)
    joblib.dump(model, os.path.join(OUTDIR, f"{name}_base_model.pkl"))

# Persist meta feature matrices
pd.DataFrame(train_meta, columns=list(best_models.keys())).to_csv(os.path.join(OUTDIR, "train_meta_features.csv"), index=False)
pd.DataFrame(test_meta, columns=list(best_models.keys())).to_csv(os.path.join(OUTDIR, "test_meta_features.csv"), index=False)
print("Saved train/test meta features.")

# ------------------------
# Evaluate each base on test set, compute ROC & collect for plotting
# ------------------------
print("\n=== Evaluate base models on test set and prepare ROC curves ===")
roc_items = []
base_metrics = []
for i, name in enumerate(best_models.keys()):
    probs = test_meta[:, i]
    auc = roc_auc_score(test_y, probs)
    fpr, tpr, _ = roc_curve(test_y, probs)
    roc_items.append((name, fpr, tpr, auc))

    # auto-threshold by F1
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr, best_f1 = 0.5, -1
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        f1 = f1_score(test_y, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    pred_final = (probs >= best_thr).astype(int)
    base_metrics.append({
        "Model": name,
        "Precision": precision_score(test_y, pred_final),
        "Recall": recall_score(test_y, pred_final),
        "F1": f1_score(test_y, pred_final),
        "Accuracy": accuracy_score(test_y, pred_final),
        "AUC": auc,
        "Best_Threshold": best_thr
    })
# Save base metrics
pd.DataFrame(base_metrics).sort_values("AUC", ascending=False).to_csv(os.path.join(OUTDIR, "Base_Model_Performance_ft23.csv"), index=False)
print("Saved Base_Model_Performance_ft23.csv")

# ------------------------
# Plot combined ROC
# ------------------------
plt.figure(figsize=(8, 6))
for (name, fpr, tpr, auc) in roc_items:
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Base Models ROC Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "ROC_Base_Compare.png"))
plt.close()
print("Saved ROC_Base_Compare.png")

# ------------------------
# Feature importance: built-in or permutation importance (scoring=AUC)
# ------------------------
feature_names = X_final.columns.tolist()
print("\n=== Compute and plot feature importances for each base model ===")
from math import ceil
for name in best_models.keys():
    model = joblib.load(os.path.join(OUTDIR, f"{name}_base_model.pkl"))
    print(f"\n-> Computing importance for {name} ...")
    importance_df = None

    # If model has feature_importances_ attribute (trees, adb)
    # If model is a pipeline (scaled), extract underlying estimator for checking
    est = model
    if isinstance(model, Pipeline):
        est = model.named_steps["model"]

    # Case A: tree-based with feature_importances_
    if hasattr(est, "feature_importances_"):
        try:
            imp = est.feature_importances_
            # map imp to original feature names (train_meta uses selected features; base models used selected features)
            # but base models were trained on train_X which corresponds to original X_df columns
            # We kept feature_names earlier
            imp_series = pd.Series(imp, index=feature_names).sort_values(ascending=False)
            importance_df = pd.DataFrame({"feature": imp_series.index, "importance": imp_series.values})
        except Exception as e:
            print("Could not get built-in importances for", name, e)

    # Case B: linear models with coef_
    elif hasattr(est, "coef_"):
        try:
            coef = np.abs(est.coef_).ravel()
            imp_series = pd.Series(coef, index=feature_names).sort_values(ascending=False)
            importance_df = pd.DataFrame({"feature": imp_series.index, "importance": imp_series.values})
        except Exception as e:
            print("Could not get coef_ for", name, e)

    # Case C: fallback to permutation importance using ROC AUC as scoring
    if importance_df is None:
        print("Using permutation importance (this may be slow) for", name)
        # permutation_importance expects the fitted pipeline or estimator (pipeline handles scaling)
        # compute on test set (use DataFrame to preserve columns)
        try:
            pi = permutation_importance(model, test_X_df, test_y, scoring="roc_auc", n_repeats=20, n_jobs=N_JOBS_PI, random_state=RANDOM_STATE)
            imp_means = pi.importances_mean
            imp_series = pd.Series(imp_means, index=feature_names).sort_values(ascending=False)
            importance_df = pd.DataFrame({"feature": imp_series.index, "importance": imp_series.values})
        except Exception as e:
            print("Permutation importance failed for", name, "error:", e)
            # fallback to zeros
            importance_df = pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})

    # Save importance dataframe
    importance_df_sorted = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df_sorted.to_csv(os.path.join(OUTDIR, f"importance_{name}.csv"), index=False)

    # Plot top-k
    topk = min(TOP_K_IMPORTANCE, importance_df_sorted.shape[0])
    top_df = importance_df_sorted.head(topk).iloc[::-1]  # reverse for horizontal bar plot
    plt.figure(figsize=(8, max(3, 0.20 * topk)))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Importance (higher is more important)")
    plt.title(f"{name} Feature Importance (top {topk})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"importance_{name}.png"))
    plt.close()
    print(f"Saved importance_{name}.png and importance_{name}.csv")



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

# =========================================
# 保存 Meta 模型预测结果（追加到原始数据末尾）
# =========================================
# print("\n📁 保存 Meta 模型预测结果并与原始数据合并...")

# # 读取测试集原始数据（需与 test_y 对应）
# test_data_path = os.path.join("./data", "test.csv")  # ✅ 这里可改为你的真实测试数据路径
# if os.path.exists(test_data_path):
#     test_df_original = pd.read_csv(test_data_path)
# else:
#     # 若 test 数据不是文件形式（比如已在内存中），则使用 meta_X_test 对应列名
#     test_df_original = pd.DataFrame(meta_X_test, columns=[f"feature_{i}" for i in range(meta_X_test.shape[1])])

# # 确保长度匹配
# if len(test_df_original) != len(test_y):
#     print(f"⚠️ Warning: 测试集行数({len(test_df_original)}) 与标签行数({len(test_y)}) 不匹配。尝试自动截取。")
#     min_len = min(len(test_df_original), len(test_y))
#     test_df_original = test_df_original.iloc[:min_len]
#     final_pred = final_pred[:min_len]
#     final_y_pred = final_y_pred[:min_len]
#     test_y = test_y[:min_len]

# # 拼接预测结果列
# test_df_with_pred = test_df_original.copy()
# test_df_with_pred["True_Label"] = test_y
# test_df_with_pred["Pred_Prob"] = final_pred
# test_df_with_pred["Pred_Class"] = final_y_pred
# test_df_with_pred["Error"] = (test_df_with_pred["True_Label"] != test_df_with_pred["Pred_Class"]).astype(int)

# # 保存完整预测结果
# meta_pred_full_path = os.path.join(OUTDIR, f"Meta_{best_meta_type}_Predictions_Full.csv")
# test_df_with_pred.to_csv(meta_pred_full_path, index=False, encoding="utf-8-sig")

# print(f"✅ Meta 模型预测结果（含原始特征）已保存至:\n   {meta_pred_full_path}")


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
print("\nALL DONE. Outputs are in", OUTDIR)