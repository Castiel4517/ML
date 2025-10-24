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

warnings.filterwarnings('ignore')

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

data = pd.read_csv(r'D:\20251018ML\1023ML\1018.csv')


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

# 目标变量可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 柱状图
axes[0].bar(target_counts.index, target_counts.values)
axes[0].set_title(f'{main_target} 分布')
axes[0].set_xlabel('类别')
axes[0].set_ylabel('样本数量')
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom')

# 饼图
axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title(f'{main_target} 比例分布')
# 箱线图
sns.boxplot(x=data[main_target], y=data[feature_cols[0]], ax=axes[2])
axes[2].set_title(f'{feature_cols[0]} vs {main_target} 箱线图')
axes[2].set_xlabel(main_target)
axes[2].set_ylabel(feature_cols[0])
plt.tight_layout()
plt.savefig('target_variable_analysis.png', dpi=300)
 #plt.show()

# 4. 特征变量分布分析
print("\n4. 特征变量分布分析")
print("-" * 50)

# 计算需要的子图数量
n_features = len(feature_cols)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

# 特征分布直方图
print("绘制特征分布直方图...")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

# 保证 axes 一维化
if n_rows > 1 or n_cols > 1:
    axes = axes.ravel()
else:
    axes = [axes]

for i, col in enumerate(feature_cols):
    if i < len(axes):
        # 直方图
        axes[i].hist(data[col].dropna(), bins=30, density=True,
                     alpha=0.7, edgecolor='black')

        # 添加 KDE 曲线
        try:
            data[col].dropna().plot.density(ax=axes[i], linewidth=2)
        except Exception:
            pass

        axes[i].set_title(f'{col} 分布')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('密度')
        axes[i].grid(True, alpha=0.3)

        # 添加统计信息
        mean_val = data[col].mean()
        median_val = data[col].median()
        axes[i].axvline(mean_val, linestyle='--', alpha=0.7,
                        label=f'均值: {mean_val:.2f}')
        axes[i].axvline(median_val, linestyle='--', alpha=0.7,
                        label=f'中位数: {median_val:.2f}')
        axes[i].legend(fontsize=8)

# 隐藏多余的子图
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('feature_distribution_analysis.png', dpi=300)
 #plt.show()

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

# 相关性热力图
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    fmt='.3f'
)
plt.title('特征相关性矩阵')
plt.tight_layout()
plt.savefig('feature_correlation_matrix.png', dpi=300)
 #plt.show()

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

print(f"绘制不同 {main_target} 类别下的特征分布对比...")

# 为每个特征创建分类对比图
for idx, col in enumerate(feature_cols[:6]):  # 只显示前6个特征避免图太多
    plt.figure(figsize=(15, 5))

    # 小提琴图
    plt.subplot(1, 3, 1)
    sns.violinplot(data=data, x=main_target, y=col)
    plt.title(f'{col} - 小提琴图')
    plt.xticks(rotation=45)

    # 箱线图
    plt.subplot(1, 3, 2)
    sns.boxplot(data=data, x=main_target, y=col)
    plt.title(f'{col} - 箱线图对比')
    plt.xticks(rotation=45)

    # 直方图叠加
    plt.subplot(1, 3, 3)
    for target in unique_targets:
        subset = data[data[main_target] == target][col].dropna()
        plt.hist(subset, alpha=0.6, label=f'{target} (n={len(subset)})', bins=20)
    plt.xlabel(col)
    plt.ylabel('频次')
    plt.title(f'{col} - 分布对比')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'feature_{col}_by_{main_target}.png', dpi=300)
     #plt.show()

# 8. 特征与目标变量相关性
print("\n8. 特征与目标变量相关性")
print("-" * 50)

# 自动判断目标变量类型：若为数值型且唯一值多 -> 回归相关性；否则 -> 分类方差分析
is_numeric = data[main_target].dtype in ['int64', 'float64']
is_classification = (not is_numeric) or (len(unique_targets) <= 10)

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


# 7. 机器学习模型定义
print("\n7. 机器学习模型定义")
print("-" * 50)

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义13种机器学习模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SGD Classifier': SGDClassifier(loss='log_loss',max_iter=1000, tol=1e-3),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'AdaBoost': AdaBoostClassifier(n_estimators=100),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100),
    'Support Vector Machine': SVC(probability=True),
    'Gaussian Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Multi-layer Perceptron': MLPClassifier(max_iter=1000),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'lightGBM': LGBMClassifier()
}

print(f"定义了{len(models)}种机器学习模型")

# 8. 模型训练和评估
print("\n8. 模型训练和评估")
print("-" * 50)

# 存储结果
results = {}
cv_scores = {}
predictions = {}

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("开始训练和评估模型...")
for name, model in models.items():
    print(f"训练 {name}...", end=' ')

    try:
        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba') and len(np.unique(y_final)) == 2:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 交叉验证
        cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

        # 存储结果
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_score.mean(),
            'cv_std': cv_score.std()
        }

        cv_scores[name] = cv_score
        predictions[name] = {'y_pred': y_pred, 'y_pred_proba': y_pred_proba}

        print("✓")

    except Exception as e:
        print(f"✗ 错误: {str(e)}")
        continue


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------- 2. 训练所有模型，保存分数 -------------------
print("\n=== 模型训练并缓存预测概率 ===")

y_test_bin = label_binarize(y_final, classes=sorted(y_final.unique()))
n_classes = y_test_bin.shape[1]

# 重新切分（保持与前面一致）
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)
y_test_bin = label_binarize(y_test, classes=sorted(y_final.unique()))


model_scores = {}  # 存AUC
model_fpr_tpr = {}  # 存曲线 (fpr, tpr)
skip_models = []  # 无法画ROC的模型

for name, model in models.items():
    try:
        model.fit(X_train, y_train)

        # 取得“连续输出”以绘制 ROC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)  # shape = (n_samples, n_classes)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            # 若 decision_function 只给 (n_samples,), 转成 (n_samples, n_classes)
            if y_score.ndim == 1:
                y_score = np.column_stack([-y_score, y_score])
        else:
            print(f"⚠️  {name} 既无 predict_proba 也无 decision_function，跳过 ROC。")
            skip_models.append(name)
            continue

        # 计算 micro-average & macro-average AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # macro
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        model_scores[name] = roc_auc
        model_fpr_tpr[name] = (fpr, tpr)
        print(f"✓ {name} - macro AUC: {roc_auc['macro']:.3f}")

    except Exception as e:
        print(f"✗ {name} 训练或预测出错: {e}")
        skip_models.append(name)
        continue

# ------------- 3. 绘制一张大图：macro-average ROC -----------------
print("\n=== 绘制 ROC 曲线 ===")
plt.figure(figsize=(10, 8))
colors = cycle(plt.cm.tab20.colors)  # 至少 20 种颜色

for (name, color) in zip(model_scores.keys(), colors):
    fpr, tpr = model_fpr_tpr[name]
    auc_val = model_scores[name]["macro"]
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        color=color,
        lw=2,
        label=f"{name} (AUC = {auc_val:.3f})"
    )

# 对角线
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Macro-Average ROC Curves (3-class, 12 Models)", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("all_models_macro_roc.png", dpi=300)
 #plt.show()

# ------------- 4. （可选）再画 micro-average -----------------
plt.figure(figsize=(10, 8))
colors = cycle(plt.cm.Dark2.colors)

for (name, color) in zip(model_scores.keys(), colors):
    fpr, tpr = model_fpr_tpr[name]
    auc_val = model_scores[name]["micro"]
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        color=color,
        lw=2,
        label=f"{name} (AUC = {auc_val:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Micro-Average ROC Curves (3-class, 12 Models)", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("all_models_micro_roc.png", dpi=300)
 #plt.show()

# ------------- 5. 简单汇总表 -----------------
print("\n=== 主要 AUC 汇总 (macro / micro) ===")
for name, scores in model_scores.items():
    print(f"{name:25s}  Macro AUC: {scores['macro']:.3f}  |  Micro AUC: {scores['micro']:.3f}")

if skip_models:
    print("\n⚠️ 以下模型因缺少连续输出而未绘制 ROC：", ", ".join(skip_models))


# 9. 结果可视化对比
print("\n9. 模型性能可视化对比")
print("-" * 50)

# 创建结果DataFrame
results_df = pd.DataFrame(results).T
print("\n模型性能对比表:")
print(results_df.round(4))

# 性能对比可视化
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']
metric_names = ['准确率', '精确率', '召回率', 'F1分数', '交叉验证均值']

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    if i < len(axes):
        data_to_plot = results_df[metric].sort_values(ascending=False)
        bars = axes[i].bar(range(len(data_to_plot)), data_to_plot.values, color='skyblue', alpha=0.8)
        axes[i].set_title(f'{name} 对比')
        axes[i].set_xticks(range(len(data_to_plot)))
        axes[i].set_xticklabels(data_to_plot.index, rotation=45, ha='right')
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)

        # 添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 交叉验证分数箱线图
if len(cv_scores) > 0:
    axes[5].boxplot([cv_scores[name] for name in results.keys()],
                    labels=[name for name in results.keys()])
    axes[5].set_title('交叉验证分数分布')
    axes[5].set_xticklabels(results.keys(), rotation=45, ha='right')
    axes[5].set_ylabel('准确率')
    axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300)
 #plt.show()


# 10. 综合排名和最优模型选择
print("\n10. 综合排名和最优模型选择")
print("-" * 50)

# 计算综合得分
weights = {
    'accuracy': 0.3,
    'precision': 0.2,
    'recall': 0.2,
    'f1_score': 0.2,
    'cv_mean': 0.1
}

results_df['综合得分'] = 0
for metric, weight in weights.items():
    # 标准化到0-1范围
    normalized = (results_df[metric] - results_df[metric].min()) / (results_df[metric].max() - results_df[metric].min())
    results_df['综合得分'] += normalized * weight

# 排序
final_ranking = results_df.sort_values('综合得分', ascending=False)

print("最终模型排名:")
print("=" * 70)
ranking_display = final_ranking[['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean', '综合得分']].round(4)

for i, (name, row) in enumerate(ranking_display.iterrows(), 1):
    print(f"{i:2d}. {name:15s} | 综合得分: {row['综合得分']:.4f} | "
          f"准确率: {row['accuracy']:.4f} | F1: {row['f1_score']:.4f} | "
          f"CV: {final_ranking.loc[name, 'cv_mean']:.4f}±{final_ranking.loc[name, 'cv_std']:.4f}")

# 选择最优模型
best_model_name = final_ranking.index[0]
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]

print(f"\n🏆 最优模型: {best_model_name}")
print(f"   综合得分: {final_ranking.iloc[0]['综合得分']:.4f}")
print(f"   准确率: {final_ranking.iloc[0]['accuracy']:.4f}")
print(f"   F1分数: {final_ranking.iloc[0]['f1_score']:.4f}")


# 11. 模型详细分析
print("\n11. 模型详细分析")
print("-" * 50)

# 12种模型中的混淆矩阵
for name in results.keys():
    y_pred = predictions[name]['y_pred']
    y_pred_proba = predictions[name]['y_pred_proba']

    print(f"\n🔍 {name} 模型详细分析")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(f'{name}_confusion_matrix.png', dpi=300)
     #plt.show()

print("\n" + "=" * 80)
print("🎉 完整的机器学习Pipeline已完成!")
print("🔍 包含EDA分析 → 数据预处理 → 模型对比 → 结果分析")
print("📊 生成了详细的可视化图表和分析报告")
print("=" * 80)

# 测试集预测结果，计算各个类别概率
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None

test_results_df = pd.DataFrame({
    '真实标签': y_test,
    '预测标签': y_test_pred
})

if y_test_pred_proba is not None:
    for i in range(y_test_pred_proba.shape[1]):
        test_results_df[f'类别_{i}_概率'] = y_test_pred_proba[:, i]

print("\n测试集预测结果示例:")
print(test_results_df.head())


# SHAP可视化 - 使用LightGBM模型
# 训练模型
import shap
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score
import numpy as np

# 定义目标函数
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return np.mean(scores)


print("开始贝叶斯优化...")
# 创建study并优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"最佳交叉验证分数: {study.best_value:.4f}")
print("最佳参数:")
for param, value in study.best_params.items():
    print(f"  {param}: {value}")

# 使用最佳参数训练最终模型
model = lgb.LGBMClassifier(**study.best_params, random_state=42, n_jobs=-1, verbose=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\n使用优化后参数的结果:")
print(f"训练集准确率: {train_score:.4f}")
print(f"测试集准确率: {test_score:.4f}")

# 1. 计算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 判断分类类型
if isinstance(shap_values, list) and len(shap_values) > 1:
    # 多分类
    n_classes = len(shap_values)
    shap_values_list = shap_values
else:
    # 二分类或单输出
    n_classes = 2
    shap_values_list = [np.array(shap_values), np.array(shap_values)]  # 保证兼容循环
    # 如果二分类，通常选择正类 SHAP 值
    shap_values_list[1] = shap_values_list[0]

# class_names_display 已存在
if 'class_names_display' not in locals():
    class_names_display = [f'Class_{i}' for i in range(n_classes)]

# ==================== SHAP Bar Plot - 综合特征重要性 ====================
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_list[1] if n_classes == 2 else shap_values_list[0],
                  X_test,
                  feature_names=feature_cols,
                  plot_type="bar",
                  class_names=class_names_display,
                  show=False)
plt.title('SHAP 特征重要性排序 - LightGBM', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_bar_plot_lightgbm.png', dpi=300, bbox_inches='tight')
 #plt.show()

# ==================== SHAP Feature Importance by Class ====================
fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 6))
if n_classes == 1:
    axes = [axes]
colors = ['steelblue', 'coral', 'lightgreen', 'orange', 'purple']  # 支持最多5类

for i in range(n_classes):
    class_importance = np.mean(np.abs(shap_values_list[i]), axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': class_importance
    }).sort_values('importance', ascending=True)

    axes[i].barh(importance_df['feature'], importance_df['importance'],
                 color=colors[i])
    axes[i].set_title(f'{class_names_display[i]}\nSHAP特征重要性 - LightGBM',
                      fontsize=12, fontweight='bold')
    axes[i].set_xlabel('平均|SHAP值|')

plt.tight_layout()
plt.savefig('shap_importance_by_class_lightgbm.png', dpi=300, bbox_inches='tight')
 #plt.show()

# ==================== SHAP Overall Feature Importance ====================
overall_importance = np.mean([np.mean(np.abs(shap_values_list[i]), axis=0)
                              for i in range(n_classes)], axis=0)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': overall_importance
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
bars = plt.barh(importance_df['feature'], importance_df['importance'],
                color='steelblue', alpha=0.8)
plt.title('SHAP 综合特征重要性 (所有类别平均) - LightGBM', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('平均|SHAP值|')
plt.grid(axis='x', alpha=0.3)

# 添加数值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
             f'{width:.3f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('shap_overall_importance_lightgbm.png', dpi=300, bbox_inches='tight')
 #plt.show()

# SHAP Feature Importance by Class - 每个类别的特征重要性
print("\nSHAP Feature Importance by Class - 每个类别的特征重要性")

# 类别数量
n_classes = len(class_names_display)
colors = ['steelblue', 'coral', 'lightgreen']  # 可根据类别数扩展

# 创建子图
fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 6))

# 如果只有一个类别，axes不是数组，需要处理
if n_classes == 1:
    axes = [axes]

for i in range(n_classes):
    # 计算该类别的平均绝对SHAP值
    class_importance = np.mean(np.abs(shap_values_list[i]), axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': class_importance
    }).sort_values('importance', ascending=True)

    axes[i].barh(importance_df['feature'], importance_df['importance'],
                 color=colors[i % len(colors)])
    axes[i].set_title(f'类别 {class_names_display[i]}\nSHAP特征重要性 - LightGBM',
                      fontsize=12, fontweight='bold')
    axes[i].set_xlabel('平均|SHAP值|')

plt.tight_layout()
plt.savefig('shap_importance_by_class_lightgbm.png', dpi=300, bbox_inches='tight')
 #plt.show()


# SHAP Overall Feature Importance - 综合特征重要性
print("\nSHAP Overall Feature Importance - 综合特征重要性")

# 类别数量
n_classes = len(shap_values_list)

# 计算所有类别的平均特征重要性
overall_importance = np.mean(
    [np.mean(np.abs(shap_values_list[i]), axis=0) for i in range(n_classes)],
    axis=0
)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': overall_importance
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
bars = plt.barh(importance_df['feature'], importance_df['importance'],
                color='steelblue', alpha=0.8)
plt.title('SHAP 综合特征重要性 (所有类别平均) - LightGBM',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('平均|SHAP值|')
plt.grid(axis='x', alpha=0.3)

# 添加数值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
             f'{width:.3f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('shap_overall_importance_lightgbm.png', dpi=300, bbox_inches='tight')
 #plt.show()
