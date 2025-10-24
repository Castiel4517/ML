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
    stratify=y
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
    X_final, y_final, test_size=0.2, random_state=42, stratify=y)
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

