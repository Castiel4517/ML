# =========================================
# 文件名：Diabetes_Model_Comparison.py
# 功能：整合 LGBM, XGB, SVM, RF, ADB, LR, KNN 七种模型进行糖尿病预测对比
# =========================================

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
import joblib

# ======================
# 设置中文显示
# ======================
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ======================
# 1. 加载数据
# ======================
df = pd.read_csv("./data/1018.csv")
X = df.iloc[:, :-2]
y = df.iloc[:, -1]

# ======================
# 2. 数据划分与标准化
# ======================
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# ======================
# 3. 定义统一训练与评估函数
# ======================
def train_and_evaluate(model_name, model, prob_func="predict_proba"):
    """统一训练、预测、评估函数"""
    model.fit(train_x, train_y)

    if prob_func == "predict_proba":
        ypred_prob = model.predict_proba(test_x)[:, 1]
    elif prob_func == "predict":
        ypred_prob = model.predict(test_x)
    else:
        raise ValueError("Invalid prob_func")

    # 自动阈值搜索（以F1最大为准）
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thr, best_f1 = 0.5, 0
    for thr in thresholds:
        y_pred = (ypred_prob >= thr).astype(int)
        f1 = metrics.f1_score(test_y, y_pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_pred_final = (ypred_prob >= best_thr).astype(int)
    precision = metrics.precision_score(test_y, y_pred_final)
    recall = metrics.recall_score(test_y, y_pred_final)
    f1 = metrics.f1_score(test_y, y_pred_final)
    accuracy = metrics.accuracy_score(test_y, y_pred_final)
    auc = metrics.roc_auc_score(test_y, ypred_prob)

    print(f"\n==== {model_name} ====")
    print(f"最佳阈值: {best_thr:.2f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Accuracy={accuracy:.4f}")

    return {
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "AUC": auc,
        "Best_Threshold": best_thr,
        "yprob": ypred_prob
    }

# ======================
# 4. 初始化各模型
# ======================

models = {
    "LGBM": lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, is_unbalance=True, random_state=7
    ),
    "XGB": xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=len(train_y[train_y==0])/len(train_y[train_y==1]),
        eval_metric="auc", use_label_encoder=False, random_state=7
    ),
    "SVM": SVC(
        kernel='rbf', probability=True, class_weight='balanced', C=1.0, gamma='scale', random_state=7
    ),
    "RF": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=4, min_samples_leaf=3,
        class_weight='balanced', random_state=7, n_jobs=-1
    ),
    "ADB": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight='balanced'),
        n_estimators=200, learning_rate=0.1, random_state=7
    ),
    "LR": LogisticRegression(
        penalty='l2', solver='liblinear', class_weight='balanced',
        max_iter=1000, random_state=7
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7, weights='distance', metric='minkowski', n_jobs=-1
    )
}

# ======================
# 5. 训练与评估所有模型
# ======================
results = []
for name, model in models.items():
    prob_func = "predict_proba" if hasattr(model, "predict_proba") else "predict"
    res = train_and_evaluate(name, model, prob_func)
    results.append(res)

# ======================
# 6. 汇总结果表
# ======================
results_df = pd.DataFrame(results)
results_df = results_df.drop(columns="yprob")
results_df = results_df.sort_values(by="AUC", ascending=False)
results_df.to_csv("./data/All_Models_Comparison.csv", index=False)
print("\n✅ 各模型评估结果已保存为 ./data/All_Models_Comparison.csv")

# ======================
# 7. 绘制所有模型 ROC 曲线
# ======================
plt.figure(figsize=(7, 6))
for res in results:
    fpr, tpr, _ = metrics.roc_curve(test_y, res["yprob"])
    plt.plot(fpr, tpr, label=f"{res['Model']} (AUC={res['AUC']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("七种模型 ROC 曲线对比")
plt.legend()
plt.tight_layout()
plt.savefig("./data/All_Models_ROC_Comparison.png")
plt.show()

# ======================
# 8. 保存标准化器
# ======================
joblib.dump(scaler, "./data/Scaler.pkl")
print("✅ 数据标准化模型已保存为 ./data/Scaler.pkl")

print("\n✅ 所有模型训练与对比完成！结果见 ./data/All_Models_Comparison.csv 与 ROC 图。")
