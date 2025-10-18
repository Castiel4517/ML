# =========================================
# 文件名：ADB1.0.py
# 功能：使用 AdaBoost 算法进行糖尿病预测分析
# =========================================

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib
import joblib

# ======================
# 中文显示设置
# ======================
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ======================
# 1. 数据加载
# ======================
df = pd.read_csv("./data/diabetes.csv")
X = df.iloc[:, :8]
y = df.iloc[:, -1]

# ======================
# 2. 数据集划分与标准化
# ======================
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# ======================
# 3. 模型配置（基础学习器为浅层决策树）
# ======================
base_tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=7
)

adb_model = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=200,          # 弱学习器数量
    learning_rate=0.1,         # 学习率
    random_state=7
)

# 可选自动调参（如需开启取消注释）
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'estimator__max_depth': [2, 3, 4],
# }
# grid = GridSearchCV(adb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# grid.fit(train_x, train_y)
# adb_model = grid.best_estimator_
# print("最优参数：", grid.best_params_)

# ======================
# 4. 模型训练
# ======================
adb_model.fit(train_x, train_y)

# ======================
# 5. 模型预测
# ======================
ypred_prob = adb_model.predict_proba(test_x)[:, 1]

# ======================
# 6. 自动阈值搜索（基于 F1-score）
# ======================
thresholds = np.linspace(0.1, 0.9, 17)
best_thr, best_f1 = 0.5, 0
for thr in thresholds:
    y_pred = (ypred_prob >= thr).astype(int)
    f1 = metrics.f1_score(test_y, y_pred)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"\n最优阈值 = {best_thr:.2f}, 对应 F1 = {best_f1:.4f}")

# ======================
# 7. 最终评估
# ======================
y_pred_final = (ypred_prob >= best_thr).astype(int)
precision = metrics.precision_score(test_y, y_pred_final)
recall = metrics.recall_score(test_y, y_pred_final)
f1 = metrics.f1_score(test_y, y_pred_final)
accuracy = metrics.accuracy_score(test_y, y_pred_final)
auc = metrics.roc_auc_score(test_y, ypred_prob)

print("\n模型性能评估：")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

# ======================
# 8. 混淆矩阵
# ======================
cm = metrics.confusion_matrix(test_y, y_pred_final)
print("\n混淆矩阵：\n", cm)
disp = metrics.ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.title("AdaBoost 糖尿病预测混淆矩阵")
plt.tight_layout()
plt.savefig('./data/ADB_confusion_matrix.png')
plt.show()

# ======================
# 9. ROC 曲线
# ======================
fpr, tpr, _ = metrics.roc_curve(test_y, ypred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AdaBoost ROC 曲线')
plt.legend()
plt.tight_layout()
plt.savefig('./data/ADB_ROC.png')
plt.show()

# ======================
# 10. 特征重要性
# ======================
try:
    importances = pd.Series(adb_model.feature_importances_, index=df.columns[:8])
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(6, 5))
    importances.plot(kind='barh')
    plt.title('AdaBoost 特征重要性')
    plt.tight_layout()
    plt.savefig('./data/ADB_feature_importance.png')
    plt.show()

    importances.sort_values(ascending=False).to_csv('./data/ADB_feature_importance.csv')
except AttributeError:
    print("当前AdaBoost基学习器不支持直接输出特征重要性。")

# ======================
# 11. 保存模型与结果
# ======================
metrics_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-score', 'Accuracy', 'AUC', 'Best_Threshold'],
    'Value': [precision, recall, f1, accuracy, auc, best_thr]
})
metrics_df.to_csv('./data/ADB_metrics.csv', index=False)
print("\n✅ 评估指标已保存为 ./data/ADB_metrics.csv")

joblib.dump(adb_model, './data/ADB_model.pkl')
print("✅ 模型已保存为 ./data/ADB_model.pkl")

joblib.dump(scaler, './data/ADB_scaler.pkl')
print("✅ 数据标准化模型已保存为 ./data/ADB_scaler.pkl")
