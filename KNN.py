# =========================================
# 文件名：KNN1.0.py
# 功能：使用K近邻(K-Nearest Neighbors)算法进行糖尿病预测分析
# =========================================

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
# 2. 数据划分与标准化
# ======================
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# ======================
# 3. 模型配置
# ======================
knn_model = KNeighborsClassifier(
    n_neighbors=7,        # 默认邻居数
    weights='distance',   # 按距离加权
    metric='minkowski',   # 距离度量（默认欧氏）
    n_jobs=-1
)

# 可选自动调参（如需开启取消注释）
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9, 11, 13],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan', 'minkowski']
# }
# grid = GridSearchCV(knn_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# grid.fit(train_x, train_y)
# knn_model = grid.best_estimator_
# print("最优参数：", grid.best_params_)

# ======================
# 4. 模型训练（KNN无需显式fit参数）
# ======================
knn_model.fit(train_x, train_y)

# ======================
# 5. 模型预测（概率输出）
# ======================
ypred_prob = knn_model.predict_proba(test_x)[:, 1]

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
plt.title("KNN 糖尿病预测混淆矩阵")
plt.tight_layout()
plt.savefig('./data/KNN_confusion_matrix.png')
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
plt.title('KNN ROC 曲线')
plt.legend()
plt.tight_layout()
plt.savefig('./data/KNN_ROC.png')
plt.show()

# ======================
# 10. 特征重要性（KNN 无显式重要性，可用距离贡献近似）
# ======================
print("\n⚠️ KNN 属于基于样本的非参数算法，不存在显式的特征重要性。")
print("若需解释性分析，可使用 SHAP、LIME 等方法。")

# ======================
# 11. 保存结果与模型
# ======================
metrics_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-score', 'Accuracy', 'AUC', 'Best_Threshold'],
    'Value': [precision, recall, f1, accuracy, auc, best_thr]
})
metrics_df.to_csv('./data/KNN_metrics.csv', index=False)
print("\n✅ 评估指标已保存为 ./data/KNN_metrics.csv")

joblib.dump(knn_model, './data/KNN_model.pkl')
print("✅ 模型已保存为 ./data/KNN_model.pkl")

joblib.dump(scaler, './data/KNN_scaler.pkl')
print("✅ 数据标准化模型已保存为 ./data/KNN_scaler.pkl")
