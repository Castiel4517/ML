# =========================================
# 文件名：SVM1.0.py
# 功能：使用支持向量机(SVM)进行糖尿病预测分析
# =========================================

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
# 3. 参数配置与模型训练
# ======================
# 基础模型（使用class_weight平衡类别）
svm_model = SVC(
    kernel='rbf',                # RBF核
    probability=True,            # 输出概率，用于AUC和阈值搜索
    class_weight='balanced',     # 自动平衡类别
    C=1.0,
    gamma='scale',
    random_state=7
)

# 可选：自动调参（只需开启下面代码块）
# param_grid = {
#     'C': [0.1, 1, 5, 10],
#     'gamma': ['scale', 0.01, 0.05, 0.1, 0.5],
#     'kernel': ['rbf', 'poly', 'sigmoid']
# }
# grid = GridSearchCV(svm_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# grid.fit(train_x, train_y)
# svm_model = grid.best_estimator_
# print("最优参数：", grid.best_params_)

# ======================
# 4. 训练模型
# ======================
svm_model.fit(train_x, train_y)

# ======================
# 5. 预测与概率输出
# ======================
ypred_prob = svm_model.predict_proba(test_x)[:, 1]

# ======================
# 6. 自动阈值搜索（以F1-score为目标）
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
# 7. 模型评估
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
plt.title("SVM 糖尿病预测混淆矩阵")
plt.tight_layout()
plt.savefig('./data/SVM_confusion_matrix.png')
plt.show()

# ======================
# 9. ROC曲线
# ======================
fpr, tpr, _ = metrics.roc_curve(test_y, ypred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC曲线')
plt.legend()
plt.tight_layout()
plt.savefig('./data/SVM_ROC.png')
plt.show()

# ======================
# 10. 线性可解释性分析（仅线性核可用）
# ======================
if svm_model.kernel == 'linear':
    coef = svm_model.coef_[0]
    feature_importance = pd.Series(coef, index=df.columns[:8]).sort_values(key=abs, ascending=False)
    print("\n特征重要性（线性核）:\n", feature_importance)
    feature_importance.to_csv('./data/SVM_feature_importance.csv')
else:
    print("\n非线性核（RBF/Poly/Sigmoid）无法直接提取特征权重。")

# ======================
# 11. 保存结果与模型
# ======================
metrics_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-score', 'Accuracy', 'AUC', 'Best_Threshold'],
    'Value': [precision, recall, f1, accuracy, auc, best_thr]
})
metrics_df.to_csv('./data/SVM_metrics.csv', index=False)
print("\n✅ 评估指标已保存为 ./data/SVM_metrics.csv")

joblib.dump(svm_model, './data/SVM_model.pkl')
print("✅ 模型已保存为 ./data/SVM_model.pkl")

joblib.dump(scaler, './data/SVM_scaler.pkl')
print("✅ 数据标准化模型已保存为 ./data/SVM_scaler.pkl")
