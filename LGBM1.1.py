import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# =====================
# 设置中文显示
# =====================
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False     # 解决负号问题

# =====================
# 1. 数据加载
# =====================
df = pd.read_csv("./data/1018.csv")
data = df.iloc[:, :-2]
target = df.iloc[:, -1]

# =====================
# 2. 数据集划分
# =====================
train_x, test_x, train_y, test_y = train_test_split(
    data, target, test_size=0.2, random_state=7, stratify=target
)

# =====================
# 3. LightGBM 数据结构
# =====================
lgb_train = lgb.Dataset(train_x, label=train_y)
lgb_eval = lgb.Dataset(test_x, label=test_y, reference=lgb_train)

# =====================
# 4. 参数设置（改进版）
# =====================
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': True,  # 处理类别不平衡
    'learning_rate': 0.05,
    'max_depth': 4,
    'num_leaves': 15,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l2': 2.0,
    'reg_alpha': 0.1,
    'seed': 7,
    'verbosity': -1
}

callbacks = [
    lgb.early_stopping(stopping_rounds=30),
    lgb.log_evaluation(period=50)
]

# =====================
# 5. 模型训练
# =====================
bst = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_eval],
    callbacks=callbacks
)

# =====================
# 6. 模型预测
# =====================
ypred = bst.predict(test_x)

# =====================
# 7. 自动阈值搜索
# =====================
thresholds = np.linspace(0.1, 0.9, 9)
best_f1 = 0
best_thr = 0.5
for thr in thresholds:
    y_pred = (ypred >= thr).astype(int)
    f1 = metrics.f1_score(test_y, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"\n最优阈值 = {best_thr:.2f}, 对应F1 = {best_f1:.4f}\n")

# =====================
# 8. 使用最佳阈值计算指标
# =====================
y_pred_final = (ypred >= best_thr).astype(int)
precision = metrics.precision_score(test_y, y_pred_final)
recall = metrics.recall_score(test_y, y_pred_final)
f1 = metrics.f1_score(test_y, y_pred_final)
accuracy = metrics.accuracy_score(test_y, y_pred_final)
auc = metrics.roc_auc_score(test_y, ypred)

# 打印结果
print("模型性能评估：")
print("Precision: %.4f" % precision)
print("Recall: %.4f" % recall)
print("F1-score: %.4f" % f1)
print("Accuracy: %.4f" % accuracy)
print("AUC: %.4f" % auc)

# =====================
# 9. 混淆矩阵
# =====================
cm = metrics.confusion_matrix(test_y, y_pred_final)
print("\n混淆矩阵：\n", cm)
disp = metrics.ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.title("LightGBM 糖尿病预测混淆矩阵")
plt.savefig('./data/confusion_matrix.png')
plt.show()

# =====================
# 10. 保存评估指标
# =====================
metrics_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-score', 'Accuracy', 'AUC', 'Best_Threshold'],
    'Value': [precision, recall, f1, accuracy, auc, best_thr]
})
metrics_df.to_csv('./data/metrics.csv', index=False)
print("\n✅ 评估指标已保存为 ./data/metrics.csv")

# =====================
# 11. 模型保存与加载
# =====================
bst.save_model('diabetes_model.txt')
print("✅ 模型已保存为 diabetes_model.txt")

# =====================
# 12. 特征重要性
# =====================
plt.figure(figsize=(8, 6))
lgb.plot_importance(bst, height=0.8, title='影响糖尿病的重要特征', ylabel='特征')
plt.rc('font', family='SimHei', size=12)
plt.tight_layout()
plt.savefig('./data/feature_importance.png')
plt.show()
