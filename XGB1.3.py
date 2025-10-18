import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
import chardet

# ======================
# 设置中文显示
# ======================
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ======================
# 1. 数据加载
# ======================
# 自动检测文件编码
with open("./data/1018.csv", 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']

df = pd.read_csv("./data/1018.csv", encoding=encoding)

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

data = df.iloc[:, :-2]
target = df.iloc[:, -1]

# ======================
# 2. 数据划分
# ======================
train_x, test_x, train_y, test_y = train_test_split(
    data, target, test_size=0.2, random_state=7, stratify=target
)

# 计算类别不平衡比例
pos_weight = len(train_y[train_y == 0]) / len(train_y[train_y == 1])
print(f"类别不平衡比: scale_pos_weight = {pos_weight:.2f}")

# ======================
# 3. 构造 DMatrix
# ======================
dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)

# ======================
# 4. 参数优化配置
# ======================
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'logloss'],
    'max_depth': 4,
    'min_child_weight': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eta': 0.03,
    'gamma': 0.4,
    'lambda': 2,
    'alpha': 0.1,
    'scale_pos_weight': pos_weight,
    'seed': 7,
    'nthread': 8,
    'verbosity': 1
}

# ======================
# 5. 模型训练
# ======================
watchlist = [(dtrain, 'train'), (dtest, 'eval')]
evals_result = {}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=watchlist,
    evals_result=evals_result,
    early_stopping_rounds=30,
    verbose_eval=50
)

# ======================
# 6. 模型预测
# ======================
ypred = bst.predict(dtest)

# ======================
# 7. 自动阈值搜索
# ======================
thresholds = np.linspace(0.1, 0.9, 17)
best_thr, best_f1 = 0.5, 0
for thr in thresholds:
    y_pred = (ypred >= thr).astype(int)
    f1 = metrics.f1_score(test_y, y_pred)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"\n最优阈值 = {best_thr:.2f}, 对应 F1-score = {best_f1:.4f}")

# ======================
# 8. 最终模型评估
# ======================
y_pred_final = (ypred >= best_thr).astype(int)

precision = metrics.precision_score(test_y, y_pred_final)
recall = metrics.recall_score(test_y, y_pred_final)
f1 = metrics.f1_score(test_y, y_pred_final)
accuracy = metrics.accuracy_score(test_y, y_pred_final)
auc = metrics.roc_auc_score(test_y, ypred)

print("\n模型性能评估：")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

# ======================
# 9. 混淆矩阵
# ======================
cm = metrics.confusion_matrix(test_y, y_pred_final)
print("\n混淆矩阵：\n", cm)
disp = metrics.ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.title("XGBoost 糖尿病预测混淆矩阵")
plt.tight_layout()
plt.savefig('./data/confusion_matrix.png')
plt.show()

# ======================
# 10. AUC 学习曲线
# ======================
plt.figure(figsize=(6, 4))
plt.plot(evals_result['train']['auc'], label='训练集AUC')
plt.plot(evals_result['eval']['auc'], label='验证集AUC')
plt.xlabel('迭代轮数')
plt.ylabel('AUC')
plt.title('XGBoost AUC 学习曲线')
plt.legend()
plt.tight_layout()
plt.savefig('./data/auc_curve.png')
plt.show()

# ======================
# 11. 特征重要性
# ======================
xgb.plot_importance(bst, importance_type='gain', height=0.8, title='特征重要性 (按增益)')
plt.tight_layout()
plt.savefig('./data/feature_importance.png')
plt.show()

# ======================
# 12. 保存结果与模型
# ======================
metrics_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-score', 'Accuracy', 'AUC', 'Best_Threshold'],
    'Value': [precision, recall, f1, accuracy, auc, best_thr]
})
metrics_df.to_csv('./data/metrics.csv', index=False)
print("\n✅ 评估指标已保存为 ./data/metrics.csv")

bst.save_model('diabetes_xgb_model.json')
print("✅ 模型已保存为 diabetes_xgb_model.json")
