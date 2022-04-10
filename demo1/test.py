import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import sys

print(sys.version)

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])

# 通过阈值，将得分转类别
threshold = 0.5
y_pred = np.zeros(y_scores.size)
for index, y_score in enumerate(y_scores):
    if y_score >= threshold:
        y_pred[index] = 1
    else:
        y_pred[index] = 0


# 混淆矩阵
print(y_pred)
ret = metrics.confusion_matrix(y_true, y_pred)
print('confusion matrix: \n', ret)

# 准确率accuracy
ret = metrics.accuracy_score(y_true, y_pred)
print('accuracy score: ', ret)

# 精确率precision
ret = metrics.precision_score(y_true, y_pred)
print('precision score: ', ret)

# 召回率
ret = metrics.recall_score(y_true, y_pred)
print('recall score:', ret)

# 画PR曲线
precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
print('PR precision:', precision)
print('PR recall:', recall)
print('PR thresholds:', thresholds)

plt.figure(0)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall')
# plt.legend(loc="lower left")
plt.show()

# 画ROC曲线
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)

print('ROC FPR:', fpr)
print('ROC TPR:', tpr)
print('ROC thresholds:', thresholds)

plt.figure(1)
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.grid(True)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('ROC')
# plt.legend(loc="lower left")
plt.show()

ret = metrics.auc(fpr, tpr)
print('auc', ret)
ret = metrics.roc_auc_score(y_true, y_scores)
print('auroc', ret)

