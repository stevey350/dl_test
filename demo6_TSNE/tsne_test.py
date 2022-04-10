'''
使用手写数字数据集，t-sne进行降维显示
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

digits = datasets.load_digits(n_class=10)
print(type(digits))         # Bunch类继承dict类，拥有dict类的基本功能，比如对键/值的遍历，或者简单查询一个属性是否存在
print('digits key:', digits.keys())
X, y = digits.data, digits.target
n_samples, n_features = X.shape    # n_samples=1797，表示总共1797个样本；n_features=64，每张图片8*8，共64维

print('X shape', X.shape)

'''显示原始数据'''
n = 20  # 每行20个数字，每列20个数字
img = np.ones((10 * n, 10 * n))
for i in range(n):
    ix = 10 * i + 1
    for j in range(n):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
plt.figure(figsize=(8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()


'''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print("Original data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
print(type(x_min), x_min)
print(type(x_max), x_max)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()

