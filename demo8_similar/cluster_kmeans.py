from sklearn.cluster import KMeans
import numpy as np

# X表示一组图像的特征向量
X = np.array([[1, 2], [1, 4], [1, 0],
           [4, 2], [4, 4], [4, 0]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
# array([0, 0, 0, 1, 1, 1], dtype=int32)

kmeans.predict([[0, 0], [4, 4]])
# array([0, 1], dtype=int32)

kmeans.cluster_centers_
# array([[ 1.,  2.],
#        [ 4.,  2.]])
