from sklearn.cluster import AgglomerativeClustering  
from sklearn.datasets import make_blobs  
import matplotlib.pyplot as plt  
import numpy as np
# 创建模拟数据集  
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)  
  
# 绘制原始数据的散点图  
plt.scatter(X[:, 0], X[:, 1], c='gray', marker='o', s=50, edgecolor='black')  
plt.title("Original Data")  
plt.xlabel("Feature 1")  
plt.ylabel("Feature 2")  
plt.show()  

matrix = np.zeros(300,300)

# 使用 AgglomerativeClustering 进行聚类  
# 这里指定了簇的数量为4  
clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(X)

# 对数据进行拟合  
# clustering.fit_predict(X)  
  
# 绘制聚类后的数据  
plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, cmap='viridis', marker='o', s=50, edgecolor='black')  
plt.title("Agglomerative Clustering")  
plt.xlabel("Feature 1")  
plt.ylabel("Feature 2")  
plt.show()