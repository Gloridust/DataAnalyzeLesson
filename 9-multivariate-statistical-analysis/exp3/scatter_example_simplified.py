# 散点图聚类示例
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# 设置MacOS默认中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
if not os.path.exists('result'):
    os.makedirs('result')

# 示例数据点
sn = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
x1 = np.array([2.5, 3, 6, 6.6, 7.2, 4, 4.7, 4.5, 5.5])
x2 = np.array([2.1, 2.5, 2.5, 1.5, 3, 6.4, 5.6, 7.6, 6.9])

# 1. 绘制原始散点图
plt.figure(figsize=(8, 6))
plt.plot(x1, x2, '.')
plt.xlabel('x1')
plt.ylabel('x2')

# 添加点标签
for i in range(len(sn)):
    plt.text(x1[i], x2[i], str(sn[i]))

plt.savefig('result/scatter_plot.png')
plt.close()

# 2. 使用K-means进行聚类
X = np.column_stack((x1, x2))
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# 3. 绘制聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(x1, x2, c=labels, cmap='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')

# 添加点标签
for i in range(len(sn)):
    plt.text(x1[i], x2[i], str(sn[i]))

plt.title('K-means聚类结果 (k=3)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('result/kmeans_result.png')
plt.close()

# 4. 计算欧氏距离矩阵
dist_matrix = np.zeros((len(x1), len(x1)))
for i in range(len(x1)):
    for j in range(len(x1)):
        dist_matrix[i, j] = np.sqrt((x1[i] - x1[j])**2 + (x2[i] - x2[j])**2)

# 5. 绘制距离矩阵热图
plt.figure(figsize=(8, 6))
plt.imshow(dist_matrix, cmap='viridis')
plt.colorbar(label='距离')
plt.title('欧氏距离矩阵')

# 设置刻度标签
plt.xticks(range(len(sn)), sn)
plt.yticks(range(len(sn)), sn)

# 添加数值标签
for i in range(len(sn)):
    for j in range(len(sn)):
        plt.text(j, i, f'{dist_matrix[i, j]:.1f}', 
                ha='center', va='center', 
                color='white' if dist_matrix[i, j] > dist_matrix.mean() else 'black')

plt.savefig('result/distance_matrix.png')
plt.close()

# 6. 使用层次聚类
from scipy.cluster.hierarchy import dendrogram, linkage

# 计算层次聚类
Z = linkage(X, method='ward')

# 绘制树状图
plt.figure(figsize=(10, 6))
plt.title('层次聚类树状图')
dendrogram(Z, labels=sn)
plt.xlabel('样本')
plt.ylabel('距离')
plt.savefig('result/dendrogram.png')
plt.close()

print("聚类分析完成，结果保存在result/目录")
