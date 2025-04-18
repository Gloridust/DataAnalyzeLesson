# 多元统计分析及Python建模 - 聚类分析实现

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, AgglomerativeClustering
import os

# 设置MacOS默认中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
if not os.path.exists('result'):
    os.makedirs('result')

# 计算距离矩阵
def calculate_distance_matrix(data, method='euclidean'):
    if method == 'mahalanobis':
        cov = np.cov(data.T)
        inv_cov = np.linalg.inv(cov)
        n = data.shape[0]
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                diff = data[i] - data[j]
                d[i, j] = d[j, i] = np.sqrt(diff.dot(inv_cov).dot(diff.T))
        return d
    else:
        return squareform(pdist(data, metric=method))

# 系统聚类方法
def hierarchical_clustering(data, method='single', n_clusters=None):
    if method in ['median', 'centroid']:
        Z = linkage(data, method)
        
        if n_clusters is not None:
            # 对于'median'和'centroid'方法，使用'average'作为AgglomerativeClustering的linkage参数
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
            labels = model.fit_predict(data)
            return Z, labels
    else:
        Z = linkage(data, method)
    
        if n_clusters is not None:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
            labels = model.fit_predict(data)
            return Z, labels
    return Z, None

# 绘制聚类树状图
def plot_dendrogram(Z, labels=None, title="系统聚类树状图", method_name=""):
    plt.figure(figsize=(10, 7))
    plt.title(title)
    dendrogram(Z, labels=labels)
    plt.xlabel("数据点")
    plt.ylabel("距离")
    plt.savefig(f'result/dendrogram_{method_name}.png')
    plt.close()

# K-means聚类
def kmeans_clustering(data, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return labels, centers

# 绘制聚类结果散点图
def plot_clusters(data, labels, centers=None, title="聚类结果"):
    if data.shape[1] > 2:
        print("注意: 数据维度 > 2, 仅显示前两个维度")
    
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X')
    
    plt.title(title)
    plt.xlabel("特征1")
    plt.ylabel("特征2")
    plt.savefig(f'result/{title.replace(" ", "_").lower()}.png')
    plt.close()

# 展示标记点示例
def example_plot_labeled_points():
    sn = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    x1 = np.array([2.5, 3, 6, 6.6, 7.2, 4, 4.7, 4.5, 5.5])
    x2 = np.array([2.1, 2.5, 2.5, 1.5, 3, 6.4, 5.6, 7.6, 6.9])
    
    plt.figure(figsize=(10, 7))
    plt.plot(x1, x2, '.')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    for i in range(len(sn)):
        plt.text(x1[i], x2[i], str(sn[i]))
    
    plt.savefig('result/labeled_points_example.png')
    plt.close()
    
    return np.column_stack((x1, x2))

# 演示所有聚类方法
def demo_all_methods():
    # 示例数据
    data = example_plot_labeled_points()
    print("数据集形状:", data.shape)
    
    # 1. 计算距离矩阵
    print("\n1. 计算距离矩阵")
    dist_methods = ['euclidean', 'cityblock', 'minkowski', 'mahalanobis']
    for method in dist_methods:
        dist_matrix = calculate_distance_matrix(data, method=method)
        print(f"\n{method}距离矩阵:")
        print(np.round(dist_matrix, 2))
        
        # 保存距离矩阵到CSV
        pd.DataFrame(dist_matrix).to_csv(f'result/distance_matrix_{method}.csv')
    
    # 2. 系统聚类方法
    print("\n2. 系统聚类方法")
    n_clusters = 3
    hc_methods = ['single', 'complete', 'average', 'median', 'centroid', 'ward']
    
    for method in hc_methods:
        print(f"\n使用{method}法进行聚类:")
        Z, labels = hierarchical_clustering(data, method=method, n_clusters=n_clusters)
        
        # 绘制树状图
        plot_dendrogram(Z, title=f"使用{method}法的层次聚类树状图", method_name=method)
        
        # 绘制聚类结果
        if labels is not None:
            plot_clusters(data, labels, title=f"{method}聚类结果")
            print("聚类标签:", labels)
    
    # 3. K-means聚类
    print("\n3. K-means聚类")
    labels, centers = kmeans_clustering(data, n_clusters=n_clusters)
    print("K-means聚类标签:", labels)
    print("聚类中心:", centers)
    
    # 绘制K-means聚类结果
    plot_clusters(data, labels, centers, title="K-means聚类结果")
    
    # 4. 评估轮廓系数
    from sklearn.metrics import silhouette_score, silhouette_samples
    import matplotlib.cm as cm
    
    # K-means聚类结果评估
    kmeans_sil = silhouette_score(data, labels)
    print(f"K-means聚类轮廓系数: {kmeans_sil:.3f}")
    
    # 5. 评估不同聚类数的K-means结果
    inertia = []
    silhouette = []
    clusters_range = range(2, 8)
    
    for n_clusters in clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        
        inertia.append(kmeans.inertia_)
        if n_clusters > 1:
            silhouette.append(silhouette_score(data, labels))
        else:
            silhouette.append(0)
    
    # 绘制肘部法则图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(clusters_range, inertia, 'o-', markersize=8)
    plt.title('肘部法则')
    plt.xlabel('聚类数')
    plt.ylabel('惯性')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(clusters_range, silhouette, 'o-', markersize=8)
    plt.title('轮廓系数法')
    plt.xlabel('聚类数')
    plt.ylabel('轮廓系数')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('result/kmeans_evaluation.png')
    plt.close()
    
    # 6. 绘制距离矩阵热图
    plt.figure(figsize=(10, 8))
    dist_matrix = calculate_distance_matrix(data, method='euclidean')
    plt.imshow(dist_matrix, cmap='viridis')
    plt.colorbar(label='距离')
    plt.title('欧氏距离矩阵')
    plt.savefig('result/distance_heatmap.png')
    plt.close()
    
    print("\n所有演示完成，结果保存在 result/ 目录")

if __name__ == "__main__":
    demo_all_methods()
