# 区域经济发展聚类分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import os

# 设置MacOS默认中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
if not os.path.exists('result'):
    os.makedirs('result')

# 创建中国各省经济数据
def create_economic_data():
    # 省份列表
    provinces = [
        '北京', '天津', '河北', '山西', '内蒙古', 
        '辽宁', '吉林', '黑龙江', '上海', '江苏', 
        '浙江', '安徽', '福建', '江西', '山东', 
        '河南', '湖北', '湖南', '广东', '广西', 
        '海南', '重庆', '四川', '贵州', '云南', 
        '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
    ]
    
    # 设置随机种子
    np.random.seed(42)
    
    # 不同区域
    eastern = ['北京', '天津', '上海', '江苏', '浙江', '福建', '山东', '广东', '海南']
    central = ['河北', '山西', '安徽', '江西', '河南', '湖北', '湖南']
    western = ['内蒙古', '广西', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']
    northeast = ['辽宁', '吉林', '黑龙江']
    
    # 各区域基准值
    base_values = {
        'eastern': [12000, 100000, 75, 60, 8000, 5000],
        'central': [5000, 50000, 60, 45, 4000, 2500],
        'western': [3000, 40000, 50, 40, 3000, 1500],
        'northeast': [4000, 45000, 65, 50, 3500, 2000]
    }
    
    # 创建数据
    data = []
    for province in provinces:
        if province in eastern:
            region = 'eastern'
        elif province in central:
            region = 'central'
        elif province in western:
            region = 'western'
        else:
            region = 'northeast'
        
        base = base_values[region]
        
        # 添加随机浮动
        gdp = base[0] * (1 + np.random.uniform(-0.3, 0.3))
        gdp_per_capita = base[1] * (1 + np.random.uniform(-0.3, 0.3))
        urbanization = base[2] * (1 + np.random.uniform(-0.15, 0.15))
        tertiary_industry = base[3] * (1 + np.random.uniform(-0.15, 0.15))
        fixed_investment = base[4] * (1 + np.random.uniform(-0.3, 0.3))
        retail_sales = base[5] * (1 + np.random.uniform(-0.3, 0.3))
        
        data.append([
            province, gdp, gdp_per_capita, urbanization, 
            tertiary_industry, fixed_investment, retail_sales
        ])
    
    columns = ['省份', 'GDP(亿元)', '人均GDP(元)', '城镇化率(%)', 
               '第三产业占比(%)', '固定资产投资(亿元)', '社会消费品零售总额(亿元)']
    df = pd.DataFrame(data, columns=columns)
    
    df.to_csv('result/economic_data.csv', index=False, encoding='utf-8')
    return df

# 数据预处理
def preprocess_data(df):
    # 提取特征和省份
    features = df.iloc[:, 1:].values
    provinces = df.iloc[:, 0].values
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, provinces

# 主成分分析
def do_pca(features_scaled):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    
    # 计算解释方差
    explained_variance = pca.explained_variance_ratio_
    print(f"主成分解释的方差比例: {explained_variance}")
    print(f"累计解释的方差比例: {sum(explained_variance):.4f}")
    
    return principal_components

# K-means聚类
def do_kmeans(features_scaled, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    return labels

# 层次聚类
def do_hierarchical(features_scaled, n_clusters=4):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(features_scaled)
    return labels

# 可视化聚类结果
def visualize_clusters(principal_components, provinces, labels, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                c=labels, cmap='viridis', s=80, alpha=0.8)
    
    # 添加省份标签
    for i, txt in enumerate(provinces):
        plt.annotate(txt, (principal_components[i, 0], principal_components[i, 1]), 
                   fontsize=9, xytext=(5, 5), textcoords='offset points')
    
    plt.legend(*scatter.legend_elements(), title="聚类")
    plt.title(title)
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.grid(True)
    plt.savefig(f'result/{title.replace(" ", "_")}.png')
    plt.close()

# 分析聚类特征
def analyze_clusters(df, labels, n_clusters):
    # 添加聚类标签
    df_with_labels = df.copy()
    df_with_labels['聚类'] = labels
    
    # 计算各聚类的特征均值 - 排除'省份'列
    cluster_means = df_with_labels.groupby('聚类').mean(numeric_only=True)
    
    # 保存聚类特征均值
    cluster_means.to_csv('result/cluster_means.csv', encoding='utf-8')
    
    # 可视化各聚类的特征均值
    plt.figure(figsize=(12, 8))
    ax = plt.imshow(cluster_means, cmap='viridis')
    plt.colorbar(ax)
    plt.title('各聚类的经济指标均值')
    plt.savefig('result/cluster_means.png')
    plt.close()
    
    return cluster_means

# 主函数
def main():
    print("开始区域经济发展聚类分析...")
    
    # 1. 创建经济数据
    df = create_economic_data()
    print(f"数据集大小: {df.shape}")
    
    # 2. 数据预处理
    features_scaled, provinces = preprocess_data(df)
    
    # 3. 主成分分析
    principal_components = do_pca(features_scaled)
    
    # 4. K-means聚类
    n_clusters = 4
    kmeans_labels = do_kmeans(features_scaled, n_clusters)
    
    # 5. 层次聚类
    hc_labels = do_hierarchical(features_scaled, n_clusters)
    
    # 6. 可视化聚类结果
    visualize_clusters(principal_components, provinces, kmeans_labels, "K-means聚类结果")
    visualize_clusters(principal_components, provinces, hc_labels, "层次聚类结果")
    
    # 7. 分析聚类特征
    cluster_means = analyze_clusters(df, kmeans_labels, n_clusters)
    
    # 8. 查看聚类结果
    cluster_provinces = {}
    for i in range(n_clusters):
        provinces_in_cluster = df.iloc[kmeans_labels == i, 0].tolist()
        cluster_provinces[i] = provinces_in_cluster
        print(f"\n聚类 {i}: {', '.join(provinces_in_cluster)}")
    
    print("\n区域经济发展聚类分析完成，结果保存在result/目录")

if __name__ == "__main__":
    main()
