import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_circles, make_moons, load_iris
import os
import seaborn as sns
from scipy import stats
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# 创建结果目录
if not os.path.exists('result'):
    os.makedirs('result')

# 设置matplotlib适配macOS字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

def plot_decision_boundary(X, y, model, title, filename):
    """绘制决策边界"""
    X = X.values if hasattr(X, 'values') else X
    y = y.values if hasattr(y, 'values') else y
    
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA']))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF', '#00FF00']), edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.tight_layout()
    plt.savefig(f'result/{filename}.png', dpi=300)
    plt.close()

# 11.1 数据分类与模型选择
# 11.1.1 变量的取值类型
# 11.1.2 模型选择方式
def model_selection_demo():
    """模型选择示例"""
    # 生成示例数据
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                               n_informative=2, random_state=1, n_clusters_per_class=1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 模型选择
    models = [
        ("线性判别分析", LinearDiscriminantAnalysis()),
        ("二次判别分析", QuadraticDiscriminantAnalysis())
    ]
    
    results = []
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((name, accuracy))
        
        # 绘制决策边界
        plot_decision_boundary(X, y, model, f'{name} (准确率: {accuracy:.2f})', f'model_selection_{name}')
    
    return results

# 11.2 方差分析模型
def variance_analysis_demo():
    """方差分析模型示例"""
    # 11.2.1 完全随机设计模型
    def completely_randomized_design():
        # 生成三组不同均值的正态分布数据
        np.random.seed(42)
        group1 = np.random.normal(10, 2, 30)
        group2 = np.random.normal(12, 2, 30)
        group3 = np.random.normal(14, 2, 30)
        
        # 进行单因素方差分析
        f_stat, p_value = stats.f_oneway(group1, group2, group3)
        
        # 可视化
        plt.figure(figsize=(10, 6))
        sns.histplot(group1, kde=True, label='组1')
        sns.histplot(group2, kde=True, label='组2')
        sns.histplot(group3, kde=True, label='组3')
        plt.title(f'完全随机设计模型 (F={f_stat:.2f}, p={p_value:.4f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig('result/completely_randomized_design.png', dpi=300)
        plt.close()
        
        return f_stat, p_value
    
    # 11.2.2 随机区组设计模型
    def randomized_block_design():
        np.random.seed(42)
        
        treatments = 3
        blocks = 4
        
        # 生成模拟数据
        data = np.zeros((blocks, treatments))
        for i in range(blocks):
            block_effect = np.random.normal(0, 1)
            for j in range(treatments):
                treatment_effect = j * 2
                data[i, j] = 10 + block_effect + treatment_effect + np.random.normal(0, 1)
        
        # 转换为适合scipy的格式
        treatment_data = [data[:, j] for j in range(treatments)]
        
        # 使用单因素方差分析近似
        f_stat, p_value = stats.f_oneway(*treatment_data)
        
        # 可视化
        plt.figure(figsize=(10, 6))
        for i in range(blocks):
            plt.plot(range(treatments), data[i, :], 'o-', label=f'区组 {i+1}')
        plt.title(f'随机区组设计模型 (F={f_stat:.2f}, p={p_value:.4f})')
        plt.xlabel('处理')
        plt.ylabel('响应')
        plt.legend()
        plt.tight_layout()
        plt.savefig('result/randomized_block_design.png', dpi=300)
        plt.close()
        
        return f_stat, p_value
    
    # 11.2.3 析因设计模型
    def factorial_design():
        np.random.seed(42)
        
        # 两因素析因设计
        factor_a_levels = 3
        factor_b_levels = 2
        replications = 5
        
        # 生成模拟数据
        data = np.zeros((factor_a_levels, factor_b_levels, replications))
        
        for i in range(factor_a_levels):
            for j in range(factor_b_levels):
                effect_a = i * 2
                effect_b = j * 3
                interaction = i * j * 0.5
                for k in range(replications):
                    data[i, j, k] = 10 + effect_a + effect_b + interaction + np.random.normal(0, 1)
        
        # 可视化
        plt.figure(figsize=(10, 6))
        for j in range(factor_b_levels):
            means = [np.mean(data[i, j, :]) for i in range(factor_a_levels)]
            plt.plot(range(factor_a_levels), means, 'o-', label=f'因素B水平 {j+1}')
        plt.title('析因设计模型 - 交互效应图')
        plt.xlabel('因素A水平')
        plt.ylabel('平均响应')
        plt.legend()
        plt.tight_layout()
        plt.savefig('result/factorial_design.png', dpi=300)
        plt.close()
    
    # 11.2.4 正交设计模型
    def orthogonal_design():
        # 简单的正交设计示例 - L4(2^3)正交表
        L4 = np.array([
            [1, 1, 1],
            [1, 2, 2],
            [2, 1, 2],
            [2, 2, 1]
        ])
        
        np.random.seed(42)
        # 假设响应值
        responses = np.array([10, 14, 12, 18]) + np.random.normal(0, 1, 4)
        
        # 可视化
        plt.figure(figsize=(10, 6))
        for i in range(3):  # 3个因素
            level1_idx = L4[:, i] == 1
            level2_idx = L4[:, i] == 2
            
            level1_mean = np.mean(responses[level1_idx])
            level2_mean = np.mean(responses[level2_idx])
            
            plt.plot([1, 2], [level1_mean, level2_mean], 'o-', label=f'因素{i+1}')
        
        plt.title('正交设计模型 - 主效应图')
        plt.xlabel('因素水平')
        plt.ylabel('平均响应')
        plt.xticks([1, 2])
        plt.legend()
        plt.tight_layout()
        plt.savefig('result/orthogonal_design.png', dpi=300)
        plt.close()
    
    # 执行所有方差分析演示
    completely_randomized_design()
    randomized_block_design()
    factorial_design()
    orthogonal_design()

# 11.3 广义线性模型
def generalized_linear_models_demo():
    """广义线性模型示例"""
    # 11.3.1 广义线性模型概述
    
    # 11.3.2 Logistic模型
    def logistic_model():
        """Logistic回归示例"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                  n_informative=2, random_state=1, n_clusters_per_class=1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        
        y_pred = logreg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 决策边界可视化
        plot_decision_boundary(X, y, logreg, f'Logistic模型 (准确率: {accuracy:.2f})', 'logistic_model')
        
        # Logistic曲线可视化
        plt.figure(figsize=(8, 6))
        x = np.linspace(-6, 6, 100)
        y = 1 / (1 + np.exp(-x))
        plt.plot(x, y)
        plt.title('Logistic函数曲线')
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.grid(True)
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig('result/logistic_curve.png', dpi=300)
        plt.close()
        
        return accuracy
    
    # Logistic模型优缺点和VS线性模型在logistic_model函数中已经隐含体现
    
    # 11.3.3 对数线性模型
    def log_linear_model():
        """对数线性模型示例"""
        # 生成指数分布的数据
        np.random.seed(42)
        X = np.linspace(0.1, 5, 100)
        y = np.exp(0.8 * X + 1) + np.random.normal(0, 0.5 * np.exp(0.8 * X), 100)
        
        # 对数转换
        log_y = np.log(y)
        
        # 线性拟合
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X_reshaped = X.reshape(-1, 1)
        model.fit(X_reshaped, log_y)
        
        # 预测并反转换
        log_y_pred = model.predict(X_reshaped)
        y_pred = np.exp(log_y_pred)
        
        # 可视化原始空间
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, alpha=0.5)
        plt.plot(X, y_pred, 'r-', linewidth=2)
        plt.title('原始空间的对数线性模型')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # 可视化对数空间
        plt.subplot(1, 2, 2)
        plt.scatter(X, log_y, alpha=0.5)
        plt.plot(X, log_y_pred, 'r-', linewidth=2)
        plt.title('对数空间的线性关系')
        plt.xlabel('X')
        plt.ylabel('log(Y)')
        
        plt.tight_layout()
        plt.savefig('result/log_linear_model.png', dpi=300)
        plt.close()
    
    # 执行所有广义线性模型演示
    logistic_model()
    log_linear_model()

# Fisher判别分析
def fisher_discriminant_analysis():
    """Fisher判别分析示例"""
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data[:, :2]  # 只使用前两个特征便于可视化
    y = iris.target
    
    # 训练Fisher线性判别模型
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # 计算判别准确率
    y_pred = lda.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    # 绘制判别结果
    plot_decision_boundary(X, y, lda, f'Fisher线性判别分析 (准确率: {accuracy:.2f})', 'fisher_lda')
    
    # 可视化投影
    plt.figure(figsize=(10, 6))
    for i, c, marker in zip(range(3), ['r', 'g', 'b'], ['o', '^', 's']):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=c, marker=marker, label=f'类别 {i}')
    
    # 绘制类中心
    class_means = [np.mean(X[y == i], axis=0) for i in range(3)]
    for i, c in zip(range(3), ['r', 'g', 'b']):
        plt.scatter(class_means[i][0], class_means[i][1], c=c, marker='*', s=200, edgecolor='k')
    
    # 绘制Fisher判别方向
    for i in range(len(lda.coef_)):
        plt.arrow(0, 0, lda.coef_[i, 0], lda.coef_[i, 1], color='k', 
                 head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    plt.title('Fisher判别分析 - 判别方向')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('result/fisher_directions.png', dpi=300)
    plt.close()
    
    return accuracy

# Bayes判别分析
def bayes_discriminant_analysis():
    """贝叶斯判别分析示例"""
    np.random.seed(42)
    
    # 生成两个高斯分布的数据
    n1, n2 = 100, 100
    mu1 = [0, 0]
    mu2 = [3, 3]
    sigma1 = np.array([[2, 0.5], [0.5, 1]])
    sigma2 = np.array([[1, -0.5], [-0.5, 2]])
    
    X1 = np.random.multivariate_normal(mu1, sigma1, n1)
    X2 = np.random.multivariate_normal(mu2, sigma2, n2)
    
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(n1), np.ones(n2)))
    
    # 计算先验概率
    prior1 = n1 / (n1 + n2)
    prior2 = n2 / (n1 + n2)
    
    # 训练贝叶斯分类器（用QDA实现，考虑不同协方差矩阵）
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)
    
    # 预测
    y_pred = qda.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    # 可视化
    plot_decision_boundary(X, y, qda, f'贝叶斯判别分析 (准确率: {accuracy:.2f})', 'bayes_qda')
    
    # 可视化高斯分布
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=X1[:, 0], y=X1[:, 1], levels=7, color='blue', alpha=0.6, label='类别1')
    sns.kdeplot(x=X2[:, 0], y=X2[:, 1], levels=7, color='red', alpha=0.6, label='类别2')
    plt.scatter(X1[:, 0], X1[:, 1], c='blue', alpha=0.3, edgecolor='k')
    plt.scatter(X2[:, 0], X2[:, 1], c='red', alpha=0.3, edgecolor='k')
    plt.title('贝叶斯判别分析 - 高斯概率密度')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('result/bayes_density.png', dpi=300)
    plt.close()
    
    return accuracy

# 非线性判别函数
def nonlinear_discriminant_analysis():
    """非线性判别分析示例"""
    # 非线性数据集示例
    X1, y1 = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
    X2, y2 = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    datasets = [
        ("圆形数据", X1, y1),
        ("半月形数据", X2, y2)
    ]
    
    # 使用二次判别分析
    for name, X, y in datasets:
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X, y)
        y_pred = qda.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        plot_decision_boundary(X, y, qda, f'非线性判别分析 - {name} (准确率: {accuracy:.2f})', f'nonlinear_qda_{name}')

# 多总体距离判别
def multiple_population_distance_discriminant():
    """多总体距离判别示例"""
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data[:, :2]  # 只使用前两个特征便于可视化
    y = iris.target
    
    # 训练模型
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # 预测
    y_pred = lda.predict(X)
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    # 可视化
    plot_decision_boundary(X, y, lda, f'多总体距离判别 (准确率: {accuracy:.2f})', 'multi_population_lda')
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('多总体距离判别 - 混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig('result/multi_population_confusion.png', dpi=300)
    plt.close()
    
    # 计算各类之间的马氏距离
    class_means = [np.mean(X[y == i], axis=0) for i in range(3)]
    
    # 使用池化协方差矩阵（LDA假设）
    pooled_cov = np.zeros((2, 2))
    for i in range(3):
        n_i = np.sum(y == i)
        cov_i = np.cov(X[y == i], rowvar=False)
        pooled_cov += (n_i - 1) * cov_i
    pooled_cov /= (len(X) - 3)
    
    # 计算马氏距离矩阵
    mahalanobis_dist = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                mahalanobis_dist[i, j] = 0
            else:
                diff = class_means[i] - class_means[j]
                mahalanobis_dist[i, j] = np.sqrt(diff @ np.linalg.inv(pooled_cov) @ diff.T)
    
    # 可视化马氏距离
    plt.figure(figsize=(8, 6))
    sns.heatmap(mahalanobis_dist, annot=True, fmt='.2f', cmap='viridis')
    plt.title('类别间马氏距离')
    plt.xlabel('类别')
    plt.ylabel('类别')
    plt.tight_layout()
    plt.savefig('result/mahalanobis_distance.png', dpi=300)
    plt.close()
    
    return accuracy, mahalanobis_dist

# 正态总体判别
def normal_population_discriminant():
    """正态总体判别示例"""
    np.random.seed(42)
    
    # 生成三个正态总体
    n = 100
    mu1 = [0, 0]
    mu2 = [4, 0]
    mu3 = [2, 4]
    
    sigma1 = np.array([[2, 0], [0, 1]])
    sigma2 = np.array([[1, 0], [0, 2]])
    sigma3 = np.array([[1.5, 0.5], [0.5, 1.5]])
    
    X1 = np.random.multivariate_normal(mu1, sigma1, n)
    X2 = np.random.multivariate_normal(mu2, sigma2, n)
    X3 = np.random.multivariate_normal(mu3, sigma3, n)
    
    X = np.vstack((X1, X2, X3))
    y = np.hstack((np.zeros(n), np.ones(n), np.ones(n)*2))
    
    # 使用LDA和QDA进行判别
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis()
    
    lda.fit(X, y)
    qda.fit(X, y)
    
    y_pred_lda = lda.predict(X)
    y_pred_qda = qda.predict(X)
    
    acc_lda = accuracy_score(y, y_pred_lda)
    acc_qda = accuracy_score(y, y_pred_qda)
    
    # 可视化
    plot_decision_boundary(X, y, lda, f'正态总体LDA判别 (准确率: {acc_lda:.2f})', 'normal_pop_lda')
    plot_decision_boundary(X, y, qda, f'正态总体QDA判别 (准确率: {acc_qda:.2f})', 'normal_pop_qda')
    
    # 可视化正态分布
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=X1[:, 0], y=X1[:, 1], levels=7, color='red', alpha=0.6, label='总体1')
    sns.kdeplot(x=X2[:, 0], y=X2[:, 1], levels=7, color='blue', alpha=0.6, label='总体2')
    sns.kdeplot(x=X3[:, 0], y=X3[:, 1], levels=7, color='green', alpha=0.6, label='总体3')
    
    plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=0.3, edgecolor='k')
    plt.scatter(X2[:, 0], X2[:, 1], c='blue', alpha=0.3, edgecolor='k')
    plt.scatter(X3[:, 0], X3[:, 1], c='green', alpha=0.3, edgecolor='k')
    
    plt.title('正态总体密度')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('result/normal_pop_density.png', dpi=300)
    plt.close()
    
    return acc_lda, acc_qda

# 主函数
def main():
    print("开始执行多元统计分析演示...")
    
    print("\n1. 模型选择示例")
    model_results = model_selection_demo()
    for name, accuracy in model_results:
        print(f"  {name}: 准确率 = {accuracy:.4f}")
    
    print("\n2. 方差分析模型")
    variance_analysis_demo()
    print("  方差分析模型示例完成，结果已保存到result目录")
    
    print("\n3. 广义线性模型")
    generalized_linear_models_demo()
    print("  广义线性模型示例完成，结果已保存到result目录")
    
    print("\n4. Fisher判别分析")
    fisher_acc = fisher_discriminant_analysis()
    print(f"  Fisher判别分析准确率: {fisher_acc:.4f}")
    
    print("\n5. Bayes判别分析")
    bayes_acc = bayes_discriminant_analysis()
    print(f"  Bayes判别分析准确率: {bayes_acc:.4f}")
    
    print("\n6. 非线性判别分析")
    nonlinear_discriminant_analysis()
    print("  非线性判别分析示例完成，结果已保存到result目录")
    
    print("\n7. 多总体距离判别")
    multi_acc, dist_matrix = multiple_population_distance_discriminant()
    print(f"  多总体距离判别准确率: {multi_acc:.4f}")
    print("  类别间马氏距离矩阵:")
    print(dist_matrix)
    
    print("\n8. 正态总体判别")
    lda_acc, qda_acc = normal_population_discriminant()
    print(f"  正态总体LDA判别准确率: {lda_acc:.4f}")
    print(f"  正态总体QDA判别准确率: {qda_acc:.4f}")
    
    print("\n所有演示完成，结果图表已保存到result目录")

if __name__ == "__main__":
    main()