import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from scipy import stats
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

print("=== 10.2.1 简单相关分析 ===")

# 生成模拟数据
np.random.seed(42)
n = 100

# 生成相关联的变量
x1 = np.random.normal(100, 15, n)
x2 = 0.8 * x1 + np.random.normal(0, 8, n)
x3 = -0.3 * x1 + np.random.normal(50, 10, n)

y1 = -0.4 * x1 + 0.2 * x2 + np.random.normal(80, 12, n)
y2 = -0.5 * x1 - 0.6 * x2 + np.random.normal(60, 10, n)
y3 = 0.15 * x1 + 0.25 * x2 + np.random.normal(40, 8, n)

# 创建数据框
data = pd.DataFrame({
    'x1': x1, 'x2': x2, 'x3': x3,
    'y1': y1, 'y2': y2, 'y3': y3
})

print("数据基本信息:")
print(data.describe())
print("\n相关系数矩阵:")
correlation_matrix = data.corr()
print(correlation_matrix)

# 绘制相关系数热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f')
plt.title('变量间相关系数热力图')
plt.tight_layout()
plt.show()

# 绘制散点图矩阵
plt.figure(figsize=(12, 10))
pd.plotting.scatter_matrix(data, alpha=0.6, figsize=(12, 10), diagonal='hist')
plt.suptitle('变量散点图矩阵')
plt.tight_layout()
plt.show()

# 10.3 典型相关分析原理
print("\n=== 10.3 典型相关分析原理 ===")

# 10.4.1 计算典型系数及变量
print("\n=== 10.4.1 计算典型系数及变量 ===")

def canonical_correlation_analysis(X, Y, n_components=None):
    """
    典型相关分析函数
    """
    X = np.array(X)
    Y = np.array(Y)
    
    if n_components is None:
        n_components = min(X.shape[1], Y.shape[1])
    
    # 标准化数据
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_std = np.std(X, axis=0, ddof=1)
    Y_std = np.std(Y, axis=0, ddof=1)
    
    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std
    
    # 计算协方差矩阵
    n = X.shape[0]
    Sxx = np.cov(X_norm.T)
    Syy = np.cov(Y_norm.T)
    Sxy = np.cov(X_norm.T, Y_norm.T)[:X.shape[1], X.shape[1]:]
    Syx = Sxy.T
    
    # 计算典型相关系数
    try:
        Sxx_inv = np.linalg.inv(Sxx)
        Syy_inv = np.linalg.inv(Syy)
    except:
        Sxx_inv = np.linalg.pinv(Sxx)
        Syy_inv = np.linalg.pinv(Syy)
    
    # 求解特征值问题
    M1 = Sxx_inv @ Sxy @ Syy_inv @ Syx
    M2 = Syy_inv @ Syx @ Sxx_inv @ Sxy
    
    eigenvals1, eigenvecs1 = np.linalg.eig(M1)
    eigenvals2, eigenvecs2 = np.linalg.eig(M2)
    
    # 排序
    idx = np.argsort(eigenvals1)[::-1]
    eigenvals1 = eigenvals1[idx]
    eigenvecs1 = eigenvecs1[:, idx]
    
    idx = np.argsort(eigenvals2)[::-1]
    eigenvals2 = eigenvals2[idx]
    eigenvecs2 = eigenvecs2[:, idx]
    
    # 典型相关系数
    canonical_corrs = np.sqrt(np.real(eigenvals1[:n_components]))
    
    # 典型变量系数
    A = np.real(eigenvecs1[:, :n_components])
    B = np.real(eigenvecs2[:, :n_components])
    
    # 典型变量
    U = X_norm @ A
    V = Y_norm @ B
    
    return {
        'canonical_correlations': canonical_corrs,
        'x_coefficients': A,
        'y_coefficients': B,
        'x_canonical_vars': U,
        'y_canonical_vars': V,
        'x_loadings': np.corrcoef(X_norm.T, U.T)[:X.shape[1], X.shape[1]:],
        'y_loadings': np.corrcoef(Y_norm.T, V.T)[:Y.shape[1], Y.shape[1]:]
    }

# 分离X和Y变量组
X_vars = data[['x1', 'x2', 'x3']]
Y_vars = data[['y1', 'y2', 'y3']]

# 执行典型相关分析
cca_results = canonical_correlation_analysis(X_vars, Y_vars)

print("典型相关系数:")
for i, corr in enumerate(cca_results['canonical_correlations']):
    print(f"第{i+1}对典型变量相关系数: {corr:.4f}")

print("\nX变量组典型变量系数:")
x_coef_df = pd.DataFrame(cca_results['x_coefficients'], 
                        index=['x1', 'x2', 'x3'],
                        columns=[f'U{i+1}' for i in range(len(cca_results['canonical_correlations']))])
print(x_coef_df)

print("\nY变量组典型变量系数:")
y_coef_df = pd.DataFrame(cca_results['y_coefficients'],
                        index=['y1', 'y2', 'y3'], 
                        columns=[f'V{i+1}' for i in range(len(cca_results['canonical_correlations']))])
print(y_coef_df)

# 10.4.2 典型相关的实证分析
print("\n=== 10.4.2 典型相关的实证分析 ===")

# 典型相关系数显著性检验
def canonical_correlation_test(X, Y, canonical_corrs):
    """
    典型相关系数显著性检验
    """
    n = X.shape[0]
    p = X.shape[1]
    q = Y.shape[1]
    
    results = []
    for i, r in enumerate(canonical_corrs):
        # Wilks' Lambda检验
        lambda_val = 1 - r**2
        chi_stat = -(n - 1 - (p + q + 1)/2) * np.log(lambda_val)
        df = (p - i) * (q - i)
        p_value = 1 - chi2.cdf(chi_stat, df)
        
        results.append({
            'pair': i + 1,
            'correlation': r,
            'chi_square': chi_stat,
            'df': df,
            'p_value': p_value
        })
    
    return pd.DataFrame(results)

# 进行显著性检验
test_results = canonical_correlation_test(X_vars, Y_vars, cca_results['canonical_correlations'])
print("典型相关系数显著性检验:")
print(test_results)

# 绘制第一对典型变量散点图
plt.figure(figsize=(10, 6))
plt.scatter(cca_results['x_canonical_vars'][:, 0], 
           cca_results['y_canonical_vars'][:, 0], alpha=0.6)
plt.xlabel('第一典型变量 U1')
plt.ylabel('第一典型变量 V1')
plt.title(f'第一对典型变量散点图 (r = {cca_results["canonical_correlations"][0]:.4f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 案例10 R&D投入与产出的典型相关分析
print("\n=== 案例10 R&D投入与产出的典型相关分析 ===")

# 生成R&D案例数据（模拟中国29个省级区域数据）
np.random.seed(123)
provinces = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
           '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
           '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州',
           '云南', '西藏', '陕西', '甘肃', '宁夏']

# 生成R&D投入指标数据
rd_personnel = np.random.lognormal(8, 1.2, 29) * 10  # R&D人员全时当量
rd_expenditure = rd_personnel * np.random.uniform(80, 200, 29)  # R&D经费支出
tech_personnel = rd_personnel * np.random.uniform(0.6, 1.4, 29)  # 专业技术人员

# 生成R&D产出指标数据（与投入相关）
patent_applications = rd_expenditure * np.random.uniform(0.1, 0.5, 29) + np.random.normal(0, 1000, 29)
patent_grants = patent_applications * np.random.uniform(0.3, 0.8, 29)
tech_contracts = rd_expenditure * np.random.uniform(0.05, 0.2, 29) + np.random.normal(0, 500, 29)
contract_value = tech_contracts * np.random.uniform(100, 1000, 29)
hi_tech_output = rd_expenditure * np.random.uniform(2, 8, 29) + np.random.normal(0, 50000, 29)
hi_tech_sales = hi_tech_output * np.random.uniform(0.8, 1.2, 29)

# 创建R&D数据框
rd_data = pd.DataFrame({
    '省份': provinces,
    'x1_RD人员': rd_personnel,
    'x2_RD经费': rd_expenditure, 
    'x3_专业技术人员': tech_personnel,
    'y1_专利申请': patent_applications,
    'y2_专利授权': patent_grants,
    'y3_技术合同': tech_contracts,
    'y4_合同金额': contract_value,
    'y5_高新产值': hi_tech_output,
    'y6_高新销售': hi_tech_sales
})

print("R&D投入与产出数据样本:")
print(rd_data.head(10))

print("\n基本统计描述:")
print(rd_data.describe())

# R&D数据的典型相关分析
X_rd = rd_data[['x1_RD人员', 'x2_RD经费', 'x3_专业技术人员']]
Y_rd = rd_data[['y1_专利申请', 'y2_专利授权', 'y3_技术合同', 'y4_合同金额', 'y5_高新产值', 'y6_高新销售']]

# 执行典型相关分析
rd_cca_results = canonical_correlation_analysis(X_rd, Y_rd)

print("\nR&D投入与产出典型相关分析结果:")
print("典型相关系数:")
for i, corr in enumerate(rd_cca_results['canonical_correlations']):
    print(f"第{i+1}对典型变量相关系数: {corr:.4f}")

print("\nR&D投入指标典型变量系数:")
rd_x_coef = pd.DataFrame(rd_cca_results['x_coefficients'],
                        index=['RD人员', 'RD经费', '专业技术人员'],
                        columns=[f'U{i+1}' for i in range(len(rd_cca_results['canonical_correlations']))])
print(rd_x_coef)

print("\nR&D产出指标典型变量系数:")
rd_y_coef = pd.DataFrame(rd_cca_results['y_coefficients'],
                        index=['专利申请', '专利授权', '技术合同', '合同金额', '高新产值', '高新销售'],
                        columns=[f'V{i+1}' for i in range(len(rd_cca_results['canonical_correlations']))])
print(rd_y_coef)

# R&D典型相关系数检验
rd_test_results = canonical_correlation_test(X_rd, Y_rd, rd_cca_results['canonical_correlations'])
print("\nR&D典型相关系数显著性检验:")
print(rd_test_results)

# 绘制R&D第一对典型变量关系图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(rd_cca_results['x_canonical_vars'][:, 0], 
           rd_cca_results['y_canonical_vars'][:, 0], alpha=0.7, s=60)
plt.xlabel('R&D投入能力 U1')
plt.ylabel('R&D产出能力 V1') 
plt.title(f'第一对典型变量关系图\n(r = {rd_cca_results["canonical_correlations"][0]:.4f})')
plt.grid(True, alpha=0.3)

# 绘制典型相关系数柱状图
plt.subplot(1, 2, 2)
pairs = [f'第{i+1}对' for i in range(len(rd_cca_results['canonical_correlations']))]
plt.bar(pairs, rd_cca_results['canonical_correlations'], alpha=0.7, color='skyblue')
plt.ylabel('典型相关系数')
plt.title('各对典型变量相关系数')
plt.xticks(rotation=45)
for i, v in enumerate(rd_cca_results['canonical_correlations']):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 变量载荷分析
print("\nX变量组载荷矩阵（与典型变量的相关性）:")
x_loadings_df = pd.DataFrame(rd_cca_results['x_loadings'],
                           index=['RD人员', 'RD经费', '专业技术人员'],
                           columns=[f'U{i+1}' for i in range(len(rd_cca_results['canonical_correlations']))])
print(x_loadings_df)

print("\nY变量组载荷矩阵（与典型变量的相关性）:")
y_loadings_df = pd.DataFrame(rd_cca_results['y_loadings'],
                           index=['专利申请', '专利授权', '技术合同', '合同金额', '高新产值', '高新销售'],
                           columns=[f'V{i+1}' for i in range(len(rd_cca_results['canonical_correlations']))])
print(y_loadings_df)

# 冗余度分析
def redundancy_analysis(X, Y, cca_results):
    """
    计算冗余度
    """
    n_pairs = len(cca_results['canonical_correlations'])
    
    X_redundancy = []
    Y_redundancy = []
    
    for i in range(n_pairs):
        # X对Y的冗余度
        r_squared = cca_results['canonical_correlations'][i] ** 2
        x_variance_extracted = np.mean(cca_results['x_loadings'][:, i] ** 2)
        x_redundancy = r_squared * x_variance_extracted
        X_redundancy.append(x_redundancy)
        
        # Y对X的冗余度  
        y_variance_extracted = np.mean(cca_results['y_loadings'][:, i] ** 2)
        y_redundancy = r_squared * y_variance_extracted
        Y_redundancy.append(y_redundancy)
    
    return X_redundancy, Y_redundancy

x_redundancy, y_redundancy = redundancy_analysis(X_rd, Y_rd, rd_cca_results)

print("\n冗余度分析:")
redundancy_df = pd.DataFrame({
    '典型变量对': [f'第{i+1}对' for i in range(len(rd_cca_results['canonical_correlations']))],
    'X对Y冗余度': x_redundancy,
    'Y对X冗余度': y_redundancy
})
print(redundancy_df)

print("\n=== 分析总结 ===")
print("1. 第一对典型变量相关系数最高，表明R&D投入能力与产出能力存在显著正相关关系")
print("2. R&D经费支出在投入能力中权重最大，专利申请和高新技术产值在产出能力中贡献突出")
print("3. 典型相关分析有效揭示了R&D投入与产出之间的潜在关联结构")
print("4. 可为区域科技创新政策制定提供数据支撑")