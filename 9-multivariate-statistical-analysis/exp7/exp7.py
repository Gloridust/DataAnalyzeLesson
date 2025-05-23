import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def correspondence_analysis(data):
    P = data / data.sum().sum()
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    
    Dr = np.diag(1/np.sqrt(r))
    Dc = np.diag(1/np.sqrt(c))
    
    S = Dr @ (P - np.outer(r, c)) @ Dc
    
    U, sigma, Vt = np.linalg.svd(S)
    
    F = Dr @ U
    G = Dc @ Vt.T
    
    eigenvalues = sigma**2
    inertia = eigenvalues / eigenvalues.sum() * 100
    
    return F, G, eigenvalues, inertia

def CA_plot(ca_result, df, title="对应分析图"):
    F, G, eigenvalues, inertia = ca_result
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    row_coords = F[:, :2]
    col_coords = G[:, :2]
    
    ax.scatter(row_coords[:, 0], row_coords[:, 1], c='red', s=100, alpha=0.7, label='行类别')
    ax.scatter(col_coords[:, 0], col_coords[:, 1], c='blue', s=100, alpha=0.7, label='列类别')
    
    for i, txt in enumerate(df.index):
        ax.annotate(txt, (row_coords[i, 0], row_coords[i, 1]), fontsize=9)
    
    for i, txt in enumerate(df.columns):
        ax.annotate(txt, (col_coords[i, 0], col_coords[i, 1]), fontsize=9)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    ax.set_xlabel(f'第一维 ({inertia[0]:.2f}% 惯性)')
    ax.set_ylabel(f'第二维 ({inertia[1]:.2f}% 惯性)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_income_education_data():
    data = {
        '低收入户': [13.53, 69.77, 97.69, 14.00, 3.77, 1.24],
        '中低收入户': [3.68, 29.14, 55.28, 9.20, 2.33, 0.37],
        '中等收入户': [3.51, 24.99, 56.36, 11.05, 3.28, 0.81],
        '中高收入户': [3.09, 20.96, 57.93, 12.54, 3.74, 1.74],
        '高收入户': [2.24, 19.75, 49.85, 17.50, 6.72, 3.94]
    }
    
    index = ['文盲或半文盲', '小学程度', '初中程度', '高中程度', '中专程度', '大专程度']
    
    df = pd.DataFrame(data, index=index)
    return df

def create_income_source_data():
    data = {
        '低收入户': [52.49, 280.34, 388.23, 535.60, 3480.68, 159.99, 34.32],
        '中低收入户': [73.87, 257.72, 940.18, 358.95, 2069.17, 158.30, 32.57],
        '中等收入户': [156.25, 322.94, 1511.76, 291.32, 2244.54, 239.27, 63.95],
        '中高收入户': [227.37, 299.17, 2484.98, 303.71, 2782.37, 344.35, 119.43],
        '高收入户': [741.94, 1297.58, 2870.31, 475.49, 6479.68, 661.23, 699.20]
    }
    
    index = ['在非企业组织中得到', '在本地企业中得到', '常住人口外出得到', 
             '其他工资性收入', '家庭经营收入', '转移性收入', '财产性收入']
    
    df = pd.DataFrame(data, index=index)
    return df

def analyze_correspondence(df, title):
    print(f"\n{title}")
    print("="*50)
    
    print("原始数据:")
    print(df)
    
    chi2, p_value, dof, expected = chi2_contingency(df)
    print(f"\n卡方检验:")
    print(f"卡方统计量: {chi2:.4f}")
    print(f"p值: {p_value:.4f}")
    print(f"自由度: {dof}")
    
    ca_result = correspondence_analysis(df)
    F, G, eigenvalues, inertia = ca_result
    
    print(f"\n特征值:")
    for i, (eig, iner) in enumerate(zip(eigenvalues, inertia)):
        print(f"维度 {i+1}: {eig:.4f} (贡献率: {iner:.2f}%)")
    
    print(f"\n累积贡献率:")
    cumulative_inertia = np.cumsum(inertia)
    for i, cum_iner in enumerate(cumulative_inertia):
        print(f"前 {i+1} 维: {cum_iner:.2f}%")
    
    print(f"\n行坐标 (前3维):")
    row_coords_df = pd.DataFrame(F[:, :3], 
                                index=df.index, 
                                columns=['Dim1', 'Dim2', 'Dim3'])
    print(row_coords_df)
    
    print(f"\n列坐标 (前3维):")
    col_coords_df = pd.DataFrame(G[:, :3], 
                                index=df.columns, 
                                columns=['Dim1', 'Dim2', 'Dim3'])
    print(col_coords_df)
    
    fig = CA_plot(ca_result, df, title)
    plt.show()
    
    return ca_result

def main():
    df1 = create_income_education_data()
    ca_result1 = analyze_correspondence(df1, "收入水平与教育程度对应分析")
    
    df2 = create_income_source_data()
    ca_result2 = analyze_correspondence(df2, "收入水平与收入来源对应分析")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    F1, G1, eigenvalues1, inertia1 = ca_result1
    row_coords1 = F1[:, :2]
    col_coords1 = G1[:, :2]
    
    axes[0].scatter(row_coords1[:, 0], row_coords1[:, 1], c='red', s=100, alpha=0.7)
    axes[0].scatter(col_coords1[:, 0], col_coords1[:, 1], c='blue', s=100, alpha=0.7)
    
    for i, txt in enumerate(df1.index):
        axes[0].annotate(txt, (row_coords1[i, 0], row_coords1[i, 1]), fontsize=8)
    for i, txt in enumerate(df1.columns):
        axes[0].annotate(txt, (col_coords1[i, 0], col_coords1[i, 1]), fontsize=8)
    
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_title('收入水平与教育程度')
    axes[0].grid(True, alpha=0.3)
    
    F2, G2, eigenvalues2, inertia2 = ca_result2
    row_coords2 = F2[:, :2]
    col_coords2 = G2[:, :2]
    
    axes[1].scatter(row_coords2[:, 0], row_coords2[:, 1], c='red', s=100, alpha=0.7)
    axes[1].scatter(col_coords2[:, 0], col_coords2[:, 1], c='blue', s=100, alpha=0.7)
    
    for i, txt in enumerate(df2.index):
        axes[1].annotate(txt, (row_coords2[i, 0], row_coords2[i, 1]), fontsize=8)
    for i, txt in enumerate(df2.columns):
        axes[1].annotate(txt, (col_coords2[i, 0], col_coords2[i, 1]), fontsize=8)
    
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_title('收入水平与收入来源')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()