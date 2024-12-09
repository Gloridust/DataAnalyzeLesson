import pandas as pd

def correlation_analysis(df, columns=None, col1=None, col2=None, method='pearson'):
    """
    执行相关分析
    params:
        df: DataFrame对象
        columns: 要分析的列名列表
        col1: 第一个要比较的列名
        col2: 第二个要比较的列名
        method: 相关系数计算方法 ('pearson', 'spearman', 'kendall')
    return: 相关系数矩阵或单个相关系数
    """
    if col1 and col2:
        # 计算两列之间的相关系数
        return df[col1].corr(df[col2], method=method)
    elif columns:
        # 计算指定列之间的相关系数矩阵
        correlation_matrix = df[columns].corr(method=method)
    else:
        # 计算所有数值列的相关系数矩阵
        correlation_matrix = df.corr(method=method)
    return correlation_matrix

# 使用示例
if __name__ == "__main__":
    # 读取数据
    df = pd.read_excel('rz4.xlsx')
    
    # 两列之间的相关度计算
    corr_value = correlation_analysis(df, col1='高代', col2='数分')
    print(f"高代和数分的相关系数：{corr_value}")
    
    # 多列之间的相关度计算
    columns = ['英语', '体育', '军训', '计算机基础', '解几', '数分', '高代']
    result = correlation_analysis(df, columns)
    print("\n多列相关分析结果：\n", result)
