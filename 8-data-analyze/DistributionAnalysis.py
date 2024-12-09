import pandas as pd
import numpy as np

def distribution_analysis(df, column_name, bins, labels=None):
    """
    执行分布分析
    params:
        df: DataFrame对象
        column_name: 要分析的列名
        bins: 分组区间
        labels: 分组标签
    return: 分布统计结果
    """
    # 创建分组
    df['分组'] = pd.cut(df[column_name], bins=bins, labels=labels)
    
    # 统计各分组频数和频率
    dist_stats = df['分组'].value_counts().sort_index()
    dist_props = df['分组'].value_counts(normalize=True).sort_index()
    
    return pd.DataFrame({
        '频数': dist_stats,
        '频率': dist_props
    })

# 使用示例
if __name__ == "__main__":
    # 读取数据
    df = pd.read_excel('rz4.xlsx')
    
    # 对总分进行分布分析
    bins = [df['总分'].min()-1, 450, 500, df['总分'].max()+1]
    labels = ['450及其以下', '450到500', '500及其以上']
    result = distribution_analysis(df, '总分', bins, labels)
    print("总分分布分析结果：\n", result)
