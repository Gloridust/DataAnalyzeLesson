import pandas as pd

def cross_analysis(df, values, index, columns, aggfunc, fill_value=0, bins=None, labels=None):
    """
    执行交叉分析
    params:
        df: DataFrame对象
        values: 统计值列名
        index: 行索引列名
        columns: 列索引列名
        aggfunc: 统计函数
        fill_value: 空值填充值
        bins: 分组区间
        labels: 分组标签
    return: 交叉分析结果
    """
    # 如果需要分组，先进行数据分组
    if bins is not None and labels is not None:
        group_name = f"{values}_分组"
        df[group_name] = pd.cut(df[values], bins=bins, labels=labels)
        # 如果index是需要分组的列，更新index为分组后的列名
        if index == values:
            index = group_name
            
    cross_table = pd.pivot_table(
        df,
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        fill_value=fill_value
    )
    return cross_table

# 使用示例
if __name__ == "__main__":
    # 读取数据
    df = pd.read_excel('rz4.xlsx')
    
    # 设置分组区间和标签
    bins = [df['总分'].min() - 1, 450, 500, df['总分'].max() + 1]
    labels = ['450及以下', '450-500', '500以上']
    
    # 分析总分和性别的关系，包含分组
    result = cross_analysis(
        df,
        values='总分',
        index='总分',  # 这里会被自动转换为分组后的列名
        columns='性别',
        aggfunc=['count', 'mean'],
        bins=bins,
        labels=labels
    )
    print("交叉分析结果：\n", result)
