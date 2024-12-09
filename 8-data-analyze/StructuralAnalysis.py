import pandas as pd

def structural_analysis(df, values, index, columns, aggfunc='sum'):
    """
    执行结构分析
    params:
        df: DataFrame对象
        values: 统计值列名或列名列表
        index: 行索引列名
        columns: 列索引列名
        aggfunc: 统计函数或函数字典
    return: 结构分析结果（包含行比重和列比重）
    """
    # 创建透视表 - 支持多个values和aggfunc
    pivot_table = pd.pivot_table(
        df,
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        margins=True  # 添加总计行和列
    )
    
    # 计算行比重
    row_proportions = pivot_table.div(pivot_table.sum(axis=1), axis=0)
    
    # 计算列比重
    col_proportions = pivot_table.div(pivot_table.sum(axis=0), axis=1)
    
    return {
        '原始数据': pivot_table,
        '行比重': row_proportions,
        '列比重': col_proportions
    }

# 使用示例
if __name__ == "__main__":
    # 读取数据
    df = pd.read_excel('rz4.xlsx')
    
    # 分析班级和性别的总分结构
    result = structural_analysis(df, '总分', '班级', '性别')
    print("结构分析结果：")
    for key, value in result.items():
        print(f"\n{key}:\n", value)
