import pandas as pd
import numpy as np

def grouping_analysis(df, group_columns, value_column, agg_functions):
    """
    执行分组统计分析
    """
    grouped_result = df.groupby(by=group_columns).agg({
        value_column: agg_functions
    })
    return grouped_result

# 使用示例
if __name__ == "__main__":
    # 读取数据
    df = pd.read_excel('rz4.xlsx')
    
    # 定义聚合函数
    agg_funcs = ['sum', 'size', 'mean', 'var', 'std', 'max', 'min']
    
    # 执行分组分析
    result = grouping_analysis(df, ['班级', '性别'], '军训', agg_funcs)
    
    # 重命名列名使其更易读
    result.columns = ['总分', '人数', '平均值', '方差', '标准差', '最高分', '最低分']
    print("分组统计结果：\n", result)
