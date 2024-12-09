import pandas as pd

def basic_statistical_analysis(df, column_name):
    """
    执行基本统计分析
    params:
        df: DataFrame对象
        column_name: 要分析的列名
    return: 基本统计结果字典
    """
    # 基本统计描述
    # describe()方法会返回以下统计量:
    # count: 非空值数量
    # mean: 平均值
    # std: 标准差
    # min: 最小值
    # 25%: 第一四分位数
    # 50%: 中位数
    # 75%: 第三四分位数
    # max: 最大值
    basic_stats = df[column_name].describe()
    
    # 附加统计指标
    additional_stats = {
        '总和': df[column_name].sum(),
        '方差': df[column_name].var(),
        '标准差': df[column_name].std(),
        '计数': df[column_name].size  # size是属性而不是方法，所以不需要括号
    }
    
    return basic_stats, additional_stats

# 使用示例
if __name__ == "__main__":
    # 从CSV文件读取数据
    df = pd.read_csv('./rz3.csv')
    
    # 对num列进行分析
    basic_stats, additional_stats = basic_statistical_analysis(df, 'num')
    
    print("基本统计描述：")
    print(f"数据计数：{basic_stats['count']:.2f}")
    print(f"平均值：{basic_stats['mean']:.2f}")
    print(f"标准差：{basic_stats['std']:.2f}")
    print(f"最小值：{basic_stats['min']:.2f}")
    print(f"25%分位：{basic_stats['25%']:.2f}")
    print(f"中位数：{basic_stats['50%']:.2f}")
    print(f"75%分位：{basic_stats['75%']:.2f}")
    print(f"最大值：{basic_stats['max']:.2f}")
