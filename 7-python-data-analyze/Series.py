import pandas as pd
import numpy as np

# 1. Series 基础操作
print("===== Series 基础操作 =====")
# Series 是 pandas 中的一种数据结构，类似于一维数组，可以存储任意数据类型，并且可以使用自定义索引。

# 创建 Series
s1 = pd.Series([1, 2, 3, 4, 5])  # 从列表创建
print("基础 Series:")
print(s1)
print()

# 自定义索引
s2 = pd.Series([90, 86, 95], index=['语文', '数学', '英语'])
print("自定义索引的 Series:")
print(s2)
print()

# Series 基本运算
print("Series 加法:")
print(s2 + 10)  # 所有元素加10
print()

# Series 切片和索引
print("获取单个值:", s2['语文'])
print("切片获取:", s2[['语文', '英语']])
print()

# 2. DataFrame 基础操作
print("===== DataFrame 基础操作 =====")
# DataFrame 是 pandas 中的一个数据结构，类似于二维数组或表格，可以存储不同类型的数据。每一列可以是不同的数据类型，且每一行和每一列都可以使用自定义索引。DataFrame 提供了丰富的功能来处理和分析数据。

# 创建 DataFrame
data = {
    '姓名': ['小明', '小红', '小华'],
    '年龄': [15, 16, 15],
    '成绩': [90, 85, 88]
}
df = pd.DataFrame(data)
print("基础 DataFrame:")
print(df)
print()

# 选择列
print("选择单列:")
print(df['姓名'])
print()

# 选择多列
print("选择多列:")
print(df[['姓名', '成绩']])
print()

# 添加新列
df['及格'] = df['成绩'] >= 85
print("添加新列后的 DataFrame:")
print(df)
print()

# 基本统计
print("成绩统计信息:")
print(df['成绩'].describe())
print()

# 排序
print("按成绩排序:")
print(df.sort_values(by='成绩', ascending=False))
print()

# 条件筛选
print("成绩大于85的学生:")
print(df[df['成绩'] > 85])
