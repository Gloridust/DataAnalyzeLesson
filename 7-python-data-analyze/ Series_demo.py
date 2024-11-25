import pandas as pd
import numpy as np

print("===== Series 基础操作 =====")

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

print("===== Series 高级操作 =====")

# 追加序列
s3 = pd.Series([88, 92], index=['物理', '化学'])
s4 = s2.append(s3)
print("追加后的 Series:")
print(s4)
print()

# 根据 index 删除
s5 = s4.drop('物理')
print("删除'物理'后的 Series:")
print(s5)
print()

# 根据值删除
s6 = s4[s4 > 90]
print("只保留大于90分的科目:")
print(s6)
print()

# 修改 Series 的 index
s7 = s2.copy()
s7.index = ['Chinese', 'Math', 'English']
print("修改索引后的 Series:")
print(s7)
print()

# 重排序
print("按值排序的 Series:")
print(s4.sort_values(ascending=False))
