# -*- coding: utf-8 -*-
"""
Python基础与数据分析综合示例
包含Python编程基础、数据分析、多维数据处理和数据框的示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

print("="*50)
print("2.1 Python的编程基础")
print("="*50)

# 2.1.1 Python的数据类型
print("\n2.1.1 Python的数据类型")
print("-"*30)

# 1. Python基本类型
print("1. Python基本类型:")
basic_int = 10
basic_float = 3.14
basic_str = "Hello, Python!"
basic_bool = True
basic_none = None

print(f"整数: {basic_int}, 类型: {type(basic_int)}")
print(f"浮点数: {basic_float}, 类型: {type(basic_float)}")
print(f"字符串: {basic_str}, 类型: {type(basic_str)}")
print(f"布尔值: {basic_bool}, 类型: {type(basic_bool)}")
print(f"空值: {basic_none}, 类型: {type(basic_none)}")

# 2. Python扩展类型
print("\n2. Python扩展类型:")
list_example = [1, 2, 3, 4, 5]
tuple_example = (1, 2, 3, 4, 5)
dict_example = {"name": "Python", "version": 3.10}
set_example = {1, 2, 3, 4, 5}

print(f"列表: {list_example}, 类型: {type(list_example)}")
print(f"元组: {tuple_example}, 类型: {type(tuple_example)}")
print(f"字典: {dict_example}, 类型: {type(dict_example)}")
print(f"集合: {set_example}, 类型: {type(set_example)}")

# 2.1.2 运算符及控制语句
print("\n2.1.2 运算符及控制语句")
print("-"*30)

# 1. 条件语句
print("1. 条件语句:")
x = 10
if x > 0:
    print("x是正数")
elif x < 0:
    print("x是负数")
else:
    print("x是零")

# 2. 循环语句
print("\n2. 循环语句:")
print("for循环示例:")
for i in range(5):
    print(f"循环计数: {i}")

print("\nwhile循环示例:")
count = 0
while count < 3:
    print(f"while计数: {count}")
    count += 1

# 2.1.3 Python函数的使用
print("\n2.1.3 Python函数的使用")
print("-"*30)

# 1. 内置函数
print("1. 内置函数:")
numbers = [1, 2, 3, 4, 5]
print(f"最大值: {max(numbers)}")
print(f"最小值: {min(numbers)}")
print(f"总和: {sum(numbers)}")
print(f"长度: {len(numbers)}")
print(f"排序: {sorted(numbers, reverse=True)}")

# 2. 包及安装
print("\n2. 包及安装:")
print("本示例使用了numpy, pandas和matplotlib包")
print(f"numpy版本: {np.__version__}")
print(f"pandas版本: {pd.__version__}")

# 3. 和函数使用
print("\n3. 包函数使用:")
random_array = np.random.rand(5)
print(f"随机数组: {random_array}")
print(f"均值: {np.mean(random_array)}")
print(f"标准差: {np.std(random_array)}")

# 4. 自定义函数
print("\n4. 自定义函数:")
def calculate_stats(numbers):
    """计算一组数字的统计信息"""
    stats = {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers)
    }
    return stats

example_numbers = [10, 20, 30, 40, 50]
stats_result = calculate_stats(example_numbers)
for key, value in stats_result.items():
    print(f"{key}: {value}")

print("\n="*50)
print("2.2 Python数值分析")
print("="*50)

# 2.2.1 一维数组（向量）运算
print("\n2.2.1 一维数组（向量）运算")
print("-"*30)

# 创建向量
vector1 = np.array([1, 2, 3, 4, 5])
vector2 = np.array([5, 4, 3, 2, 1])

print(f"向量1: {vector1}")
print(f"向量2: {vector2}")
print(f"向量加法: {vector1 + vector2}")
print(f"向量减法: {vector1 - vector2}")
print(f"向量乘法(元素级): {vector1 * vector2}")
print(f"向量除法(元素级): {vector1 / vector2}")
print(f"向量点积: {np.dot(vector1, vector2)}")
print(f"向量1的范数: {np.linalg.norm(vector1)}")

# 2.2.2 二维数组（矩阵）运算
print("\n2.2.2 二维数组（矩阵）运算")
print("-"*30)

# 创建矩阵
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

print("矩阵1:")
print(matrix1)
print("\n矩阵2:")
print(matrix2)
print("\n矩阵加法:")
print(matrix1 + matrix2)
print("\n矩阵乘法(元素级):")
print(matrix1 * matrix2)
print("\n矩阵乘法(矩阵乘法):")
print(np.matmul(matrix1, matrix2))
print("\n矩阵1的转置:")
print(matrix1.T)
print("\n矩阵1的行列式:")
print(np.linalg.det(matrix1))

print("\n="*50)
print("2.3 多元数据的收集和整理")
print("="*50)

# 2.3.1 数据格式
print("\n2.3.1 数据格式")
print("-"*30)

# 创建结构化数据
student_data = [
    {"id": 1, "name": "Alice", "age": 20, "score": 85},
    {"id": 2, "name": "Bob", "age": 22, "score": 92},
    {"id": 3, "name": "Charlie", "age": 21, "score": 78},
    {"id": 4, "name": "Diana", "age": 23, "score": 95},
    {"id": 5, "name": "Eve", "age": 19, "score": 88}
]

print("学生数据示例:")
for student in student_data:
    print(student)

# 2.3.2 变量的分类
print("\n2.3.2 变量的分类")
print("-"*30)

print("变量类型示例:")
print("1. 数值型变量: 年龄(age), 分数(score)")
print("2. 分类变量: 学生ID(id)")
print("3. 文本变量: 姓名(name)")

# 将数据转换为DataFrame
df_students = pd.DataFrame(student_data)
print("\n使用Pandas DataFrame:")
print(df_students)
print("\n数据类型:")
print(df_students.dtypes)

print("\n="*50)
print("2.4 Python的数据框")
print("="*50)

# 2.4.1 数据框的构成
print("\n2.4.1 数据框的构成")
print("-"*30)

# 1. 从数据读取
print("1. 从数组构建DataFrame:")
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'age': [25, 30, 35, 40, 45],
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
    'salary': [50000, 60000, 70000, 80000, 90000]
}
df = pd.DataFrame(data)
print(df)

# 2. 从CSV读取
print("\n2. 从CSV文件读取:")
# 创建一个临时CSV文件
temp_csv_path = "temp_data.csv"
df.to_csv(temp_csv_path, index=False)
df_from_csv = pd.read_csv(temp_csv_path)
print("从CSV读取的数据:")
print(df_from_csv)

# 3. 数据存储
print("\n3. 数据的保存:")
df.to_csv("output_data.csv", index=False)
df.to_excel("output_data.xlsx", index=False)
print(f"数据已保存到 output_data.csv 和 output_data.xlsx")

# 2.4.2 数据框的应用
print("\n2.4.2 数据框的应用")
print("-"*30)

# 创建更复杂的示例数据
dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
df_sales = pd.DataFrame({
    'date': dates,
    'product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
    'sales': np.random.randint(50, 200, size=10),
    'price': np.random.uniform(10.0, 50.0, size=10).round(2),
    'region': np.random.choice(['East', 'West', 'North', 'South'], size=10)
})

print("销售数据示例:")
print(df_sales)

# 2.4.3 数据框的基本操作
print("\n2.4.3 数据框的基本操作")
print("-"*30)

# 1. 数据框信息
print("1. 数据框信息:")
print(df_sales.info())
print("\n数据统计摘要:")
print(df_sales.describe())

# 2. 数据框显示
print("\n2. 数据框显示:")
print("前5行:")
print(df_sales.head())
print("\n后5行:")
print(df_sales.tail())

# 3. 数据框取值
print("\n3. 数据框取值:")
print("选择'sales'列:")
print(df_sales['sales'])
print("\n选择前3行:")
print(df_sales.iloc[:3])
print("\n选择sales>100的行:")
print(df_sales[df_sales['sales'] > 100])
print("\n按region和product分组统计销售总额:")
print(df_sales.groupby(['region', 'product'])['sales'].sum())

# 可视化示例
print("\n数据可视化示例:")
plt.rcParams['font.family'] = 'Songti SC'
plt.figure(figsize=(10, 6))

# 产品销售量比较
plt.subplot(2, 2, 1)
plt.rcParams['font.family'] = 'Songti SC'
product_sales = df_sales.groupby('product')['sales'].sum()
product_sales.plot(kind='bar', title='产品销售量')

# 区域销售量比较
plt.subplot(2, 2, 2)
plt.rcParams['font.family'] = 'Songti SC'
region_sales = df_sales.groupby('region')['sales'].sum()
region_sales.plot(kind='pie', autopct='%1.1f%%', title='区域销售占比')

# 日期销售趋势
plt.subplot(2, 2, 3)
plt.rcParams['font.family'] = 'Songti SC'
df_sales.set_index('date')['sales'].plot(title='销售趋势')

# 价格与销售量的关系
plt.rcParams['font.family'] = 'Songti SC'
plt.subplot(2, 2, 4)
plt.scatter(df_sales['price'], df_sales['sales'])
plt.title('价格与销售量关系')
plt.xlabel('价格')
plt.ylabel('销售量')

plt.tight_layout()
plt.savefig('sales_analysis.png')
print("可视化结果已保存到 'sales_analysis.png'")

# 清理临时文件
if os.path.exists(temp_csv_path):
    os.remove(temp_csv_path)

print("\n示例完成!")