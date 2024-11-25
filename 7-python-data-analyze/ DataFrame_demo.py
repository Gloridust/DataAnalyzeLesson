import pandas as pd
import numpy as np

print("===== DataFrame 基础操作 =====")

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
print()

print("===== DataFrame 高级操作 =====")

# 按列访问
print("访问'年龄'列:")
print(df['年龄'])
print()

# 按行访问
print("访问第2行:")
print(df[1:2])
print()

# 按行列号访问
print("访问第1行第1列:")
print(df.iloc[0:1, 0:1])
print()

# 创建带索引的 DataFrame 演示 at 访问
df_indexed = df.copy()
df_indexed.index = ['first', 'second', 'third']
print("按行索引、列名访问:")
print(df_indexed.at['first', '姓名'])
print()

# DataFrame append 示例
new_student = pd.DataFrame({
    '姓名': ['小李'],
    '年龄': [16],
    '成绩': [93],
    '及格': [True]
}, index=['fourth'])
df_appended = df_indexed.append(new_student)
print("追加新行后的 DataFrame:")
print(df_appended)
print()

# shift 操作
print("成绩列向下移动一位:")
print(df['成绩'].shift(1))
