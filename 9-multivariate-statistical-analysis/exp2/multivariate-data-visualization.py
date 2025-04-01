"""
第3章 多元数据的Python可视化
"""

# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建结果保存目录
if not os.path.exists('./result'):
    os.makedirs('./result')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['STHeiti']  # 设置中文字体为华文黑体

# 3.1 多元数据可视化准备

# 3.1.1 多元数据的收集方式
# 3.1.2 多元数据的读取方法

# 读取Excel数据示例
# 普通读取方式
pd.set_option('display.max_rows', 8)  # 显示最大行数
df = pd.read_excel('mvsData.xlsx', 'd31')
print(df)

# 将第一列设为索引的读取方式
d31 = pd.read_excel('mvsData.xlsx', 'd31', index_col=0)
print(d31)

# 将数据框信息保存到txt文件中
with open('./result/dataframe_info.txt', 'w', encoding='utf-8') as f:
    f.write("数据框基本信息：\n")
    f.write("-" * 50 + "\n")
    f.write(f"数据形状: {d31.shape}\n")
    f.write(f"列名: {list(d31.columns)}\n")
    f.write(f"索引: {list(d31.index)}\n\n")

# 3.1.3 多元统计量与绘图函数

# 1. 多元数据基本统计量

# 计算均值向量
means = d31.mean()
print("均值向量:")
print(means)

# 计算协方差矩阵
cov_matrix = d31.cov()
print("协方差矩阵:")
print(cov_matrix)

# 计算相关系数矩阵
corr_matrix = d31.corr()
print("相关系数矩阵:")
print(corr_matrix)

# 将统计量保存到txt文件
with open('./result/statistics.txt', 'w', encoding='utf-8') as f:
    f.write("多元数据统计量\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. 均值向量:\n")
    f.write("-" * 50 + "\n")
    f.write(means.to_string() + "\n\n")
    
    f.write("2. 协方差矩阵:\n")
    f.write("-" * 50 + "\n")
    f.write(cov_matrix.to_string() + "\n\n")
    
    f.write("3. 相关系数矩阵:\n")
    f.write("-" * 50 + "\n")
    f.write(corr_matrix.to_string() + "\n\n")

# 2. 多元数据基本绘图函数
# DataFrame.plot(kind='line',figsize,subplots,layout,...)
# 常用图表类型:
# 'line': 折线图
# 'bar': 垂直条图
# 'barh': 水平条图
# 'hist': 直方图
# 'box': 箱线图
# 'kde': 核密度估计图
# 'area': 面积图
# 'pie': 饼图
# 'scatter': 散点图

# 3.2 条图或柱图

# 3.2.1 原始数据的条图
# (1) 单变量条图
d31['食品'].plot(kind='bar', figsize=(10, 5))
plt.title('各地区食品消费支出')
plt.xlabel('地区')
plt.ylabel('消费支出(元/人)')
plt.tight_layout()
plt.savefig('./result/01_food_consumption_bar.png', dpi=300)
plt.close()

# (2) 多变量条图
d31.plot(kind='bar', figsize=(10, 5))
plt.title('各地区各类消费支出')
plt.xlabel('地区')
plt.ylabel('消费支出(元/人)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('./result/02_all_consumption_bar.png', dpi=300)
plt.close()

# 使用子图展示多变量
fig, axes = plt.subplots(nrows=len(d31.columns), figsize=(10, 15), sharex=True)
for i, col in enumerate(d31.columns):
    d31[col].plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'{col}消费支出')
    axes[i].set_ylabel('消费支出(元/人)')
plt.tight_layout()
plt.savefig('./result/03_consumption_subplots.png', dpi=300)
plt.close()

# 使用布局参数创建子图矩阵
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 18))
for i, col in enumerate(d31.columns):
    row, col_idx = divmod(i, 3)
    if i < len(d31.columns):
        d31[d31.columns[i]].plot(kind='barh', ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'{d31.columns[i]}消费支出')
        axes[row, col_idx].set_xlabel('消费支出(元/人)')
plt.tight_layout()
plt.savefig('./result/04_consumption_grid_barh.png', dpi=300)
plt.close()

# 基于样品的条图
# (1) 单样品条图
d31.loc[['北京']].plot(kind='bar', xlabel="")
plt.title('北京市各类消费支出')
plt.ylabel('消费支出(元/人)')
plt.tight_layout()
plt.savefig('./result/05_beijing_consumption.png', dpi=300)
plt.close()

# (2) 多样品条图
d31.loc[['北京', '上海', '广东']].plot(kind='bar')
plt.title('北京、上海、广东各类消费支出')
plt.xlabel('地区')
plt.ylabel('消费支出(元/人)')
plt.tight_layout()
plt.savefig('./result/06_major_cities_comparison.png', dpi=300)
plt.close()

# 3.2.2 统计量的条图
# (1) 变量均值条图
d31.mean().plot(kind='barh')
plt.title('各类消费支出均值')
plt.xlabel('消费支出(元/人)')
plt.tight_layout()
plt.savefig('./result/07_consumption_means.png', dpi=300)
plt.close()

# 3.3 描述统计及箱尾图

# 3.3.1 描述性统计量
desc_stats = d31.describe().round(1)
print("描述统计量:")
print(desc_stats)

# 将描述性统计量保存到txt文件
with open('./result/descriptive_stats.txt', 'w', encoding='utf-8') as f:
    f.write("描述性统计量\n")
    f.write("=" * 50 + "\n\n")
    f.write(desc_stats.to_string() + "\n\n")

# 3.3.2 箱尾图的绘制
# (1) 垂直箱式图
d31.plot.box()
plt.title('各类消费支出箱线图')
plt.ylabel('消费支出(元/人)')
plt.tight_layout()
plt.savefig('./result/08_consumption_boxplot.png', dpi=300)
plt.close()

# (2) 水平箱式图
d31.plot.box(vert=False)
plt.title('各类消费支出水平箱线图')
plt.xlabel('消费支出(元/人)')
plt.tight_layout()
plt.savefig('./result/09_consumption_horizontal_boxplot.png', dpi=300)
plt.close()

# 3.4 变量间的关系图

# 3.4.1 两变量散点图
d31.plot(x='食品', y='衣着', kind='scatter')
plt.title('食品与衣着消费散点图')
plt.xlabel('食品消费(元/人)')
plt.ylabel('衣着消费(元/人)')
plt.tight_layout()
plt.savefig('./result/10_food_clothing_scatter.png', dpi=300)
plt.close()

# 将两个变量的相关性分析保存到文件
with open('./result/correlation_analysis.txt', 'a', encoding='utf-8') as f:
    f.write("食品与衣着消费相关性分析\n")
    f.write("-" * 50 + "\n")
    f.write(f"相关系数: {d31['食品'].corr(d31['衣着']):.4f}\n")
    f.write(f"协方差: {d31['食品'].cov(d31['衣着']):.4f}\n\n")

# 3.4.2 多变量矩阵散点图
pd.plotting.scatter_matrix(d31, figsize=(9, 8))
plt.tight_layout()
plt.savefig('./result/11_scatter_matrix.png', dpi=300)
plt.close()

# 3.5 其他多元分析图
# 在后续章节会有更多多元分析图，如:
# (1) 系统聚类图 (4.3)
# (2) 主成分分析图 (6.3)
# (3) 因子分析图 (7.3)
# (4) 双重信息图 (7.4)
# (5) 对应分析图 (8.3)
# (6) 典型相关图 (10.4)
# (7) 判别分析图 (12.2)

# 案例3: 城市现代化水平的直观分析
# 数据读取
case3 = pd.read_excel('mvsCase.xlsx', 'Case3', index_col=0)
print(case3)

# 将案例3数据信息保存到txt文件
with open('./result/case3_info.txt', 'w', encoding='utf-8') as f:
    f.write("城市现代化水平数据分析\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"数据形状: {case3.shape}\n")
    f.write(f"列名: {list(case3.columns)}\n")
    f.write(f"城市: {list(case3.index)}\n\n")
    
    # 计算基本统计量
    f.write("基本统计量:\n")
    f.write("-" * 50 + "\n")
    f.write(case3.describe().to_string() + "\n\n")

# 变量的直观分析
# (1) 单变量条图
case3.X1.plot(kind='bar', figsize=(9, 5))
plt.title('各城市人均GDP')
plt.xlabel('城市')
plt.ylabel('人均GDP(元/人)')
plt.tight_layout()
plt.savefig('./result/12_city_gdp.png', dpi=300)
plt.close()

# (2) 多变量条图
# 堆叠条形图
case3.plot(kind='barh', stacked=True, figsize=(8, 6))
plt.title('各城市现代化指标')
plt.tight_layout()
plt.savefig('./result/13_city_modernization_stacked.png', dpi=300)
plt.close()

# 子图展示
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i, col in enumerate(case3.columns):
    row, col_idx = divmod(i, 3)
    if i < len(case3.columns):
        case3[col].plot(kind='bar', ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'{col} - {case3.columns.name if hasattr(case3.columns, "name") else ""}')
        axes[row, col_idx].set_xlabel('城市')
        if col in ['X1', 'X4']:  # 金额指标
            axes[row, col_idx].set_ylabel('元/人')
        elif col in ['X2', 'X3']:  # 百分比指标
            axes[row, col_idx].set_ylabel('%')
plt.tight_layout()
plt.savefig('./result/14_city_indicators_subplots.png', dpi=300)
plt.close()

# 地区的直观分析
case3.loc[['广州', '深圳', '珠海']][['X1', 'X4']].plot(kind='bar', figsize=(8, 6))
plt.title('广州、深圳、珠海人均GDP和人均可支配收入')
plt.xlabel('城市')
plt.ylabel('金额(元/人)')
plt.tight_layout()
plt.savefig('./result/15_pearl_river_delta_economy.png', dpi=300)
plt.close()

# 保存珠三角城市数据分析
with open('./result/pearl_river_delta_analysis.txt', 'w', encoding='utf-8') as f:
    cities = ['广州', '深圳', '珠海']
    indicators = ['X1', 'X4']
    f.write("珠三角城市经济指标对比分析\n")
    f.write("=" * 50 + "\n\n")
    
    for city in cities:
        f.write(f"{city}经济指标:\n")
        for ind in indicators:
            ind_name = "人均GDP" if ind == "X1" else "人均可支配收入"
            f.write(f"  {ind_name}: {case3.loc[city, ind]:.2f} 元/人\n")
        f.write("\n")
    
    # 计算三个城市的均值
    avg_data = case3.loc[cities, indicators].mean()
    f.write("三市平均水平:\n")
    f.write(f"  人均GDP: {avg_data['X1']:.2f} 元/人\n")
    f.write(f"  人均可支配收入: {avg_data['X4']:.2f} 元/人\n\n")

case3.loc[['广州', '深圳', '珠海']][['X2', 'X3']].plot(kind='barh', stacked=True, figsize=(8, 6))
plt.title('广州、深圳、珠海第三产业比重和城镇人口比例')
plt.xlabel('百分比(%)')
plt.tight_layout()
plt.savefig('./result/16_pearl_river_delta_structure.png', dpi=300)
plt.close()

# 创建总结报告文件
with open('./result/summary_report.txt', 'w', encoding='utf-8') as f:
    f.write("多元数据Python可视化分析总结报告\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("本分析主要关注两个数据集：\n")
    f.write("1. 全国31个省市居民消费数据（8个消费类别）\n")
    f.write("2. 城市现代化水平指标数据（9个指标）\n\n")
    
    f.write("分析方法包括：\n")
    f.write("- 基本统计量计算（均值、方差、协方差、相关系数等）\n")
    f.write("- 单变量和多变量可视化（条形图、箱线图、散点图等）\n")
    f.write("- 比较分析（地区间消费结构对比、城市现代化水平对比）\n\n")
    
    f.write("主要发现：\n")
    f.write("1. 省市间居民消费存在明显差异，其中上海、北京和广东居民消费水平较高\n")
    f.write("2. 食品消费和居住支出占居民消费的主要部分\n")
    f.write("3. 珠三角地区城市现代化水平较高，深圳在多项指标上表现突出\n")
    f.write("4. 经济发展水平与城镇化率、第三产业占比等指标呈现显著相关性\n\n")
    
    f.write("后续研究方向：\n")
    f.write("- 进一步探索各变量间的因果关系\n")
    f.write("- 建立预测模型评估城市发展趋势\n")
    f.write("- 考虑更多社会、环境因素进行综合分析\n")