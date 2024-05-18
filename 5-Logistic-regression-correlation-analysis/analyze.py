# 贷款利率与客户流失率关系分析
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

data_path = 'data3.xlsx'
data = pd.read_excel(data_path)

# 定义预测器和目标变量
X = data[['贷款年利率']]
y_A = data['信誉评级A客户流失率']
y_B = data['信誉评级B客户流失率']
y_C = data['信誉评级C客户流失率']

# 构建模型
models = {}
predictions = {}
scores = {}

# 分别为每个目标变量训练模型
for target, y in zip(['A', 'B', 'C'], [y_A, y_B, y_C]):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建线性回归模型
    model = LinearRegression()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 保存模型和预测结果（可选）
    models[target] = model
    predictions[target] = y_pred
    
    # 计算并保存性能指标
    scores[target] = {
        '均方根误差 (RMSE)': root_mean_squared_error(y_test, y_pred),
        '决定系数 (R²)': r2_score(y_test, y_pred)
    }

# 输出每个模型的评分
for target, score in scores.items():
    print(f"信誉评级 {target} 客户的模型:")
    print(f"  均方根误差 (RMSE): {score['均方根误差 (RMSE)']}")
    print(f"  决定系数 (R²): {score['决定系数 (R²)']}\n")

#####可视化#####

# 导入可视化所需的库
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set(style="whitegrid")

# 创建一个图形框架，三个子图对应三个信誉评级
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# 信誉评级列表
credit_ratings = ['A', 'B', 'C']
data_columns = ['信誉评级A客户流失率', '信誉评级B客户流失率', '信誉评级C客户流失率']

# 循环绘制每个信誉评级的散点图和回归线
for ax, target, y, column in zip(axes, credit_ratings, [y_A, y_B, y_C], data_columns):
    # 散点图
    sns.scatterplot(x=X['贷款年利率'], y=y, ax=ax, color='blue', label='实际数据')
    
    # 预测回归线
    # 需要首先生成贷款年利率的一系列值，以便在这个范围内进行预测
    line_X = pd.DataFrame({'贷款年利率': [x / 1000.0 for x in range(int(min(X['贷款年利率']*1000)), int(max(X['贷款年利率']*1000)))]})
    line_y = models[target].predict(line_X)
    sns.lineplot(x=line_X['贷款年利率'], y=line_y, ax=ax, color='red', label='回归线')
    
    # 设置图表标题和标签
    ax.set_title(f'信誉评级 {target} 客户流失率')
    ax.set_xlabel('贷款年利率')
    ax.set_ylabel(column)

# 调整图例位置
axes[0].legend(loc='upper left')
axes[1].legend(loc='upper left')
axes[2].legend(loc='upper left')

# 显示图形
plt.tight_layout()
plt.show()
