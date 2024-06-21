import numpy as np
import matplotlib.pyplot as plt

# 这里我们使用numpy中的linspace函数生成一个数组X。linspace的参数说明：
# -np.pi：数组的起始值（-π）。
# np.pi：数组的结束值（π）。
# 256：数组中的元素个数（在-π到π之间均匀分布256个点）。
# endpoint=True：包括结束值π。
# 这样，我们得到了一个包含256个点的数组X，这些点均匀分布在-π到π之间。
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)

# 我们使用numpy的cos和sin函数分别计算数组X中每个点的余弦和正弦值，并将结果存储在数组C和S中。
C,S = np.cos(X), np.sin(X)

# 这里我们使用matplotlib.pyplot中的plot函数绘制两条曲线：
# plt.plot(X, C)：以X为横坐标，C（余弦值）为纵坐标绘制曲线。
# plt.plot(X, S)：以X为横坐标，S（正弦值）为纵坐标绘制曲线。
plt.plot(X,C,label='cos')   # label:标签
plt.plot(X,S,label='sin')

# 添加标题
plt.title('Title: Cosine and Sine Functions')

# 添加轴标签
plt.xlabel('X values')
plt.ylabel('Y values')

# 添加图例 label:标签
plt.legend()

# 显示绘制的图像
plt.show()