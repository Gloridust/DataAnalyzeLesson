# 导入 fetch_openml 函数用于获取数据集
from sklearn.datasets import fetch_openml

# 使用 fetch_openml 函数获取 "adult" 数据集的特征矩阵 X_adult 和标签向量 y_adult
X_adult, y_adult = fetch_openml("adult", version=2, return_X_y=True)

# 删除 X_adult 中的冗余特征列和非特征列，即 "education-num" 和 "fnlwgt" 列
X_adult = X_adult.drop(["education-num", "fnlwgt"], axis="columns")

# 打印 X_adult 的数据和类型
print(X_adult)
print(X_adult.dtypes)
