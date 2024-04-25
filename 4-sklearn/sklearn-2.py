# 导入 train_test_split 函数用于划分数据集
from sklearn.model_selection import train_test_split
# 导入 datasets 模块
from sklearn import datasets
# 导入 KNeighborsClassifier 用于K近邻分类
from sklearn.neighbors import KNeighborsClassifier  

# 加载 iris 数据集
iris = datasets.load_iris()
# 提取 iris 数据集的特征矩阵 iris_X 和标签向量 iris_y
iris_X = iris.data
iris_y = iris.target

# 将 iris 数据集划分为训练集和测试集，其中测试集占总数据集的30%
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

# 打印训练集的标签
print("训练集的标签:",y_train)

# 创建 K 近邻分类器对象
knn = KNeighborsClassifier()

# 使用训练集数据对 K 近邻分类器进行训练
knn.fit(X_train, y_train)

# 使用训练好的 K 近邻分类器对测试集数据进行预测，并打印预测结果
print("预测结果:",knn.predict(X_test))

# 打印测试集的真实标签
print("真实标签:",y_test)


# 1. fetch_openml 函数用于从 OpenML 数据集库中获取数据集。
# 2. 该代码段首先获取了一个名为 "adult" 的数据集，并将其特征矩阵存储在 X_adult 中，标签向量存储在 y_adult 中。
# 3. 通过删除 X_adult 中的 "education-num" 和 "fnlwgt" 列，去除了冗余和非特征信息。
# 4. 使用 train_test_split 函数将 iris 数据集划分为训练集（X_train 和 y_train）和测试集（X_test 和 y_test），其中测试集占总数据集的30%。
# 5. 创建了一个 K 近邻分类器对象 knn，并使用训练集数据对其进行训练。
# 6. 对测试集数据进行预测，并将预测结果打印出来。
# 7. 打印测试集的真实标签，用于与预测结果进行对比。
