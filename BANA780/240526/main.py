import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 从seaborn库加载Titanic数据集
titanic = sns.load_dataset('titanic')

# 显示数据集的前几行
titanic.head()

# 检查缺失值
titanic.isnull().sum()

# 用中位数填充缺失的年龄值
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# 用众数填充缺失的登船港口值
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# 删除包含缺失值的'甲板'和'登船城镇'列
titanic.drop(columns=['deck', 'embark_town'], inplace=True)

# 再次检查缺失值
titanic.isnull().sum()

# 绘制每个变量的常见值图
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 生存状态
sns.countplot(data=titanic, x='survived', ax=axs[0, 0])
axs[0, 0].set_title('生存状态')

# 乘客等级
sns.countplot(data=titanic, x='pclass', ax=axs[0, 1])
axs[0, 1].set_title('乘客等级')

# 性别
sns.countplot(data=titanic, x='sex', ax=axs[1, 0])
axs[1, 0].set_title('性别')

# 登船地点
sns.countplot(data=titanic, x='embarked', ax=axs[1, 1])
axs[1, 1].set_title('登船地点')

plt.tight_layout()
plt.show()

# 绘制不同等级乘客的生存情况
plt.figure(figsize=(10, 6))
sns.countplot(data=titanic, x='pclass', hue='survived')
plt.title('不同等级乘客的生存情况')
plt.show()

# 绘制不同性别乘客的生存情况
plt.figure(figsize=(10, 6))
sns.countplot(data=titanic, x='sex', hue='survived')
plt.title('不同性别乘客的生存情况')
plt.show()

# 绘制不同年龄乘客的生存情况
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='age', hue='survived', multiple='stack', kde=False)
plt.title('不同年龄乘客的生存情况')
plt.show()

# 创建年龄组
titanic['age_group'] = pd.cut(titanic['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])

# 绘制不同年龄组和等级中的生存情况
plt.figure(figsize=(12, 8))
sns.countplot(data=titanic, x='age_group', hue='survived', palette='pastel')
plt.title('不同年龄组和等级中的生存情况')
plt.show()

# 绘制不同登船地点和等级中的生存情况
plt.figure(figsize=(12, 8))
sns.countplot(data=titanic, x='embarked', hue='survived', palette='pastel')
plt.title('不同登船地点和等级中的生存情况')
plt.show()
