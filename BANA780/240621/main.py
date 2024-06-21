import pandas as pd

# 加载数据
train_transaction = pd.read_csv('./data/train_transaction_example.csv')
train_identity = pd.read_csv('./data/train_identity.csv')
test_transaction = pd.read_csv('./data/test_transaction_example.csv')
test_identity = pd.read_csv('./data/test_identity.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')

# 查看数据的基本信息
print("Train Transaction Data Info:")
print(train_transaction.info())
print("\nTrain Identity Data Info:")
print(train_identity.info())
print("\nTest Transaction Data Info:")
print(test_transaction.info())
print("\nTest Identity Data Info:")
print(test_identity.info())

# 合并训练集数据
train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

# 合并测试集数据
test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

print("Merged Train Data Info:")
print(train_data.info())
print("\nMerged Test Data Info:")
print(test_data.info())

# 检查缺失值
missing_train = train_data.isnull().sum()
missing_test = test_data.isnull().sum()

# 显示缺失值情况
print("Missing values in Train Data:")
print(missing_train[missing_train > 0])
print("\nMissing values in Test Data:")
print(missing_test[missing_test > 0])

# 示例：填充缺失值
train_data.fillna(-999, inplace=True)
test_data.fillna(-999, inplace=True)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import category_encoders as ce

# 加载数据
train_transaction = pd.read_csv('./data/train_transaction_example.csv')
train_identity = pd.read_csv('./data/train_identity.csv')
test_transaction = pd.read_csv('./data/test_transaction_example.csv')
test_identity = pd.read_csv('./data/test_identity.csv')

# 合并数据
train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

# 对齐训练集和测试集的列
train_columns = train_data.columns
test_columns = test_data.columns

for col in train_columns:
    if col not in test_columns:
        test_data[col] = -999

for col in test_columns:
    if col not in train_columns:
        train_data[col] = -999

train_data = train_data[sorted(train_data.columns)]
test_data = test_data[sorted(test_data.columns)]

train_data.fillna(-999, inplace=True)
test_data.fillna(-999, inplace=True)

encoder = ce.OrdinalEncoder()
train_data = encoder.fit_transform(train_data)
test_data = encoder.transform(test_data)

X = train_data.drop(['isFraud', 'TransactionID'], axis=1)
y = train_data['isFraud']

# 检查数据分布
print("Class distribution in y:")
print(y.value_counts())

# 确保分割时每个类别都有样本
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 检查分割后数据集的类别分布
print("Class distribution in y_train:")
print(y_train.value_counts())
print("Class distribution in y_valid:")
print(y_valid.value_counts())

# 模型训练
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_valid)
print(classification_report(y_valid, y_pred))
try:
    roc_auc = roc_auc_score(y_valid, y_pred)
    print("ROC-AUC Score:", roc_auc)
except ValueError as e:
    print(e)

# 预测测试集
X_test = test_data.drop(['TransactionID'], axis=1)
test_data['isFraud'] = model.predict(X_test)

# 生成提交文件
submission = test_data[['TransactionID', 'isFraud']]
submission.to_csv('./data/submission.csv', index=False)
