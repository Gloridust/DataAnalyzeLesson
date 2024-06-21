import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import category_encoders as ce
from imblearn.over_sampling import SMOTE
import numpy as np

# 加载数据
train_transaction = pd.read_csv('./data/train_transaction.csv')
train_identity = pd.read_csv('./data/train_identity.csv')
test_transaction = pd.read_csv('./data/test_transaction.csv')
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

# 填充缺失值并将对象类型特征转换为字符串
train_data.fillna('-999', inplace=True)
test_data.fillna('-999', inplace=True)

# 编码对象类型特征
encoder = ce.OrdinalEncoder()
train_data = encoder.fit_transform(train_data)
test_data = encoder.transform(test_data)

# 确保所有特征都是数值型
train_data = train_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# 特征和标签
X = train_data.drop(['isFraud', 'TransactionID'], axis=1)
y = train_data['isFraud']

# 检查数据分布
print("Class distribution in y before SMOTE:")
print(y.value_counts())

# 使用SMOTE进行数据增强
if y.nunique() > 1:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
else:
    print("训练数据集中没有欺诈交易样本，使用SMOTE进行数据增强。")
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

# 检查增强后的数据分布
print("Class distribution in y after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# 确保分割时每个类别都有样本
X_train, X_valid, y_train, y_valid = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# 检查分割后数据集的类别分布
print("Class distribution in y_train:")
print(pd.Series(y_train).value_counts())
print("Class distribution in y_valid:")
print(pd.Series(y_valid).value_counts())

# 模型训练
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 交叉验证模型性能
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='roc_auc')
print("Cross-validated AUC scores:", cv_scores)
print("Mean AUC score:", np.mean(cv_scores))

# 模型评估
y_pred = model.predict(X_valid)
print(classification_report(y_valid, y_pred))
try:
    roc_auc = roc_auc_score(y_valid, y_pred)
    print("ROC-AUC Score:", roc_auc)
except ValueError as e:
    print(e)

# 在预测之前删除 'isFraud' 列
X_test = test_data.drop(['TransactionID', 'isFraud'], axis=1, errors='ignore')
test_data['isFraud'] = model.predict(X_test)

# 检查预测结果的分布
print("Predicted class distribution in test data:")
print(test_data['isFraud'].value_counts())

# 生成提交文件
submission = test_data[['TransactionID', 'isFraud']]
submission.to_csv('./submission.csv', index=False)
