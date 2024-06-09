import pandas as pd
import re

# 读取数据
claims_data = pd.read_csv('Claims.csv')
members_data = pd.read_csv('Members.csv')
days_in_hospital_data = pd.read_csv('DaysInHospital_Y2.csv')
drug_count_data = pd.read_csv('DrugCount.csv')
lab_count_data = pd.read_csv('LabCount.csv')

# 处理 Claims 数据集的缺失值和数据类型
claims_data['ProviderID'] = claims_data['ProviderID'].ffill()
claims_data['Vendor'] = claims_data['Vendor'].ffill()
claims_data['PCP'] = claims_data['PCP'].ffill()
claims_data['Specialty'] = claims_data['Specialty'].ffill()
claims_data['PlaceSvc'] = claims_data['PlaceSvc'].ffill()
claims_data['DSFS'] = claims_data['DSFS'].ffill()
claims_data['PrimaryConditionGroup'] = claims_data['PrimaryConditionGroup'].ffill()
claims_data['ProcedureGroup'] = claims_data['ProcedureGroup'].ffill()
claims_data['LengthOfStay'] = claims_data['LengthOfStay'].fillna('0')

# 处理 PayDelay 列
claims_data['PayDelay'] = pd.to_numeric(claims_data['PayDelay'], errors='coerce').fillna(0).astype(int)

# 处理 LengthOfStay 列
def convert_length_of_stay(value):
    if 'day' in value:
        return 1
    elif 'week' in value:
        return int(re.search(r'\d+', value).group()) * 7
    elif 'month' in value:
        return int(re.search(r'\d+', value).group()) * 30
    elif '+' in value:
        return int(value.replace('+', '').strip())
    else:
        return int(value)

claims_data['LengthOfStay'] = claims_data['LengthOfStay'].apply(convert_length_of_stay)

# 处理 CharlsonIndex 列
def convert_charlson_index(value):
    if '-' in value:
        return (int(value.split('-')[0]) + int(value.split('-')[1])) / 2
    elif '+' in value:
        return int(value.replace('+', '').strip())
    else:
        return float(value)

claims_data['CharlsonIndex'] = claims_data['CharlsonIndex'].apply(convert_charlson_index)

# 处理 Members 数据集的缺失值和数据类型
members_data['AgeAtFirstClaim'] = members_data['AgeAtFirstClaim'].ffill()
members_data['Sex'] = members_data['Sex'].fillna('Unknown')

# 处理 DrugCount 数据集的缺失值和数据类型
drug_count_data['DrugCount'] = drug_count_data['DrugCount'].replace({'7+': 7}).astype(int)

# 处理 LabCount 数据集的缺失值和数据类型
def convert_lab_count(value):
    if '+' in value:
        return int(value.replace('+', '').strip())
    else:
        return int(value)

lab_count_data['LabCount'] = lab_count_data['LabCount'].apply(convert_lab_count)

# 删除不重要的变量
claims_data = claims_data.drop(columns=['Vendor', 'PCP'])

# 创建新特征
claims_data['TotalVisits'] = claims_data['LengthOfStay'] + claims_data['SupLOS']

# 检查处理后的数据
print("Claims Data Info:")
print(claims_data.info())
print("Members Data Info:")
print(members_data.info())
print("Drug Count Data Info:")
print(drug_count_data.info())
print("Lab Count Data Info:")
print(lab_count_data.info())

# 显示前几行数据以验证处理结果
print("Claims Data Head:")
print(claims_data.head())
print("Members Data Head:")
print(members_data.head())
print("Drug Count Data Head:")
print(drug_count_data.head())
print("Lab Count Data Head:")
print(lab_count_data.head())

# 保存处理后的数据
claims_data.to_csv('Processed_Claims.csv', index=False)
members_data.to_csv('Processed_Members.csv', index=False)
drug_count_data.to_csv('Processed_DrugCount.csv', index=False)
lab_count_data.to_csv('Processed_LabCount.csv', index=False)