import pandas as pd
import codecs
import csv

latest_date = None
latest_rows = []

with codecs.open('./process_csv_1_1.csv',encoding='utf-8-sig') as f:
    for row in csv.DictReader(f, skipinitialspace=True):
        # print(row)
        
        # 比较每一行的日期，找到日期最晚的行
        if latest_date is None or row['日期'] > latest_date:
            latest_date = row['日期']
            latest_rows = [row]
        elif row['日期'] == latest_date:
            latest_rows.append(row)

# 打印日期最晚的所有行
print("\n日期最晚的所有行:")
for latest_row in latest_rows:
    print(latest_row)
