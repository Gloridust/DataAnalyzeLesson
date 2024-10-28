import pandas as pd
import codecs
import csv

with codecs.open('./process_csv_1_1.csv',encoding='utf-8-sig') as f:
    for row in csv.DictReader(f, skipinitialspace=True):
        print(row)