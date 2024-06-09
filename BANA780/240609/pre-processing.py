import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# 下载nltk数据包
nltk.download('punkt')
nltk.download('stopwords')

# 读取数据
df = pd.read_csv('amazon_reviews.csv')

# 定义数据清洗函数
def preprocess_text(text):
    # 转小写
    text = text.lower()
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 分词
    words = word_tokenize(text)
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# 应用数据清洗函数
df['Cleaned_Review'] = df['Review'].apply(preprocess_text)
