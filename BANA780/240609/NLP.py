import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

# 分词
tokenized_reviews = df['Cleaned_Review'].apply(word_tokenize)

# 训练Word2Vec模型
model = Word2Vec(tokenized_reviews, vector_size=100, window=5, min_count=2, workers=4)

# 打印词汇表中的前10个词
print(list(model.wv.index_to_key)[:10])

# 选择一个存在的词进行相似词查找
existing_word = list(model.wv.index_to_key)[0]  # 选择词汇表中的第一个词
similar_words = model.wv.most_similar(existing_word, topn=10)
print(f"Similar words to '{existing_word}':", similar_words)

# 生成词云
all_words = ' '.join(df['Cleaned_Review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

# 显示词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
