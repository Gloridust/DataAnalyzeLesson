from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from config import OPENAI_API_BASE, OPENAI_API_KEY

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

# 初始化语言模型
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_base=OPENAI_API_BASE)

# 创建一个示例文档集
documents = [
    "人工智能是计算机科学的一个分支，致力于创造智能机器。",
    "机器学习是人工智能的一个子集，专注于让系统从数据中学习。",
    "深度学习是机器学习的一种特定方法，使用神经网络进行学习。"
]

# 创建向量存储
embeddings = OpenAIEmbeddings(openai_api_base=OPENAI_API_BASE)
vectorstore = FAISS.from_texts(documents, embeddings)

# 创建检索器
retriever = vectorstore.as_retriever()

# 创建提示模板
template = """使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。

上下文: {context}

问题: {question}

回答:"""
prompt = ChatPromptTemplate.from_template(template)

# 创建RAG链
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 使用RAG链回答问题
question = "什么是深度学习？"
response = rag_chain.invoke(question)
print(response.content)