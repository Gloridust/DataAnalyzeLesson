# qa_system.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import OPENAI_API_BASE, OPENAI_API_KEY

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

# 初始化语言模型
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_base=OPENAI_API_BASE
)

# 创建一个提示模板
prompt = ChatPromptTemplate.from_template("Q: {question}\nA: ")

# 创建一个简单的链
chain = prompt | llm | StrOutputParser()

# 使用链来回答问题
question = "什么是人工智能？"
try:
    response = chain.invoke({"question": question})
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
    # 如果发生错误，打印更多调试信息
    print(f"OpenAI API Base: {OPENAI_API_BASE}")
    print(f"OpenAI API Key (first 5 chars): {OPENAI_API_KEY[:5]}...")