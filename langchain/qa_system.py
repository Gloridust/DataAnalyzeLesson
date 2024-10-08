# qa_system.py

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import OPENAI_API_BASE, OPENAI_API_KEY

# 设置OpenAI API配置
import os
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 初始化语言模型,使用自定义base URL
llm = OpenAI(temperature=0.7, openai_api_base=OPENAI_API_BASE)

# 创建一个提示模板
prompt = PromptTemplate(
    input_variables=["question"],
    template="Q: {question}\nA: ",
)

# 创建一个LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# 使用链来回答问题
question = "什么是人工智能？"
response = chain.run(question)

print(response)