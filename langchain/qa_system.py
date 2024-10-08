from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import OPENAI_API_BASE, OPENAI_API_KEY

import os
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 初始化语言模型
llm = OpenAI(temperature=0.7, openai_api_base=OPENAI_API_BASE)

# 创建一个提示模板
prompt = PromptTemplate.from_template("Q: {question}\nA: ")

# 创建一个可运行的序列
chain = (
    {"question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

# 使用链来回答问题
question = "什么是人工智能？"
try:
    response = chain.invoke(question)
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
    # 添加更多的错误信息打印
    import traceback
    print(traceback.format_exc())
    
    # 尝试直接调用 OpenAI API 以获取更多信息
    try:
        direct_response = llm.invoke(prompt.format(question=question))
        print("Direct LLM response:", direct_response)
    except Exception as e2:
        print(f"Direct API call also failed: {e2}")