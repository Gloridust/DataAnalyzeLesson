from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from config import OPENAI_API_BASE, OPENAI_API_KEY

import os
import re

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

# 初始化语言模型
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_base=OPENAI_API_BASE)

# 连接到 SQLite 数据库（这里使用内存数据库作为示例）
db = SQLDatabase.from_uri("sqlite:///:memory:")

# 创建示例表和数据
db.run("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER)")
db.run("INSERT INTO employees (name, department, salary) VALUES ('Alice', 'Engineering', 80000)")
db.run("INSERT INTO employees (name, department, salary) VALUES ('Bob', 'Sales', 70000)")
db.run("INSERT INTO employees (name, department, salary) VALUES ('Charlie', 'Marketing', 75000)")

# 创建 SQL 查询链
sql_chain = create_sql_query_chain(llm, db)

# 使用自然语言生成 SQL 查询
user_query = "列出所有员工的姓名和他们的部门"
sql_query = sql_chain.invoke({"question": user_query})

# 提取实际的 SQL 查询
def extract_sql(text):
    # 尝试找到 SQL 查询部分
    match = re.search(r'SELECT.*?;', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    else:
        # 如果没有找到明确的 SELECT 语句，返回整个文本
        return text

sql_query = extract_sql(sql_query)
print(f"生成的 SQL 查询: {sql_query}")

# 执行生成的 SQL 查询
try:
    result = db.run(sql_query)
    print(f"查询结果: {result}")
except Exception as e:
    print(f"执行查询时出错: {e}")
    print("生成的查询可能需要手动调整。")

# 创建一个更复杂的自然语言到 SQL 的链
template = """基于以下请求创建一个 SQL 查询：

{question}

仅返回 SQL 查询，不要包含任何其他文本。
"""
prompt = ChatPromptTemplate.from_template(template)

sql_response_chain = prompt | llm

# 使用更复杂的自然语言查询
complex_query = "谁是薪水最高的员工，他的薪水是多少？"
sql_response = sql_response_chain.invoke({"question": complex_query})
sql_query = extract_sql(sql_response.content)
print(f"生成的 SQL 查询: {sql_query}")

# 执行生成的 SQL 查询
try:
    result = db.run(sql_query)
    print(f"查询结果: {result}")
except Exception as e:
    print(f"执行查询时出错: {e}")
    print("生成的查询可能需要手动调整。")