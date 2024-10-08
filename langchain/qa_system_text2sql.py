from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
from config import OPENAI_API_BASE, OPENAI_API_KEY

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

# 创建一个示例 SQLite 数据库
engine = create_engine('sqlite:///example.db')
metadata = MetaData()

# 定义一个示例表
products = Table('products', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String),
    Column('category', String),
    Column('price', Float)
)

# 创建表并插入一些示例数据
metadata.create_all(engine)
with engine.connect() as conn:
    conn.execute(products.insert(), [
        {"name": "Laptop", "category": "Electronics", "price": 1000},
        {"name": "Smartphone", "category": "Electronics", "price": 500},
        {"name": "Desk Chair", "category": "Furniture", "price": 200},
        {"name": "Coffee Table", "category": "Furniture", "price": 150}
    ])

# 初始化数据库和语言模型
db = SQLDatabase.from_engine(engine)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_base=OPENAI_API_BASE)

# 创建 SQL 数据库链
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# 使用自然语言查询数据库
query = "What is the average price of Electronics products?"
result = db_chain.run(query)
print(result)