# test.py

from openai import OpenAI
from config import OPENAI_API_BASE, OPENAI_API_KEY

# 初始化客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "什么是人工智能？"}
        ]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")