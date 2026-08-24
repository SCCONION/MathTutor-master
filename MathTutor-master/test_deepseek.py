import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role":"user",
            "content":"请解释二次函数顶点坐标公式"
        }
    ]
)


print(response.choices[0].message.content)