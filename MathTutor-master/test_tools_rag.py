import os
import sys


# =========================
# 加入 src 路径
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    os.path.join(BASE_DIR, "src")
)


from backend.agents.nodes.tools.tools import (
    ingest_pdf,
    rag_tool,
    get_store_info,
    has_store,
    clear_store
)


print("=" * 60)
print("MathTutor RAG 测试")
print("=" * 60)


# =========================
# 配置
# =========================

thread_id = "test_user_001"


pdf_path = os.path.join(
    BASE_DIR,
    "test_data",
    "math_test.pdf"
)


print("\nPDF路径:")
print(pdf_path)


if not os.path.exists(pdf_path):
    print("❌ 找不到PDF")
    print("请创建:")
    print(pdf_path)
    exit()


# =========================
# 1. 清理旧索引
# =========================

if has_store(thread_id):

    print("\n发现旧知识库，清理")

    clear_store(thread_id)


# =========================
# 2. 建库
# =========================

print("\n开始 ingest_pdf...")

# 读取 PDF 文件的二进制数据
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

result = ingest_pdf(
    pdf_bytes,   # 传入二进制 bytes
    thread_id
)


print("\n建库结果:")
print(result)



# =========================
# 3. 查看store
# =========================

print("\nStore信息:")

info = get_store_info(
    thread_id
)

print(info)



# =========================
# 4. RAG查询
# =========================

query = "二次函数顶点坐标公式是什么？"


print("\n查询:")
print(query)


answer = rag_tool.invoke(
    {
        "query": query,
        "thread_id": thread_id
    }
)


print("\n========== RAG结果 ==========")

print(answer)


print("\n测试结束")