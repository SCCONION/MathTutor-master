import numpy as np
import faiss

from langchain_community.embeddings import HuggingFaceBgeEmbeddings


print("=" * 50)
print("测试 BGE + FAISS")
print("=" * 50)


# 1. 加载 BGE GPU模型
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={
        "device": "cuda"
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32
    }
)


print("模型加载成功")


# 2. 模拟知识库文本
documents = [
    "二次函数的一般形式是 y=ax²+bx+c。",
    "二次函数顶点坐标公式为 (-b/2a, f(-b/2a))。",
    "牛顿第二定律公式是 F=ma。",
    "勾股定理描述直角三角形三边关系。"
]


# 3. 文档向量化
vectors = embedding_model.embed_documents(
    documents
)


vectors = np.array(
    vectors,
    dtype="float32"
)


print("Embedding维度:")
print(vectors.shape)


# 4. 创建FAISS索引
dimension = vectors.shape[1]

index = faiss.IndexFlatIP(
    dimension
)


index.add(vectors)


print("FAISS建立成功")
print("向量数量:", index.ntotal)


# 5. 查询
query = "如何求二次函数顶点？"


query_vector = embedding_model.embed_query(
    query
)


query_vector = np.array(
    [query_vector],
    dtype="float32"
)


# 6. 搜索Top-K
top_k = 2

scores, indices = index.search(
    query_vector,
    top_k
)


print("\n查询:")
print(query)


print("\n结果:")

for score, idx in zip(scores[0], indices[0]):
    print("----------------")
    print("相似度:", score)
    print("内容:", documents[idx])


print("\n测试完成")