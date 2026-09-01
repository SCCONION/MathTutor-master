from backend.agents.utils.helper import _get_secret


# BGE Chinese embedding model
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"

EMBED_INPUT_TYPE_DOC   = "document"
EMBED_INPUT_TYPE_QUERY = "query"
EMBED_DIM              = 1024

# Minimum cosine similarity to include a chunk in results.
# 原为 0.30；经消融测试（scripts/rag_ablation.py 阈值扫描）验证，
# 0.30 挡不住"含关键词但语义无关"的诱饵文档（诱饵 cos≈0.48、真相关 cos≈0.50+）。
# 提高到 0.50 后噪声场景 Recall@5 从 86%→90%，诱饵混入率降 ~30%，且不误杀真相关。
MIN_SCORE = 0.50

# Number of chunks to retrieve per query (final after fusion)
TOP_K = 5


__all__ = [
    "_get_secret",
    "EMBED_MODEL",
    "EMBED_INPUT_TYPE_DOC",
    "EMBED_INPUT_TYPE_QUERY",
    "EMBED_DIM",
    "MIN_SCORE",
    "TOP_K",
]
