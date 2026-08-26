from __future__ import annotations

import re
import tempfile
from backend.agents import Any, Dict, List, Optional, os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import faiss
import numpy as np
import sympy as sp
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from backend.agents import logger
from backend.agents.nodes.tools import (
    _get_secret,
    EMBED_INPUT_TYPE_DOC,
    EMBED_INPUT_TYPE_QUERY,
    EMBED_DIM,
    MIN_SCORE,
    TOP_K,
)
from backend.agents.nodes.tools.mcp.tavily_mcp_client import tavily_mcp_search

embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"
)

# ══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY VECTOR STORE  (one index per thread_id)
# ══════════════════════════════════════════════════════════════════════════════

_STORES: Dict[str, Dict[str, Any]] = {}

# ── BGE Chinese Embedding ─────────────────────────────

_embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={
        "device": "cuda"
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 24
    }
)


# ── Embedding helper ──────────────────────────────────────────────────────────

def _embed_texts(
    texts: List[str],
    input_type: str
) -> np.ndarray:
    """
    Embed texts using BGE-large-zh-v1.5.

    Returns:
        L2-normalized float32 vectors
    """

    if not texts:
        return np.empty(
            (0, EMBED_DIM),
            dtype=np.float32
        )


    vecs = _embedding_model.embed_documents(
        texts
    )


    vecs = np.array(
        vecs,
        dtype=np.float32
    )


    norms = np.linalg.norm(
        vecs,
        axis=1,
        keepdims=True
    )

    norms = np.where(
        norms == 0,
        1.0,
        norms
    )


    return vecs / norms

def _tokenize(text: str) -> List[str]:
    return re.findall(r"(?u)\b\w+\b", text.lower())


# ══════════════════════════════════════════════════════════════════════════════
#  PDF INGESTION  (called by app.py on upload)
# ══════════════════════════════════════════════════════════════════════════════

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Embed and index a PDF for the given thread.
    Calling this a second time APPENDS to the existing index — all uploaded
    PDFs are searched together.

    Returns: {filename, pages, chunks, total_chunks}
    """
    if not file_bytes:
        raise ValueError("Empty file bytes — nothing to ingest.")

    fname = filename or "document.pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not docs:
        raise ValueError("PDF produced no pages — is it a scanned/image PDF?")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks   = splitter.split_documents(docs)
    texts    = [c.page_content for c in chunks]
    metadata = [c.metadata for c in chunks]

    if not texts:
        raise ValueError("No text chunks produced from PDF.")

    tokenized_chunks = [_tokenize(t) for t in texts]

    logger.info(
        f"[INGEST] Embedding {len(texts)} chunks | file={fname} | thread={thread_id}"
    )
    vecs = _embed_texts(texts, EMBED_INPUT_TYPE_DOC)

    existing = _STORES.get(thread_id)
    if existing:
        existing["index"].add(vecs)
        existing["doc_vecs"] = (
            np.vstack((existing["doc_vecs"], vecs))
            if len(existing.get("doc_vecs", [])) > 0
            else vecs
        )
        existing["chunks"].extend(texts)
        existing["metadata"].extend(metadata)
        existing["tokenized_chunks"].extend(tokenized_chunks)
        existing["bm25"] = BM25Okapi(existing["tokenized_chunks"])
        if fname not in existing["filenames"]:
            existing["filenames"].append(fname)
        total = len(existing["chunks"])
        logger.info(
            f"[INGEST] Appended | total_chunks={total} | thread={thread_id}"
        )
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(vecs)
        _STORES[thread_id] = {   # passing the sparse and dense vectors 
            "index":            index,
            "chunks":           texts,
            "metadata":         metadata,
            "filenames":        [fname],
            "bm25":             BM25Okapi(tokenized_chunks),
            "tokenized_chunks": tokenized_chunks,
            "doc_vecs":         vecs,
        }
        total = len(texts)
        logger.info(
            f"[INGEST] New index created | chunks={total} | thread={thread_id}"
        )

    return {
        "filename":     fname,
        "pages":        len(docs),
        "chunks":       len(texts),
        "total_chunks": total,
    }


def get_store_info(thread_id: str) -> Optional[dict]:
    store = _STORES.get(thread_id)
    if not store:
        return None
    return {
        "filenames": store["filenames"],
        "filename":  ", ".join(store["filenames"]),
        "chunks":    len(store["chunks"]),
    }


def has_store(thread_id: str) -> bool:
    return thread_id in _STORES


def clear_store(thread_id: str) -> None:
    _STORES.pop(thread_id, None)
    logger.info(f"[CRAG] Store cleared | thread={thread_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 1 — HYBRID CRAG
# ══════════════════════════════════════════════════════════════════════════════

@tool
def rag_tool(query: str, thread_id: str) -> str:
    """
    混合式 CRAG（纠错式检索增强生成）。

    流程：BM25 稀疏检索 + BGE 稠密检索 → 倒数排名融合（RRF）
              → 纠错式相关性过滤（余弦相似度 ≥ 0.30）

    ── 调用规则 ────────────────────────────────────────────────────────────
    • 当会话上传了文档时，一律先调用此工具 —
      即使你认为题目很简单也要先调用。
      学生上传笔记一定有其原因。

    • 使用聚焦的查询词，不要用完整题目文本。
      好：  "分部积分公式"
      差：  "用你会的任何方法求 x²sin(x)dx 的积分"

    • 如果返回"未找到相关段落"，不要重试。
      改用 web_search_tool 或你自己的知识。

    • 如果没有索引任何文档，本工具会返回一条明确的跳过消息。
      收到该消息后不要再调用 rag_tool。

    参数：
        query     : 聚焦的检索查询词。
        thread_id : 会话线程 ID（由图自动注入）。

    返回：
        带页码和分值的排序段落，或明确的"无相关内容"消息。
    """
    # ── Guard: no store ───────────────────────────────────────────────────────
    if not thread_id or not has_store(thread_id):
        logger.info(f"[CRAG] No store for thread={thread_id} — skipping")
        return (
            "CRAG: 当前会话没有索引任何文档。"
            "不要再调用 rag_tool — 改用 web_search_tool 或你自己的知识。"
        )

    store = _STORES[thread_id]
    logger.info(
        f"[CRAG] query='{query[:80]}' | thread={thread_id} | "
        f"index_size={store['index'].ntotal}"
    )

    # ── Dense retrieval (BGE) ──────────────────────────────────────────────
    q_vec = _embed_texts([query], EMBED_INPUT_TYPE_QUERY)
    _, indices = store["index"].search(q_vec, 10)
    dense_idx  = indices[0]

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────
    tokens        = _tokenize(query)
    sparse_scores = store["bm25"].get_scores(tokens)
    sparse_idx    = np.argsort(sparse_scores)[::-1][:10]

    # ── Reciprocal Rank Fusion (LangChain EnsembleRetriever algorithm) ────────
    K          = 60
    rrf: Dict[int, float] = {}
    for rank, idx in enumerate(dense_idx):    ## just using the same yet simple form of RRF as one provided by langchain 
        if 0 <= idx < len(store["chunks"]):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K + rank + 1)
    for rank, idx in enumerate(sparse_idx):
        if 0 <= idx < len(store["chunks"]):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K + rank + 1)

    fused = sorted(rrf, key=rrf.get, reverse=True)[:TOP_K]

    # ── Corrective filter (the "C" in CRAG) ───────────────────────────────────
    results: List[str] = []
    for idx in fused:
        if not (0 <= idx < len(store["chunks"])):
            continue
        cos_sim = float(np.dot(q_vec[0], store["doc_vecs"][idx]))
        if cos_sim < MIN_SCORE:
            logger.debug(f"[CRAG] Dropped idx={idx} cos={cos_sim:.3f} < {MIN_SCORE}")
            continue
        page    = store["metadata"][idx].get("page", "?")
        passage = store["chunks"][idx].strip()
        results.append(f"[Page {page} | relevance={cos_sim:.3f}]\n{passage}")

    filenames = ", ".join(store["filenames"])

    # ── No relevant content — graceful skip, NOT an error ─────────────────────
    if not results:
        msg = (
            f"CRAG: 在 '{filenames}' 中没有找到与查询 '{query}' 相关的段落。"
            "文档似乎没有涵盖这个主题。"
            "改用 web_search_tool 或自己的知识继续。"
        )
        logger.info(f"[CRAG] {msg}")
        return msg

    logger.info(f"[CRAG] Returning {len(results)} passages from '{filenames}'")
    return (
        f"混合式 CRAG — 从 '{filenames}' 中找到 {len(results)} 段相关内容：\n\n"
        + "\n\n---\n\n".join(results)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 2 — WEB SEARCH  (Tavily MCP — general)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def web_search_tool(query: str) -> str:
    """
    实时联网搜索（通过 Tavily MCP 服务器，远程服务，无需本地配置）。

    返回 Tavily AI 直接回答 + 前 5 条带摘要的排序结果。

    ── 何时调用 ──────────────────────────────────────────────────────────────
    • 学生询问数学领域的最新发现或研究突破
    • 学生要求某个主题的新题目
    • 学生询问学习资源、教材或视频讲解
    • 学生询问竞赛题目（IMO、Putnam 等）
    • CRAG 返回为空或上下文不足
    • 任何需要当前或最新信息的事实性问题

    ── 多查询策略（同一轮最多调用 3 次）──────────────────────────────────
      查询 1 → 核心公式 / 定理 / 概念
      查询 2 → 例题或分步解答
      查询 3 → 边界情况、常见错误或应用（如果需要）

    ── 不要用于 ─────────────────────────────────────────────────────────────
    • 数学计算 — 使用你自己的推理或 calculator_tool
    • 学生上传笔记中已覆盖的主题 — 先使用 rag_tool

    参数：
        query: 聚焦的、具体的搜索查询。
               示例：
                 "2025年高考数学新题型"
                 "黎曼猜想最新进展 2024 2025"
                 "分部积分高考压轴题精讲"

    返回：
        Tavily AI 直接回答 + 前 5 条结果（标题、URL、摘要）。
    """
    if not query.strip():
        return "未提供查询词。"

    logger.info(f"[TavilyMCP] web_search_tool | query='{query[:80]}'")

    result = tavily_mcp_search(
        query          = query,
        search_depth   = "advanced",
        topic          = "general",
        max_results    = 5,
    )

    logger.info(f"[TavilyMCP] web_search_tool done | {len(result)} chars returned")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 3 — SYMBOLIC CALCULATOR  (SymPy)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def calculator_tool(expression: str) -> str:
    """
    符号数学计算器（SymPy 后端）。请谨慎使用。

    解题模型自己处理所有常规的中学数学计算。
    仅在以下三种狭窄情况下调用：

      1. 非常大的阶乘 / 组合数，如 C(50,25)、100!
      2. 题目明确要求的高精度小数结果
      3. 大型矩阵运算（行列式、逆矩阵、特征值）

    不要用于：基础运算、三角恒等式、标准积分、导数或概率分数。
    这些调用没有价值。

    有效的 SymPy 表达式语法：
      "binomial(50, 25)"
      "factorial(100)"
      "N(integrate(1/sqrt(1-x**2), x), 50)"    ← 50 位精度
      "Matrix([[1,2,3],[4,5,6],[7,8,9]]).det()"

    参数：
        expression: 一个有效的 SymPy 表达式字符串。

    返回：
        数值或符号结果（字符串）。
    """
    try:
        expr   = sp.sympify(expression)
        result = sp.N(expr)
        logger.info(
            f"[Calculator] {expression[:60]} → {str(result)[:60]}"
        )
        return str(result)
    except Exception as exc:
        return f"计算器错误：{exc}。请检查 SymPy 表达式语法。"


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    rag_tool,
    web_search_tool,
    calculator_tool,
]