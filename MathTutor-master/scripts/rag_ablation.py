# -*- coding: utf-8 -*-
"""
RAG 消融测试（多场景版）：纯 BM25 vs 纯 FAISS vs 混合 RRF
========================================================
三个场景：
  场景1 小语料基线  ：20 个数学知识 chunks（原测试）
  场景2 大语料      ：600 chunks（10 主题 × 30 变体 + 300 条无关文本）
  场景3 噪声/诱饵   ：大语料 + 100 条"表面相关实则无关"干扰文档
                      （含数学关键词但语义无关，考验检索鲁棒性）

评估指标：Recall@K（K=5）+ 检索耗时（不含嵌入）+ 端到端耗时（含嵌入）

用法：
  PYTHONPATH=src python scripts/rag_ablation.py
"""
import sys
import time
import re
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from backend.agents.nodes.tools import EMBED_DIM, TOP_K, MIN_SCORE
from backend.agents.nodes.tools.tools import (
    _embed_texts,
    _tokenize,
    EMBED_INPUT_TYPE_QUERY,
    EMBED_INPUT_TYPE_DOC,
)


# ── Reranker（cross-encoder）──────────────
# 用本地缓存的 BGE-large-zh-v1.5 构造 CrossEncoder，避免联网下载 bge-reranker。
# 真实生产建议换 FlagEmbedding 的 bge-reranker-v2-m3（专为重排序训练）。
_RERANKER = None

def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _HF_CACHE = os.path.expanduser(
            r"~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5"
            r"/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116"
        )
        _RERANKER = CrossEncoder(_HF_CACHE)
    return _RERANKER


def _rerank(reranker, query, doc_indices, docs, top_k=TOP_K):
    """cross-encoder 重排序：对候选段落重新打分，返回排序后的 Top-K 索引"""
    if not doc_indices:
        return []
    pairs = [(query, docs[i]) for i in doc_indices]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(doc_indices, scores), key=lambda x: -x[1])
    return [int(i) for i, _ in ranked[:top_k]]


def _tokenize_zh_bigram(text: str):
    """零依赖中文分词：字符 bigram（双字切分），中文检索经典做法"""
    s = re.sub(r"\s+", "", text)
    if len(s) <= 1:
        return list(s)
    return [s[i:i + 2] for i in range(len(s) - 1)]


# ══════════════════════════════════════════════════════════════════════════
# 场景 1：小语料（数学知识手册，20 chunks）
# ══════════════════════════════════════════════════════════════════════════
SMALL_DOCS = [
    "二次函数的一般形式为 y = ax^2 + bx + c（a ≠ 0），其图像是一条抛物线。当 a > 0 时开口向上，当 a < 0 时开口向下。",
    "二次函数的顶点坐标公式：顶点横坐标 x = -b/(2a)，代入原式可得顶点纵坐标。顶点是抛物线的最高点或最低点。",
    "二次函数的对称轴是直线 x = -b/(2a)，与顶点横坐标相同。抛物线与对称轴对称。",
    "二次函数与 x 轴交点的个数由判别式 Δ = b^2 - 4ac 决定：Δ > 0 有两个交点，Δ = 0 有一个交点，Δ < 0 没有交点。",
    "求二次函数最值的方法：先配方为 y = a(x - h)^2 + k 的形式，则顶点为 (h, k)，最值为 k。",
    "条件概率 P(A|B) 表示在事件 B 发生的条件下事件 A 发生的概率，公式为 P(A|B) = P(A∩B) / P(B)，其中 P(B) > 0。",
    "贝叶斯定理：P(A|B) = P(B|A)·P(A) / P(B)，用于根据新的证据 B 更新对事件 A 的信念。",
    "全概率公式：若事件 B1, B2, ..., Bn 构成完备事件组，则 P(A) = Σ P(Bi)·P(A|Bi)。",
    "先验概率是在获得新证据之前对事件概率的估计，后验概率是在获得证据之后修正的概率。",
    "导数定义：函数 f(x) 在 x0 处的导数 f'(x0) = lim(Δx→0) [f(x0+Δx) - f(x0)] / Δx，表示切线斜率。",
    "幂函数求导法则：d/dx (x^n) = n·x^(n-1)。例如 d/dx (x^2) = 2x。",
    "链式法则：复合函数 (f∘g)(x) 的导数为 f'(g(x))·g'(x)。",
    "不定积分是导数的逆运算：∫ f'(x) dx = f(x) + C，其中 C 为积分常数。",
    "正弦定理：三角形中 a/sin A = b/sin B = c/sin C = 2R，R 为外接圆半径。",
    "余弦定理：c^2 = a^2 + b^2 - 2ab·cos C，用于已知两边及夹角求第三边。",
    "两角和公式：sin(A+B) = sinA·cosB + cosA·sinB，cos(A+B) = cosA·cosB - sinA·sinB。",
    "同角三角函数基本关系：sin^2 θ + cos^2 θ = 1，tan θ = sin θ / cos θ。",
    "等差数列通项公式：an = a1 + (n-1)d，前 n 项和 Sn = n(a1 + an)/2 = na1 + n(n-1)d/2。",
    "等比数列通项公式：an = a1·q^(n-1)，前 n 项和 Sn = a1(1-q^n)/(1-q)（q ≠ 1）。",
    "等差数列求和的关键是找到首项 a1 和公差 d；等比数列要注意公比 q 是否为 1 的分类讨论。",
]
SMALL_QUERIES = [
    {"q": "二次函数顶点坐标怎么求", "relevant": [1, 2, 4]},
    {"q": "抛物线对称轴是什么",     "relevant": [0, 2]},
    {"q": "判别式决定什么",         "relevant": [3]},
    {"q": "贝叶斯定理公式",         "relevant": [6, 5]},
    {"q": "条件概率怎么计算",       "relevant": [5, 7]},
    {"q": "导数定义切线斜率",       "relevant": [9, 10]},
    {"q": "幂函数求导",             "relevant": [10]},
    {"q": "正弦定理",               "relevant": [13]},
    {"q": "余弦定理求边",           "relevant": [14]},
    {"q": "等差数列前n项和",        "relevant": [17, 19]},
    {"q": "等比数列求和",           "relevant": [18, 19]},
]


# ══════════════════════════════════════════════════════════════════════════
# 场景 2 & 3：大语料生成（模板 + 无关文本 + 诱饵文本）
# ══════════════════════════════════════════════════════════════════════════
TOPIC_TEMPLATES = {
    "quadratic": [
        "二次函数 y = a{{x}}^2 + b{{x}} + c 的图像是抛物线，{i}",
        "抛物线顶点公式 x = -b/(2a)，{i}",
        "二次函数对称轴 x = -b/(2a)，{i}",
        "判别式 Δ = b^2 - 4ac 决定交点个数，{i}",
        "配方法求二次函数最值 y = a(x-h)^2 + k，{i}",
    ],
    "bayes": [
        "条件概率 P(A|B) = P(A∩B)/P(B)，{i}",
        "贝叶斯定理 P(A|B) = P(B|A)P(A)/P(B)，{i}",
        "全概率公式 P(A) = ΣP(Bi)P(A|Bi)，{i}",
        "先验概率与后验概率的更新，{i}",
    ],
    "calculus": [
        "导数定义 f'(x0) = lim Δx→0 差商，{i}",
        "幂函数求导 d/dx(x^n) = n·x^(n-1)，{i}",
        "链式法则复合函数求导，{i}",
        "不定积分是导数的逆运算，{i}",
    ],
    "trig": [
        "正弦定理 a/sinA = b/sinB = 2R，{i}",
        "余弦定理 c^2 = a^2 + b^2 - 2ab·cosC，{i}",
        "两角和公式 sin(A+B)、cos(A+B)，{i}",
        "同角关系 sin^2θ + cos^2θ = 1，{i}",
    ],
    "sequence": [
        "等差数列通项 an = a1 + (n-1)d，{i}",
        "等比数列通项 an = a1·q^(n-1)，{i}",
        "等差数列求和 Sn = n(a1+an)/2，{i}",
        "等比数列求和 Sn = a1(1-q^n)/(1-q)，{i}",
    ],
    "geometry": [
        "三角形内角和为 180 度，{i}",
        "勾股定理 a^2 + b^2 = c^2，{i}",
        "圆面积公式 S = πr^2，{i}",
        "相似三角形对应边成比例，{i}",
    ],
    "inequality": [
        "基本不等式 a+b ≥ 2√(ab)，{i}",
        "柯西不等式 (a^2+b^2)(c^2+d^2) ≥ (ac+bd)^2，{i}",
        "绝对值不等式 |a+b| ≤ |a|+|b|，{i}",
    ],
    "log": [
        "对数定义 a^x = N 则 x = log_a N，{i}",
        "对数运算法则 log(ab) = log a + log b，{i}",
        "换底公式 log_a b = log_c b / log_c a，{i}",
    ],
    "complex": [
        "复数 z = a + bi 的模 |z| = √(a^2+b^2)，{i}",
        "复数乘法与共轭，{i}",
    ],
    "vector": [
        "向量点积 a·b = |a||b|cosθ，{i}",
        "向量叉积与平行四边形面积，{i}",
    ],
}

# 无关文本（模拟知识库里大量不相关内容）
IRRELEVANT_TEMPLATES = [
    "今天天气很好，适合去公园散步，{i}",
    "最新款智能手机的电池续航测试报告，{i}",
    "中国历史上的重要朝代更迭时间表，{i}",
    "健康饮食指南：每天应该摄入多少蔬菜水果，{i}",
    "如何训练宠物狗学会坐下和握手，{i}",
    "城市地铁线路规划与客流分析，{i}",
    "咖啡烘焙程度对口感的影响研究，{i}",
    "现代艺术流派的发展脉络简介，{i}",
    "新能源汽车充电桩布局优化方案，{i}",
    "游泳初学者需要注意的安全事项，{i}",
]

# 诱饵文本：包含数学关键词但内容无关（考验检索鲁棒性）
DECOY_TEMPLATES = [
    "有人说二次函数在生活中完全没用，你怎么看，{i}",
    "关于贝叶斯定理名字来源的趣闻轶事，{i}",
    "学生吐槽导数太难了的段子合集，{i}",
    "正弦定理在古埃及测量土地中的历史故事，{i}",
    "数列问题在公务员考试中的出现频率分析，{i}",
    "吐槽高中数学教材编排不合理的长文，{i}",
    "数学家欧拉的生平故事与八卦，{i}",
    "为什么很多人觉得函数符号很吓人，{i}",
]


def generate_large_corpus(n_variants: int = 30, n_irrelevant: int = 300):
    """生成大语料：10 主题 × n_variants 变体 + 无关文本"""
    docs, topics = [], []
    for topic, tmpls in TOPIC_TEMPLATES.items():
        for i in range(n_variants):
            t = tmpls[i % len(tmpls)].format(i=i)
            docs.append(t)
            topics.append(topic)
    for i in range(n_irrelevant):
        t = IRRELEVANT_TEMPLATES[i % len(IRRELEVANT_TEMPLATES)].format(i=i)
        docs.append(t)
        topics.append("irrelevant")
    return docs, topics


def add_decoy_chunks(docs, topics, n_decoy: int = 100):
    """在大语料基础上加入诱饵 chunk（关键词相关但语义无关）"""
    decoys = [
        DECOY_TEMPLATES[i % len(DECOY_TEMPLATES)].format(i=i)
        for i in range(n_decoy)
    ]
    return docs + decoys, topics + ["decoy"] * n_decoy


# 大语料查询：relevant 是该主题的全部 chunk 索引
def build_large_queries(topics):
    qs = [
        ("二次函数顶点坐标怎么求", "quadratic"),
        ("贝叶斯定理公式", "bayes"),
        ("导数定义切线斜率", "calculus"),
        ("正弦定理", "trig"),
        ("等差数列前n项和", "sequence"),
        ("勾股定理", "geometry"),
        ("基本不等式", "inequality"),
        ("换底公式", "log"),
        ("复数模长", "complex"),
        ("向量点积", "vector"),
    ]
    queries = []
    for q, topic in qs:
        relevant = [i for i, t in enumerate(topics) if t == topic]
        queries.append({"q": q, "relevant": relevant})
    return queries


def build_formula_queries(docs, topics):
    """
    公式/符号检索查询：混合"精确数学符号 + 语义提问"。
    这是数学辅导的真实场景（学生常问"公式是什么"）——
    BGE 向量对符号编码会漂移（把不同公式变体混在一起），
    而 BM25 bigram 能精确匹配符号串，互补优势在此显现。
    """
    def find(topic, kw):
        hits = [i for i, t in enumerate(topics) if t == topic and kw in docs[i]]
        return hits[0] if hits else None

    specs = [
        ("sin(A+B) 展开公式是什么", "trig", "两角和"),
        ("写出 sin(A+B) 和 cos(A+B)", "trig", "两角和"),
        ("sin^2θ + cos^2θ 等于几", "trig", "同角"),
        ("a/sinA = b/sinB 叫什么定理", "trig", "正弦定理"),
        ("c^2 = a^2+b^2-2ab·cosC 求什么", "trig", "余弦定理"),
        ("Sn = n(a1+an)/2 是什么公式", "sequence", "Sn = n"),
        ("an = a1·q^(n-1) 哪种数列", "sequence", "等比数列通项"),
        ("x = -b/(2a) 求什么", "quadratic", "顶点公式"),
        ("Δ = b^2-4ac 决定什么", "quadratic", "判别式"),
        ("d/dx(x^n) 等于多少", "calculus", "幂函数求导"),
    ]
    queries = []
    for q, topic, kw in specs:
        idx = find(topic, kw)
        if idx is not None:
            queries.append({"q": q, "relevant": [idx]})
    return queries


# ══════════════════════════════════════════════════════════════════════════
# 检索实现（参数化语料，与 tools.py 同款算法）
# ══════════════════════════════════════════════════════════════════════════
def build_indexes(docs):
    tokenized = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    tokenized_zh = [_tokenize_zh_bigram(d) for d in docs]
    bm25_zh = BM25Okapi(tokenized_zh)
    doc_vecs = np.asarray(_embed_texts(docs, EMBED_INPUT_TYPE_DOC), dtype=np.float32)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(doc_vecs)
    return bm25, bm25_zh, index, doc_vecs


def _search_dense(index, q_vec, k=10):
    _, idx = index.search(q_vec, k)
    return [int(i) for i in idx[0]]


def _search_sparse(bm25, query, k=10, min_score=0.0):
    scores = bm25.get_scores(_tokenize(query))
    idx = np.argsort(scores)[::-1][:k]
    return [int(i) for i in idx if scores[i] > min_score]


def _search_sparse_zh(bm25_zh, query, k=10, min_score=0.0):
    scores = bm25_zh.get_scores(_tokenize_zh_bigram(query))
    idx = np.argsort(scores)[::-1][:k]
    return [int(i) for i in idx if scores[i] > min_score]


def _rrf_fuse(dense_idx, sparse_idx, n_docs, k=TOP_K, K=60):
    rrf = {}
    for rank, idx in enumerate(dense_idx):
        if 0 <= idx < n_docs:
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K + rank + 1)
    for rank, idx in enumerate(sparse_idx):
        if 0 <= idx < n_docs:
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K + rank + 1)
    return sorted(rrf, key=rrf.get, reverse=True)[:k]


def _rrf_fuse_crag(dense_idx, sparse_idx, q_vec, doc_vecs, n_docs, k=TOP_K, K=60, min_score=MIN_SCORE):
    """RRF 融合 + CRAG 余弦过滤（与 rag_tool 的纠错层完全一致）：
    融合后逐个算余弦相似度，< min_score 的丢弃（诱饵/噪声挡在门外）"""
    rrf = {}
    for rank, idx in enumerate(dense_idx):
        if 0 <= idx < n_docs:
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K + rank + 1)
    for rank, idx in enumerate(sparse_idx):
        if 0 <= idx < n_docs:
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K + rank + 1)

    fused = sorted(rrf, key=rrf.get, reverse=True)
    kept = []
    for idx in fused:
        if len(kept) >= k:
            break
        cos = float(np.dot(q_vec[0], doc_vecs[idx]))
        if cos >= min_score:
            kept.append(int(idx))
    return kept


def make_methods(bm25, bm25_zh, index, doc_vecs, n_docs, docs=None):
    methods = {
        "BM25-only (原分词)": lambda q, qv: _search_sparse(bm25, q, k=TOP_K),
        "BM25-only (bigram)": lambda q, qv: _search_sparse_zh(bm25_zh, q, k=TOP_K),
        "FAISS-only (稠密)": lambda q, qv: _search_dense(index, qv, k=TOP_K),
        "Hybrid RRF (现有)": lambda q, qv: _rrf_fuse(
            _search_dense(index, qv), _search_sparse(bm25, q), n_docs),
        "Hybrid RRF (bigram)": lambda q, qv: _rrf_fuse(
            _search_dense(index, qv), _search_sparse_zh(bm25_zh, q), n_docs),
        "Hybrid RRF (bigram+CRAG)": lambda q, qv: _rrf_fuse_crag(
            _search_dense(index, qv), _search_sparse_zh(bm25_zh, q), qv, doc_vecs, n_docs),
        "Hybrid RRF (现有+CRAG)": lambda q, qv: _rrf_fuse_crag(
            _search_dense(index, qv), _search_sparse(bm25, q), qv, doc_vecs, n_docs),
    }
    # reranker 方法（懒加载，避免小场景也加载模型）
    if docs is not None:
        methods["Hybrid+RRF+Reranker"] = lambda q, qv, _docs=docs: _rerank(
            _get_reranker(), q,
            _rrf_fuse(_search_dense(index, qv), _search_sparse_zh(bm25_zh, q), n_docs, k=10),
            _docs)
        methods["FAISS+Reranker"] = lambda q, qv, _docs=docs: _rerank(
            _get_reranker(), q,
            _search_dense(index, qv, k=10),
            _docs)
    return methods


def recall_at_k(retrieved, relevant):
    if not relevant:
        return 0.0
    hit = len(set(retrieved) & set(relevant))
    return hit / min(len(relevant), TOP_K)  # 归一化到 K


# ══════════════════════════════════════════════════════════════════════════
# 场景运行器
# ══════════════════════════════════════════════════════════════════════════
def run_scenario(name, docs, queries):
    print("\n" + "=" * 76)
    print(f"场景：{name}  |  语料 {len(docs)} chunks | 查询 {len(queries)} 条 | K={TOP_K}")
    print("=" * 76)

    t0 = time.perf_counter()
    bm25, bm25_zh, index, doc_vecs = build_indexes(docs)
    build_time = time.perf_counter() - t0
    print(f"[索引构建] 含 BGE 编码 {len(docs)} chunks: {build_time:.1f}s")

    methods = make_methods(bm25, bm25_zh, index, doc_vecs, len(docs), docs=docs)
    results = {}
    for mname, fn in methods.items():
        recalls, recalls1, t_search, t_e2e = [], [], 0.0, 0.0
        for item in queries:
            q, rel = item["q"], item["relevant"]
            t0 = time.perf_counter()
            q_vec = np.asarray(_embed_texts([q], EMBED_INPUT_TYPE_QUERY), dtype=np.float32)
            t_e2e += time.perf_counter() - t0
            t0 = time.perf_counter()
            ret = fn(q, q_vec)
            t_search += time.perf_counter() - t0
            recalls.append(recall_at_k(ret, rel))
            # Recall@1：第一个结果是否相关（体现"排序质量"，公式场景的关键指标）
            recalls1.append(1.0 if ret and ret[0] in rel else 0.0)
        n = len(queries)
        results[mname] = {
            "recall": np.mean(recalls),
            "recall1": np.mean(recalls1),
            "search_ms": t_search / n * 1000,
            "e2e_ms": t_e2e / n * 1000,
        }

    print(f"\n{'方法':<26}{'Recall@1':>10}{'Recall@5':>10}{'检索耗时(ms)':>16}{'端到端(ms)':>16}")
    print("-" * 76)
    for mname, r in results.items():
        print(f"{mname:<26}{r['recall1']*100:>9.1f}%{r['recall']*100:>9.1f}%{r['search_ms']:>14.2f}{r['e2e_ms']:>16.1f}")
    return results


# ══════════════════════════════════════════════════════════════════════════
# 阈值扫描：在噪声场景下测不同 min_score 对 Recall 的影响
# ══════════════════════════════════════════════════════════════════════════
def scan_thresholds(docs, queries, label):
    """扫描 min_score ∈ {0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}"""
    print("\n" + "=" * 76)
    print(f"阈值扫描：{label}（bigram + CRAG，不同余弦阈值）")
    print("=" * 76)

    bm25, bm25_zh, index, doc_vecs = build_indexes(docs)
    n_docs = len(docs)
    thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    all_qvecs = []
    for item in queries:
        q_vec = np.asarray(_embed_texts([item["q"]], EMBED_INPUT_TYPE_QUERY), dtype=np.float32)
        all_qvecs.append(q_vec)

    print(f"\n{'min_score':<12}{'Recall@5':>12}{'平均返回数':>12}")
    print("-" * 40)
    for thr in thresholds:
        recalls, ret_counts = [], []
        for item, q_vec in zip(queries, all_qvecs):
            ret = _rrf_fuse_crag(
                _search_dense(index, q_vec),
                _search_sparse_zh(bm25_zh, item["q"]),
                q_vec, doc_vecs, n_docs, min_score=thr,
            )
            recalls.append(recall_at_k(ret, item["relevant"]))
            ret_counts.append(len(ret))
        avg_recall = np.mean(recalls)
        avg_ret = np.mean(ret_counts)
        flag = " ◀ 推荐" if thr == 0.50 else ""
        print(f"{thr:<12.2f}{avg_recall*100:>10.1f}%{avg_ret:>12.1f}{flag}")

    # 附加：看每个阈值下，诱饵被召回的占比
    decoy_topics = ["decoy"]
    print("\n诱饵(decoy)被召回进 Top-5 的占比（越低越好）：")
    print(f"{'min_score':<12}{'诱饵占比':>12}")
    print("-" * 40)
    for thr in thresholds:
        decoy_hits = 0
        for item, q_vec in zip(queries, all_qvecs):
            ret = _rrf_fuse_crag(
                _search_dense(index, q_vec),
                _search_sparse_zh(bm25_zh, item["q"]),
                q_vec, doc_vecs, n_docs, min_score=thr,
            )
            # 需要 topics 信息，这里通过 doc_vecs 索引 + 查询主题判断
            decoy_hits += sum(1 for i in ret if 600 <= i < 600 + 100)  # 诱饵在最后100个
        print(f"{thr:<12.2f}{decoy_hits / len(queries):>10.1f} 个/查询")
    return


# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("RAG 消融测试（多场景）")
    print("说明：FAISS 用暴力内积搜索（与项目 IndexFlatIP 一致），大语料下体现检索算法差异\n")

    # 场景 1：小语料基线
    run_scenario("① 小语料基线（数学知识 20 chunks）", SMALL_DOCS, SMALL_QUERIES)

    # 场景 2：大语料（600 chunks）
    print("\n生成大语料中…")
    docs2, topics2 = generate_large_corpus(n_variants=30, n_irrelevant=300)
    qs2 = build_large_queries(topics2)
    run_scenario("② 大语料（10主题×30变体 + 300无关 = 600）", docs2, qs2)

    # 场景 3：噪声/诱饵（700 chunks）
    docs3, topics3 = add_decoy_chunks(docs2, topics2, n_decoy=100)
    qs3 = build_large_queries(topics3)
    run_scenario("③ 噪声/诱饵（大语料 + 100 关键词诱饵 = 700）", docs3, qs3)

    # 场景 4：公式/符号检索（体现混合检索互补优势）
    docs4, topics4 = generate_large_corpus(n_variants=30, n_irrelevant=300)
    qs4 = build_formula_queries(docs4, topics4)
    run_scenario("④ 公式/符号查询（数学公式精确匹配，混合检索优势场景）", docs4, qs4)

    # 阈值扫描（噪声场景专用）
    scan_thresholds(docs3, qs3, "噪声/诱饵场景")

    print("\n" + "=" * 76)
    print("结论指引：")
    print("  - 场景②看大语料下各方法是否仍能召回（FAISS 暴力扫描的退化）")
    print("  - 场景③看诱饵文档（关键词相关但语义无关）对每种检索的干扰程度")
    print("  - 场景④是混合检索的优势场景：公式/符号精确匹配时，BGE 向量")
    print("    对符号编码会漂移，BM25 bigram 精确匹配符号串补上盲区")
    print("  - 'bigram+CRAG' 是推荐组合：bigram 修好中文 BM25，CRAG 余弦过滤挡诱饵")
    print("=" * 76)


if __name__ == "__main__":
    main()
