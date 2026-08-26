from __future__ import annotations
import hashlib
import time

_USER_PREFIX = "user:"

def _short_id(full_id: str, length: int = 8) -> str:
    return full_id[:length] + "…"


def _epoch_to_date(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.localtime(float(ts)))
    except Exception:
        return str(ts)


def build_graph_data(
    student_id: str,
    redis_client,
    checkpointer,
    get_thread_history,
    max_threads: int = 15,
    include_agent_nodes: bool = True,
) -> dict:
    """
    Returns {"nodes": [...], "edges": [...]} ready for vis.js.

    Key pattern alignment (must match db_utils.py / memory_manager.py):
        episodic:{student_id}:{episode_id}   — JSON doc, client.json().set(...)
        semantic:{student_id}                — JSON doc, client.json().set(...)
        procedural:{student_id}              — JSON doc, client.json().set(...)
        user:{student_id}                    — Redis hash, client.hset(...)
        thread:{thread_id}:meta              — Redis hash, client.hset(...)
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_node_ids: set[str] = set()

    def _add_node(node: dict):
        if node["id"] not in seen_node_ids:
            seen_node_ids.add(node["id"])
            nodes.append(node)

    def _add_edge(edge: dict):
        edges.append(edge)

    # ── Student root ──────────────────────────────────────────────────────────
    try:
        user_raw = redis_client.hgetall(f"{_USER_PREFIX}{student_id}")
        user_info = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in user_raw.items()
        } if user_raw else {}
    except Exception:
        user_info = {}

    _add_node({
        "id":     student_id,
        "label":  user_info.get("display_name", "学生"),
        "type":   "student",
        "group":  "student",
        "detail": {
            "学生ID":      _short_id(student_id),
            "姓名":            user_info.get("display_name", "—"),
            "邮箱":           user_info.get("email", "—"),
            "已解决问题": user_info.get("total_problems_solved", "—"),
            "加入时间":    _epoch_to_date(user_info.get("created_at", 0)),
            "上次登录":      _epoch_to_date(user_info.get("last_login", 0)),
        },
        "title": f"<b>{user_info.get('display_name','学生')}</b><br>根节点",
    })

    # ── Thread / STM nodes ────────────────────────────────────────────────────
    try:
        threads_meta = get_thread_history(student_id)[:max_threads]
    except Exception:
        threads_meta = []

    for meta in threads_meta:
        tid     = meta.get("thread_id", "")
        summary = (meta.get("problem_summary") or "")[:35] or _short_id(tid)
        topic   = meta.get("topic", "")
        outcome = meta.get("outcome", "")

        session_node = {
            "id":     tid,
            "label":  summary,
            "type":   "session",
            "group":  "session",
            "detail": {
                "会话ID": _short_id(tid, 12),
                "题目":   summary,
                "主题":     topic or "—",
                "结果":   outcome or "—",
                "日期":      _epoch_to_date(meta.get("created_at", 0)),
            },
            "title": f"<b>{summary}</b><br>主题: {topic}",
        }
        _add_node(session_node)
        _add_edge({"from": student_id, "to": tid, "label": "会话",
                   "arrows": "to", "dashes": False})

        if not include_agent_nodes:
            continue

        try:
            cfg  = {"configurable": {"thread_id": tid}}
            snap = checkpointer.get(cfg)
            # RedisSaver.get() returns a CheckpointTuple(config, checkpoint, metadata, ...)
            # .values lives on the checkpoint field, not directly on the tuple
            if snap is None:
                vals = {}
            elif hasattr(snap, "checkpoint") and isinstance(getattr(snap, "checkpoint", None), dict):
                # LangGraph RedisSaver returns a CheckpointTuple; channel_values is the state
                vals = snap.checkpoint.get("channel_values", {})
            elif hasattr(snap, "values") and isinstance(snap.values, dict):
                vals = snap.values
            else:
                vals = {}
        except Exception:
            vals = {}

        # Agent payload log → one node per entry
        for entry in (vals.get("agent_payload_log") or []):
            node_name = entry.get("node", "unknown")
            nid       = f"{tid}__{node_name}"
            fields    = {k: str(v)[:120] for k, v in (entry.get("fields") or {}).items()
                         if v is not None and str(v) not in ("", "None", "none")}
            _add_node({
                "id":     nid,
                "label":  node_name.replace("_", "\n"),
                "type":   "agent",
                "group":  "agent",
                "detail": {
                    "节点":    node_name,
                    "摘要": entry.get("summary", "")[:100],
                    **fields,
                },
                "title": f"<b>{node_name}</b><br>{entry.get('summary','')[:80]}",
            })
            _add_edge({"from": tid, "to": nid, "label": "运行节点",
                       "arrows": "to", "dashes": True})

        # Tool calls from messages
        for msg in (vals.get("messages") or []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                tool_name = tc.get("name", "tool")
                tnid      = f"{tid}__tool__{tool_name}"
                _add_node({
                    "id":     tnid,
                    "label":  tool_name.replace("_", "\n"),
                    "type":   "tool",
                    "group":  "tool",
                    "detail": {
                        "工具":  tool_name,
                        "参数":  str(tc.get("args", {}))[:200],
                        "会话": _short_id(tid, 12),
                    },
                    "title": f"<b>{tool_name}</b>",
                })
                _add_edge({"from": tid, "to": tnid, "label": "使用工具",
                           "arrows": "to", "dashes": True})

    # ── LTM nodes — FIXED key patterns to match db_utils.py / memory_manager.py
    # ─────────────────────────────────────────────────────────────────────────
    # episodic:{student_id}:{episode_id}  → JSON doc via client.json().set(...)
    # semantic:{student_id}               → JSON doc via client.json().set(...)
    # procedural:{student_id}             → JSON doc via client.json().set(...)

    # ── Episodic memories ─────────────────────────────────────────────────────
    try:
        ep_keys = redis_client.keys(f"episodic:{student_id}:*")
    except Exception:
        ep_keys = []

    for raw_key in ep_keys:
        key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
        try:
            raw_data = redis_client.json().get(key, "$")
            mem = raw_data[0] if raw_data else {}
        except Exception:
            mem = {}
        if not mem:
            continue

        ep_id   = mem.get("episode_id", key.split(":")[-1])
        node_id = f"episodic__{ep_id}"
        summary = (mem.get("problem_summary") or ep_id)[:45]

        decay_val = mem.get("decay_score", "—")
        _add_node({
            "id":     node_id,
            "label":  summary[:30],
            "type":   "episodic",
            "group":  "episodic",
            "detail": {
                "类型":        "情景记忆",
                "题目":     summary,
                "主题":       mem.get("topic", "—"),
                "难度":  mem.get("difficulty", "—"),
                "结果":     mem.get("outcome", "—"),
                "答案":      mem.get("final_answer", "—"),
                "尝试次数":    str(mem.get("solve_attempts", "—")),
                "衰减分数": str(decay_val),
                "创建时间":     _epoch_to_date(mem.get("timestamp", 0)),
            },
            "title": f"<b>情景记忆</b><br>{summary}",
        })
        _add_edge({
            "from":   student_id,
            "to":     node_id,
            "label":  "情景记忆",
            "arrows": "to",
            "dashes": False,
        })

    # ── Semantic memory ───────────────────────────────────────────────────────
    try:
        sem_raw = redis_client.json().get(f"semantic:{student_id}", "$")
        sem = sem_raw[0] if sem_raw else {}
    except Exception:
        sem = {}

    if sem:
        weak   = sem.get("weak_topics", {})
        strong = sem.get("strong_topics", {})
        patterns = sem.get("mistake_patterns", [])

        # One node for the semantic profile
        sem_node_id = f"semantic__{student_id}"
        weak_str   = ", ".join(f"{k}({v})" for k, v in weak.items() if v > 0) or "—"
        strong_str = ", ".join(f"{k}({v})" for k, v in strong.items() if v > 0) or "—"
        pat_str    = "; ".join(
            p.get("pattern", "")[:40] for p in patterns[:3]
        ) or "—"
        _add_node({
            "id":     sem_node_id,
            "label":  "语义画像",
            "type":   "semantic",
            "group":  "semantic",
            "detail": {
                "类型":             "语义记忆",
                "薄弱主题":      weak_str[:120],
                "擅长主题":    strong_str[:120],
                "错误模式": pat_str[:200],
                "最近更新":     _epoch_to_date(sem.get("last_updated", 0)),
            },
            "title": "<b>语义记忆</b><br>主题强弱与错误",
        })
        _add_edge({
            "from":   student_id,
            "to":     sem_node_id,
            "label":  "语义记忆",
            "arrows": "to",
            "dashes": False,
        })

        # One child node per weak topic (count > 0)
        for topic, count in weak.items():
            if count < 1:
                continue
            wnid = f"semantic_weak__{student_id}__{topic}"
            _add_node({
                "id":    wnid,
                "label": topic,
                "type":  "semantic",
                "group": "semantic",
                "detail": {
                    "类型":       "薄弱主题",
                    "主题":      topic,
                    "失败次数": str(count),
                },
                "title": f"<b>薄弱: {topic}</b><br>{count} 次出错",
            })
            _add_edge({"from": sem_node_id, "to": wnid, "label": "薄弱主题",
                       "arrows": "to", "dashes": True})

        # One child node per strong topic (count > 0)
        for topic, count in strong.items():
            if count < 1:
                continue
            snid = f"semantic_strong__{student_id}__{topic}"
            _add_node({
                "id":    snid,
                "label": topic,
                "type":  "semantic",
                "group": "semantic",
                "detail": {
                    "类型":          "擅长主题",
                    "主题":         topic,
                    "成功次数": str(count),
                },
                "title": f"<b>擅长: {topic}</b><br>{count} 次成功",
            })
            _add_edge({"from": sem_node_id, "to": snid, "label": "擅长主题",
                       "arrows": "to", "dashes": True})

        # One child node per unique mistake pattern
        for pat in patterns:
            pattern_text = pat.get("pattern", "")
            pat_topic    = pat.get("topic", "")
            pat_count    = pat.get("count", 1)
            if not pattern_text:
                continue
            # Use a hash of pattern text as a stable node ID
            pid  = hashlib.md5(f"{student_id}{pat_topic}{pattern_text}".encode()).hexdigest()[:10]
            mnid = f"semantic_mistake__{student_id}__{pid}"
            _add_node({
                "id":    mnid,
                "label": pattern_text[:28] + ("…" if len(pattern_text) > 28 else ""),
                "type":  "semantic",
                "group": "semantic",
                "detail": {
                    "类型":    "错误模式",
                    "主题":   pat_topic or "—",
                    "模式": pattern_text[:200],
                    "次数":   str(pat_count),
                },
                "title": f"<b>错误 ({pat_topic})</b><br>{pattern_text[:60]}",
            })
            _add_edge({"from": sem_node_id, "to": mnid, "label": "错误模式",
                       "arrows": "to", "dashes": True})

    # ── Procedural memory ─────────────────────────────────────────────────────
    try:
        proc_raw = redis_client.json().get(f"procedural:{student_id}", "$")
        proc = proc_raw[0] if proc_raw else {}
    except Exception:
        proc = {}

    if proc:
        strats = proc.get("strategy_success", {})
        proc_node_id = f"procedural__{student_id}"

        # Show only the BEST strategy per topic in the summary (not all strategies)
        best_per_topic = []
        for _topic, _topic_strats in strats.items():
            if not _topic_strats:
                continue
            _best_name, _best_data = max(
                _topic_strats.items(),
                key=lambda kv: (kv[1].get("success_rate", 0),
                                -kv[1].get("attempts_avg", 99)),
            )
            best_per_topic.append(
                f"{_topic}: {_best_name[:40]} "
                f"({_best_data.get('success_rate', 0):.0%} | "
                f"avg {_best_data.get('attempts_avg', '?')} attempts)"
            )

        _add_node({
            "id":     proc_node_id,
            "label":  "程序画像",
            "type":   "procedural",
            "group":  "procedural",
            "detail": {
                "类型":         "程序记忆",
                "各主题最佳策略": "; ".join(best_per_topic[:5]) or "—",
                "追踪主题数": str(len(strats)),
                "最近更新": _epoch_to_date(proc.get("last_updated", 0)),
            },
            "title": "<b>程序记忆</b><br>解题策略有效性",
        })
        _add_edge({
            "from":   student_id,
            "to":     proc_node_id,
            "label":  "程序记忆",
            "arrows": "to",
            "dashes": False,
        })

        # One child node per topic strategy
        for topic, topic_strats in strats.items():
            if not topic_strats:
                continue
            best = max(topic_strats.items(),
                       key=lambda kv: (kv[1].get("success_rate", 0),
                                       -kv[1].get("attempts_avg", 99)))
            strat_name, data = best
            snid = f"procedural_strat__{student_id}__{topic}"
            _add_node({
                "id":    snid,
                "label": topic,
                "type":  "procedural",
                "group": "procedural",
                "detail": {
                    "类型":         "解题策略",
                    "主题":        topic,
                    "最佳策略":strat_name,
                    "成功率": f"{data.get('success_rate', 0):.0%}",
                    "平均尝试次数": str(data.get("attempts_avg", "—")),
                },
                "title": f"<b>{topic}</b><br>{strat_name}",
            })
            _add_edge({"from": proc_node_id, "to": snid, "label": "主题策略",
                       "arrows": "to", "dashes": True})

    return {"nodes": nodes, "edges": edges}