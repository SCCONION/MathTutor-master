import streamlit as st
from pathlib import Path
from typing import Optional

__all__=["st", "Path", "Optional"]

# ── Node metadata ─────────────────────────────────────────────────────────────
AGENT_META: dict[str, dict] = {
    "detect_input":          {"icon": "🔍", "label": "检测输入类型"},
    "ocr_node":              {"icon": "📸", "label": "OCR 识别（图片→文字）"},
    "asr_node":              {"icon": "🎤", "label": "ASR 识别（语音→文字）"},
    "guardrail_agent":       {"icon": "🛡️", "label": "安全审查"},
    "retrieve_ltm":          {"icon": "🧠", "label": "检索长期记忆"},
    "parser_agent":          {"icon": "🧩", "label": "题目解析"},
    "intent_router":         {"icon": "🗺️",  "label": "意图识别"},
    "solver_agent":          {"icon": "🧮", "label": "解题代理（ReAct）"},
    "tool_node":             {"icon": "🔧", "label": "工具执行器"},
    "verifier_agent":        {"icon": "✅", "label": "答案校验"},
    "safety_agent":          {"icon": "🔒", "label": "安全过滤器"},
    "explainer_agent":       {"icon": "📚", "label": "讲解代理"},
    "direct_response_node":  {"icon": "💬", "label": "直接回答代理"},
    "hitl_node":             {"icon": "🙋", "label": "等待人工输入"},
    "store_ltm":             {"icon": "💾", "label": "保存长期记忆"},
    "rag_tool":              {"icon": "📄", "label": "RAG — PDF 检索"},
    "web_search_tool":       {"icon": "🌐", "label": "联网搜索（Tavily）"},
    "calculator_tool":       {"icon": "🔢", "label": "符号计算器"},
}

# ── Tool metadata ─────────────────────────────────────────────────────────────
TOOL_META: dict[str, dict] = {
    "rag_tool":        {"icon": "📄", "label": "RAG — PDF 检索"},
    "web_search_tool": {"icon": "🌐", "label": "联网搜索（Tavily）"},
    "calculator_tool": {"icon": "🔢", "label": "符号计算器"},
}

# ── Answer node filter ────────────────────────────────────────────────────────
# Both explainer_agent and direct_response_node produce final_response
ANSWER_NODES: set[str] = {"explainer_agent", "direct_response_node"}

# ── HITL banner prefixes ──────────────────────────────────────────────────────
HITL_PREFIX     = "__HITL__:"
HITL_SAT_PREFIX = "__SATQ__:"