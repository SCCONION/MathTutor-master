from __future__ import annotations
import json
from pathlib import Path

import streamlit as st
from frontend.pages import *

st.set_page_config(
    page_title="记忆图谱",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not st.user.is_logged_in:
    st.switch_page("app.py")

_HERE      = Path(__file__).parent        
_FRONTEND  = _HERE.parent                 
_GRAPH_DIR = _HERE                        
_TEMPLATES = _FRONTEND / "templates"      

try:
    _app_css = (_TEMPLATES / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{_app_css}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.markdown("""
<style>
section.main > div { padding-top: 0.5rem !important; }
[data-testid="stSidebar"] { background: #0c1020 !important; }
[data-testid="stSidebar"] * { font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", "JetBrains Mono", monospace; }
</style>
""", unsafe_allow_html=True)

try:
    import redis
    from backend.agents.graph import checkpointer
    from backend.agents.utils.db_utils import get_thread_history
    from backend.agents.utils.memory_graph_reader import build_graph_data
    _BACKEND_OK = True
except Exception as _import_err:
    _BACKEND_OK = False
    _import_err_msg = str(_import_err)

@st.cache_resource
def _get_redis():
    try:
        import redis as _redis
        from backend.agents.nodes.memory import REDIS_URL
        r = _redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        return r
    except Exception as e:
        st.warning(f"Redis 连接失败: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  HTML BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_legend_rows(visible_types: set[str]) -> str:
    rows = []
    for node_type, label in LEGEND_META:
        if node_type not in visible_types:
            continue
        color = NODE_COLORS[node_type]["border"]
        rows.append(
            f'<div class="row">'
            f'<div class="dot" style="background:{color};box-shadow:0 0 5px {color}88"></div>'
            f'<span>{label}</span>'
            f'</div>'
        )
    return "\n".join(rows)


def render_graph_html(
    graph_data: dict,
    layout_preset: str,
    show_labels: bool,
    show_edge_labels: bool,
) -> str:
    """
    Reads graph.html / graph.css / graph.js from disk, injects all tokens,
    returns the complete HTML string for st.components.v1.html().
    """
    try:
        html_tpl = (_GRAPH_DIR / "graph.html").read_text(encoding="utf-8")
        css_src  = (_GRAPH_DIR / "graph.css").read_text(encoding="utf-8")
        js_src   = (_GRAPH_DIR / "graph.js").read_text(encoding="utf-8")
    except FileNotFoundError as e:
        return f"<pre style='color:red'>Missing file: {e}</pre>"

    layout_opts = PHYSICS_PRESETS.get(layout_preset, PHYSICS_PRESETS["径向（默认）"])
    physics_on  = layout_opts["physics"].get("enabled", True)

    # Filter edge labels if disabled
    filtered_edges = []
    for e in graph_data.get("edges", []):
        fe = dict(e)
        if not show_edge_labels:
            fe["label"] = ""
        filtered_edges.append(fe)

    graph_json = json.dumps({
        "nodes": graph_data.get("nodes", []),
        "edges": filtered_edges,
    })

    visible_types = {n["type"] for n in graph_data.get("nodes", [])}

    replacements = {
        "%%VIS_CDN_JS%%":       VIS_CDN_JS,
        "%%VIS_CDN_CSS%%":      VIS_CDN_CSS,
        "%%INLINE_CSS%%":       css_src,
        "%%INLINE_JS%%":        js_src,
        "%%GRAPH_JSON%%":       graph_json,
        "%%LAYOUT_OPTIONS%%":   json.dumps(layout_opts),
        "%%PHYSICS_ENABLED%%":  str(physics_on).lower(),
        "%%PHYSICS_ACTIVE%%":   "active" if physics_on else "",
        "%%PHYSICS_LABEL%%":    "⚡ 物理引擎" if physics_on else "❄ 已冻结",
        "%%NODE_COLORS%%":      json.dumps(NODE_COLORS),
        "%%TYPE_BADGE_STYLE%%": json.dumps(TYPE_BADGE_STYLE),
        "%%NODE_SIZES%%":       json.dumps(NODE_SIZES),
        "%%NODE_SHAPES%%":      json.dumps(NODE_SHAPES),
        "%%NODE_FONT_SIZES%%":  json.dumps(NODE_FONT_SIZES),
        "%%EDGE_CONFIG%%":      json.dumps(EDGE_CONFIG),
        "%%LEGEND_ROWS%%":      _build_legend_rows(visible_types),
        "%%SHOW_LABELS%%":      str(show_labels).lower(),
    }

    result = html_tpl
    for token, value in replacements.items():
        result = result.replace(token, value)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def _load_graph(student_id: str, include_agents: bool, max_threads: int) -> dict:
    if not _BACKEND_OK:
        return {"nodes": [], "edges": []}
    rc = _get_redis()
    if rc is None:
        return {"nodes": [], "edges": []}
    return build_graph_data(
        student_id          = student_id,
        redis_client        = rc,
        checkpointer        = checkpointer,
        get_thread_history  = get_thread_history,
        max_threads         = max_threads,
        include_agent_nodes = include_agents,
    )


def _filter_graph(
    raw: dict,
    visible_types: list[str],
) -> dict:
    """Keep only nodes whose type is in visible_types; prune dangling edges."""
    type_set    = set(visible_types)
    nodes       = [n for n in raw.get("nodes", []) if n.get("type") in type_set]
    valid_ids   = {n["id"] for n in nodes}
    edges       = [
        e for e in raw.get("edges", [])
        if e.get("from") in valid_ids and e.get("to") in valid_ids
    ]
    return {"nodes": nodes, "edges": edges}


# ══════════════════════════════════════════════════════════════════════════════
#  TOP NAV BAR
# ══════════════════════════════════════════════════════════════════════════════

nav_left, nav_right = st.columns([8, 2], vertical_alignment="center")
with nav_left:
    st.markdown(
        "<h2 style='margin:0;color:#c8d8f0;font-family:\"Microsoft YaHei\",\"PingFang SC\",\"Noto Sans SC\",monospace;"
        "font-size:1.3rem;letter-spacing:0.04em;'>🧠 记忆图谱</h2>",
        unsafe_allow_html=True,
    )
with nav_right:
    if st.button("🧮 数学助手", use_container_width=True, help="返回数学辅导助手"):
        st.switch_page("app.py")   # relative to frontend/

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  — filters & controls
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='font-family:\"Microsoft YaHei\",\"PingFang SC\",\"Noto Sans SC\",monospace;font-size:0.78rem;"
        "color:#3b82f6;letter-spacing:0.1em;text-transform:uppercase;"
        "padding-bottom:6px;border-bottom:1px solid #1e2d45;margin-bottom:14px'>"
        "⬡ 图谱控制</div>",
        unsafe_allow_html=True,
    )

    # ── Layout preset ─────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#2d5070;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:4px'>布局方式</div>",
        unsafe_allow_html=True,
    )
    layout_choice = st.radio(
        "layout",
        list(PHYSICS_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Node type visibility ──────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#2d5070;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:6px'>可见节点类型</div>",
        unsafe_allow_html=True,
    )

    all_node_types = [t for t, _ in LEGEND_META]
    type_visibility: dict[str, bool] = {}

    for node_type, label in LEGEND_META:
        color = NODE_COLORS[node_type]["border"]
        col_dot, col_chk = st.columns([1, 5])
        with col_dot:
            st.markdown(
                f"<div style='width:10px;height:10px;border-radius:50%;"
                f"background:{color};box-shadow:0 0 5px {color}88;"
                f"margin-top:8px'></div>",
                unsafe_allow_html=True,
            )
        with col_chk:
            type_visibility[node_type] = st.checkbox(
                label,
                value=True,
                key=f"vis_{node_type}",
            )

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Display options ───────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#2d5070;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:6px'>显示选项</div>",
        unsafe_allow_html=True,
    )
    show_labels      = st.toggle("节点标签",      value=True)
    show_edge_labels = st.toggle("边标签",       value=False)
    include_agents   = st.toggle("代理/工具节点",  value=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Thread depth ──────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#2d5070;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:4px'>最大会话数</div>",
        unsafe_allow_html=True,
    )
    max_threads = st.slider("max_threads", 1, 30, 15, label_visibility="collapsed")

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Refresh ───────────────────────────────────────────────────────────────
    if st.button("🔄 刷新图谱", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ── Graph stats (populated after load) ───────────────────────────────────
    stats_ph = st.empty()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — load + render
# ══════════════════════════════════════════════════════════════════════════════

student_id = st.session_state.get("student_id")

if not student_id:
    st.warning("未找到有效会话，请返回数学辅导助手并登录。")
    st.stop()

if not _BACKEND_OK:
    st.error(f"后端不可用: {_import_err_msg}")
    st.stop()

with st.spinner("正在加载记忆图谱…"):
    raw_graph = _load_graph(student_id, include_agents, max_threads)

# Apply visibility filters
visible_types = [t for t, checked in type_visibility.items() if checked]
graph_data    = _filter_graph(raw_graph, visible_types)

n_nodes = len(graph_data["nodes"])
n_edges = len(graph_data["edges"])

# ── Sidebar stats update ──────────────────────────────────────────────────────
with stats_ph:
    st.markdown(
        f"<div style='background:#0d1424;border:1px solid #1e2d45;border-radius:8px;"
        f"padding:10px 12px;font-family:\"Microsoft YaHei\",\"PingFang SC\",\"Noto Sans SC\",monospace;font-size:0.72rem;"
        f"color:#4a6080;margin-top:4px'>"
        f"<div style='color:#3b82f6;margin-bottom:5px;font-size:0.65rem;"
        f"text-transform:uppercase;letter-spacing:0.1em'>图谱统计</div>"
        f"<div>节点 &nbsp;<span style='color:#7eb8f7'>{n_nodes}</span></div>"
        f"<div>边 &nbsp;&nbsp;&nbsp;<span style='color:#7eb8f7'>{n_edges}</span></div>"
        f"<div>会话 <span style='color:#7eb8f7'>"
        f"{sum(1 for n in graph_data['nodes'] if n.get('type')=='session')}"
        f"</span></div>"
        f"<div>长期记忆 &nbsp;<span style='color:#7eb8f7'>"
        f"{sum(1 for n in graph_data['nodes'] if n.get('type') in ('episodic','semantic','procedural'))}"
        f"</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Empty state ───────────────────────────────────────────────────────────────
if n_nodes == 0:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:400px;flex-direction:column;gap:12px;color:#2d4060'>"
        "<div style='font-size:3rem'>🧠</div>"
        "<div style='font-family:\"Microsoft YaHei\",\"PingFang SC\",\"Noto Sans SC\",monospace;font-size:0.85rem'>"
        "暂无记忆数据，先解几道题吧！</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Render graph ──────────────────────────────────────────────────────────────
import streamlit.components.v1 as components

graph_html = render_graph_html(
    graph_data       = graph_data,
    layout_preset    = layout_choice,
    show_labels      = show_labels,
    show_edge_labels = show_edge_labels,
)

components.html(graph_html, height=720, scrolling=False)

# ── Keyboard shortcut hint ────────────────────────────────────────────────────
st.markdown(
    "<div style='font-family:\"Microsoft YaHei\",\"PingFang SC\",\"Noto Sans SC\",monospace;font-size:0.68rem;"
    "color:#1e3a5a;text-align:center;margin-top:4px'>"
    "⌨ &nbsp;F = 适应窗口 &nbsp;· &nbsp;L = 显示/隐藏标签 &nbsp;· &nbsp;"
    "Esc = 关闭面板 &nbsp;· &nbsp;双击 = 展开相邻节点"
    "</div>",
    unsafe_allow_html=True,
)