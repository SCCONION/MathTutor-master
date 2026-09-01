

> 基于 LangGraph 多智能体架构的数学辅导系统：支持文本 / 图片（OCR）/ 语音（ASR）三种输入方式，具备混合检索（Hybrid CRAG）、三层长期记忆（Episodic / Semantic / Procedural）、自纠错闭环与人工介入（HITL）能力。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 技术栈](#2-技术栈)
- [3. 功能特性](#3-功能特性)
- [4. 系统架构](#4-系统架构)
- [5. 环境要求](#5-环境要求)
- [6. 快速开始](#6-快速开始)
- [7. 环境变量配置](#7-环境变量配置)
- [8. 使用指南](#8-使用指南)
- [9. 常见问题排查](#9-常见问题排查)
- [10. 自动化测试](#10-自动化测试)
- [11. 项目结构](#11-项目结构)
- [12. 部署](#12-部署)

---

## 1. 项目简介

Math Tutor 是一个**面向数学学习场景的垂直领域 AI Agent 系统**，能够：

- 接收 **文本 / 图片（OCR）/ 语音（ASR）** 三种形式的学生提问
- 自动判断学生意图（解题 / 讲解 / 提示 / 公式查询 / 研究 / 出题）
- 结合学生上传的 **PDF 学习资料**（RAG 检索）和**长期记忆**给出个性化解答
- 通过**独立校验器**验证解答正确性，支持最多 **3 轮自动修正**，超限转人工
- **跨会话记住学生**：历史题目、知识薄弱点、有效解题策略
- 提供**记忆图谱可视化**页面，直观查看学生学习画像

---

## 2. 技术栈

| 分类 | 技术 |
|---|---|
| Agent 编排 | LangGraph（StateGraph / Checkpoint / ToolNode）、LangChain |
| LLM | DeepSeek（DeepSeek-chat，通过 Groq SDK 兼容层调用） |
| 向量检索 | BGE-large-zh-v1.5（Embedding）、FAISS（稠密检索）、BM25（稀疏检索）、RRF 融合 |
| 长期记忆 | Redis + RedisJSON + RedisVL（HNSW 向量索引） |
| 工具 | SymPy（符号计算）、Tavily MCP（联网搜索）、PyPDF（PDF 解析） |
| OCR / ASR | Google Cloud Vision（图片识别）、Groq Whisper（语音转写） |
| 前端 | Streamlit（Web UI） |
| 认证 | Google OAuth（st.login 原生登录） |
| 测试 | pytest（78 个自动化测试）+ GitHub Actions CI |

---

## 3. 功能特性

### 3.1 多模态输入
- 📝 **文本提问**：直接输入数学问题
- 📸 **图片提问**：上传题目照片，自动 OCR 识别
- 🎤 **语音提问**：录制语音，自动转写为文字

### 3.2 六类意图路由
| 意图 | 说明 | 处理路径 |
|---|---|---|
| `solve` 解题 | 求具体数值/代数题的完整解答 | 完整解题管线（验证+重试） |
| `explain` 讲解 | 概念、定理、方法的讲解 | 直接回答 |
| `hint` 提示 | 只想要解题思路提示 | 解题管线（提示专用） |
| `formula_lookup` 公式查询 | 公式/定理表述 | 解题管线（公式专用） |
| `research` 研究 | 最新进展、历史、一般数学知识 | 直接回答 + 联网搜索 |
| `generate` 出题 | 练习题目生成 | 直接回答 + 联网搜索 |

### 3.3 自纠错闭环
```
solver（解题）→ verifier（校验）→ 正确 → 放行
                    └─ 错误 → 携带反馈打回 solver（最多 3 次）→ 仍错 → 人工介入
```

### 3.4 三层长期记忆
| 层 | 内容 | 生命周期 |
|---|---|---|
| Episodic（情景） | 每道做过的题目、答案、对错 | 90 天 TTL + 遗忘衰减 |
| Semantic（语义） | 知识薄弱点、掌握较好的知识点、错误模式 | 永久 |
| Procedural（程序） | 对特定学生有效的解题策略 | 永久 |

### 3.5 记忆图谱可视化
侧边栏提供"记忆图谱"页面，以节点图形式展示学生的历史题目、弱项主题、解题策略之间的关联。

---

## 4. 系统架构

```
用户（文本 / 图片 / 音频）
   │
   ▼
Streamlit 前端（app.py）── Google OAuth 登录 ──► 映射 student_id
   │
   ▼
LangGraph 多智能体图（15 个节点）
   │
   ├─ detect_input ──► ocr / asr / guardrail
   ├─ guardrail ──► parser ──► retrieve_ltm（查记忆）──► intent_router
   ├─ intent_router ──► solver 管线 或 direct_response
   │     ├─ solver ──► tool_node（RAG / 计算器 / 搜索）◄──┐（ReAct 循环）
   │     └─ verifier ──► 正确 / 错误重试 / 人工 ─────────────┘
   ├─ safety ──► explainer ──► hitl（满意度确认）
   └─ store_ltm（写入三层记忆）──► END
```

---

## 5. 环境要求

| 依赖 | 说明 |
|---|---|
| Python | 3.11+ |
| Docker Desktop | 用于运行 Redis Stack（或本机已有 Redis） |
| 互联网 | 调用 LLM / Embedding / 搜索 API 需要 |

### 需要的 API 密钥
- **DeepSeek API Key**（LLM 推理，建议 ×2 避免限流）
- **Cohere API Key**（Embedding，备用方案）
- **Tavily API Key**（联网搜索，MCP 接入）
- **Google Cloud**（Vision OCR + OAuth 凭据）

---

## 6. 快速开始

### 6.1 克隆项目

```bash
git clone <仓库地址>
cd MathTutor-master
```

### 6.2 创建虚拟环境并安装依赖

**Windows（PowerShell）：**

```powershell
python -m venv myenv
myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux：**

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 6.3 启动 Redis Stack

```bash
docker compose up -d redis
```

验证 Redis 可用：

```powershell
docker ps   # 应看到 jee_redis 容器状态为 healthy
```

### 6.4 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填入真实密钥（详见 [第 7 节](#7-环境变量配置)）。

### 6.5 配置 Streamlit Secrets

创建 `.streamlit/secrets.toml`（内容见第 7 节）。

### 6.6 启动应用

**方式 A — Windows PowerShell 脚本（推荐）：**

```powershell
.\run.ps1
```

> ⚠️ 若提示"未对文件进行数字签名"（执行策略限制），先执行：
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

**方式 B — 手动启动（Windows）：**

```powershell
$env:PYTHONPATH = "$PSScriptRoot\src"
$env:HF_HUB_OFFLINE = "1"          # 离线加载 BGE 模型（已本地缓存），避免联网超时
$env:TRANSFORMERS_OFFLINE = "1"
streamlit run src\frontend\app.py
```

**方式 C — entrypoint 脚本（Linux/macOS）：**

```bash
chmod +x entrypoint.sh
./entrypoint.sh
```

### 6.7 访问应用

浏览器打开 **http://localhost:8501**，使用 Google 账号登录后即可使用。

---

## 7. 环境变量配置

### 7.1 `.env` 文件

```env
# LLM 推理
GROQ_API_KEY=gsk_...           # 主 Key — guardrail/parser/router/verifier/safety/explainer
GROQ_API_KEY_2=gsk_...         # 副 Key — solver + direct_response（避免限流）

# Embedding
COHERE_API_KEY=CIy...

# 联网搜索
TAVILY_API_KEY=tvly-...

# Redis
REDIS_URL=redis://:jee_secret@localhost:6379

# Google OAuth
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
OAUTH_REDIRECT_URI=http://localhost:8501/oauth2callback

# Google Vision OCR（二选一）
GOOGLE_CREDENTIALS_JSON='{"type":"service_account",...}'   # JSON 字符串（云端）
# GOOGLE_APPLICATION_CREDENTIALS=./secrets/your-key.json   # 文件路径（本地）
```

### 7.2 `.streamlit/secrets.toml` 文件

```toml
# API 密钥（与 .env 对应，供 Streamlit Cloud 使用）
GROQ_API_KEY = "gsk_..."
GROQ_API_KEY_2 = "gsk_..."
COHERE_API_KEY = "CIy..."
TAVILY_API_KEY = "tvly-..."
REDIS_URL = "redis://:jee_secret@localhost:6379"

# Google OAuth — 必须包含 server_metadata_url
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "你的随机cookie密钥"

[auth.google]
client_id = "你的-google-client-id.apps.googleusercontent.com"
client_secret = "GOCSPX-..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

> 💡 生成 `cookie_secret`：
> ```bash
> python -c "import secrets; print(secrets.token_hex(32))"
> ```

---

## 8. 使用指南

### 8.1 登录

使用 Google 账号通过 OAuth 登录。登录后系统会将你的邮箱映射为稳定的 `student_id`，用于隔离长期记忆（每个学生独立记忆空间）。

### 8.2 提问

在聊天输入框输入问题，支持三种方式：

- **文本**：直接输入，如 `帮我解 x² - 5x + 6 = 0`
- **图片**：点击上传按钮，选择题目截图（支持 png/jpg/jpeg），自动 OCR 识别
- **语音**：录制音频（支持 wav/mp3/m4a），自动转写

### 8.3 上传 PDF 学习资料

在侧边栏上传 PDF 笔记/教材，之后提问时系统会自动调用 `rag_tool` 在你的资料中检索相关内容，结合资料给出解答。

> **建议**：上传的 PDF 应包含公式、定理、例题，检索效果更佳。

### 8.4 交互式对话

- **澄清**：当题目信息不完整时，系统会主动提问澄清
- **满意度确认**：每次解答后系统会询问"讲明白了吗"，可选择：
  - ✅ 满意 → 记录学习记忆
  - 🔄 重新讲解 → 系统换一种方式重新讲解
- **人工介入（HITL）**：当自动校验无法判定、或多次尝试仍错误时，会转入人工确认流程

### 8.5 查看学习记忆

侧边栏 → **记忆图谱** 页面：

- 查看所有历史题目（Episodic）
- 查看知识薄弱点与强项（Semantic）
- 查看有效解题策略（Procedural）
- 节点图可视化展示知识点关联

### 8.6 询问学习情况

你可以直接问系统：
- `我有哪些知识点不会？`
- `我的薄弱环节是什么？`
- `我学得怎么样？`

系统会基于长期记忆档案如实回答，并给出针对性学习建议。

> ⚠️ **提示**：为了让系统记住你的薄弱点，可以直接说"我不会 XX 知识点"，系统会自动记录到你的学习档案中。

---

## 9. 常见问题排查

### 9.1 Streamlit 启动报 `st.user.is_logged_in` 错误

**原因**：8501 端口有旧进程残留，使用了旧版本配置。

**解决**：

```powershell
Get-NetTCPConnection -LocalPort 8501 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

然后重新启动。

### 9.2 报 `WinError 10060`（HuggingFace 连接超时）

**原因**：启动时尝试联网下载 BGE 模型，但网络无法访问 huggingface.co。

**解决**：设置离线环境变量（模型已本地缓存）：

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

（`run.ps1` 已内置这两个变量。）

### 9.3 Redis 连接失败

**检查**：

```powershell
docker ps                    # 确认 jee_redis 容器在运行
Get-NetTCPConnection -LocalPort 6379 -State Listen   # 确认 6379 端口监听
```

若未启动：`docker compose up -d redis`

### 9.4 页面样式异常（英文/无样式）

通常是页面运行时出错导致 CSS 未注入。先查看终端日志定位报错，修复后**必须重启 Streamlit**（Streamlit 不会热重载已修改的文件）。

### 9.5 修改代码后不生效

Streamlit 的 `config.toml` 设置了 `fileWatcherType = "none"`（关闭文件监听）。**修改代码后需手动重启**。

---

## 10. 自动化测试

### 运行全部测试

```bash
pip install -r requirements-test.txt
pytest
```

### 测试结构

```
src/tests/
├── conftest.py              # 测试环境隔离（stub 外部依赖：LLM/Redis/Google）
├── unit/                    # 单元测试（50+ 个）
│   ├── test_router_node_contract.py      # 路由契约
│   ├── test_memory_manager_flow_gates.py # 记忆写入门控
│   ├── test_state_and_policies.py        # 自纠错重试上限
│   ├── test_direct_response_agent.py     # 直接回答节点
│   ├── test_tavily_mcp_helpers.py        # MCP 客户端解析
│   └── ...
└── integration/             # 集成测试（整链路）
    ├── test_clarification_to_router_loop.py
    ├── test_memory_store_after_correct_solve.py
    └── ...
```

> 💡 测试通过 stub 替换外部依赖（假 LLM 返回固定结果、假 Redis 计数），**不产生任何 API 费用**，毫秒级完成。CI（GitHub Actions）会在每次 push 自动运行。

---

## 11. 项目结构

```
MathTutor-master/
├── src/
│   ├── backend/
│   │   ├── agents/
│   │   │   ├── graph.py                 # LangGraph 工作流（15 节点编排）
│   │   │   ├── state.py                 # AgentState 定义
│   │   │   ├── nodes/
│   │   │   │   ├── input.py             # 输入检测（text/image/audio）
│   │   │   │   ├── guardrail.py         # 输入安全审查
│   │   │   │   ├── parser.py            # 题意解析
│   │   │   │   ├── router.py            # 意图路由
│   │   │   │   ├── solver.py            # 解题（ReAct 循环）
│   │   │   │   ├── verifier.py          # 答案校验
│   │   │   │   ├── safety.py            # 输出安全审查
│   │   │   │   ├── explainer.py         # 个性化讲解
│   │   │   │   ├── direct_response.py   # 直接回答（讲解/研究/出题）
│   │   │   │   ├── hitl.py              # 人工介入
│   │   │   │   ├── memory/
│   │   │   │   │   ├── memory_manager.py # 三层记忆读写 + 档案查询
│   │   │   │   │   └── __init__.py       # 索引 schema 配置
│   │   │   │   └── tools/
│   │   │   │       ├── tools.py          # RAG / 计算器 / 搜索工具
│   │   │   │       └── mcp/              # Tavily MCP 客户端
│   │   │   └── utils/
│   │   │       ├── db_utils.py           # Redis 客户端 / 用户注册
│   │   │       └── memory_graph_reader.py # 记忆图谱数据读取
│   │   ├── logger/                       # 日志模块
│   │   └── exceptions/                   # 异常定义
│   ├── frontend/
│   │   ├── app.py                        # Streamlit 主应用
│   │   ├── pages/
│   │   │   ├── memory_viz.py             # 记忆图谱页面
│   │   │   └── graph.*                   # 图谱前端资源
│   │   └── templates/                    # 登录/个人资料/样式
│   └── tests/                            # 自动化测试
├── docs/                                 # 架构文档
├── docker-compose.yml                    # Redis Stack 编排
├── run.ps1                               # Windows 启动脚本
├── entrypoint.sh                         # Linux 启动脚本
└── requirements.txt                      # 依赖清单
```

---

## 12. 部署

### 12.1 Docker 部署（Redis）

```bash
docker compose up -d redis
```

### 12.2 Streamlit Cloud 部署

1. 将项目推送到 GitHub 仓库
2. 进入 [share.streamlit.io](https://share.streamlit.io) → New app
3. 选择仓库与主文件 `src/frontend/app.py`
4. 在 Settings → Secrets 中填入 `secrets.toml` 内容
5. 将 OAuth `redirect_uri` 更新为 `https://你的应用.streamlit.app/oauth2callback`
6. Redis 需使用云托管的 Redis（如 Redis Cloud / Upstash），并更新 `REDIS_URL`

### 12.3 运维脚本

**清理过期记忆**（decay < 0.05 且 30 天以上未访问）：

```python
from backend.agents.nodes.memory.memory_manager import prune_stale_episodic
prune_stale_episodic()                    # 全部学生
prune_stale_episodic("student_id")        # 指定学生
```

---

## 附：常见问题速查

| 问题 | 一句话解决 |
|---|---|
| 启动报错 st.user.is_logged_in | 杀 8501 端口旧进程后重启 |
| HuggingFace 超时 | 设 `HF_HUB_OFFLINE=1` |
| Redis 连不上 | `docker compose up -d redis` |
| 修改代码不生效 | 手动重启 Streamlit（已关闭热重载） |
| 想不起学生弱项 | 问"我有哪些知识点不会"，系统读档案回答 |
| 日志在哪 | 项目根 `logs/` 目录（时间戳命名） |

---

*文档版本：v1.0（2026-08）*
