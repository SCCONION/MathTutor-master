from __future__ import annotations

from backend.agents import *
from backend.agents.nodes import *
from backend.agents.utils.helper import _parse_xml_response
from backend.agents.nodes.memory.memory_manager import (
    format_ltm_for_explainer,
    build_student_profile,
)
from backend.agents.nodes.tools.tools import web_search_tool as _ws_tool

tool_signals: list = []
 
def _web_search(query: str) -> str:
    """
    Call web_search_tool's underlying Tavily function directly.
    .func() bypasses LangChain ToolException wrapping for sync use inside a node.
    """
    try:
        result = _ws_tool.func(query)
        return result or ""
    except Exception as exc:
        logger.warning(f"[DirectResponse] web_search failed: {exc}")
        return ""
 
 
class DirectResponseAgent(BaseAgent):
    """
    Handles explain / hint / formula_lookup / research / generate intents.
    Uses a single LLM call that returns a markdown response inside <content> tags.
    """
 
    # ── Prompt builders ───────────────────────────────────────────────────────
 
    def _explain_prompt(self, problem_text: str, topic: str, ltm_hint: str) -> str:
        base = f"""你是一位资深数学老师。学生请求对某个概念进行讲解。
        请清晰、严谨地解释这个概念、定理或方法。

        回答结构（content 字段）：
        1. **核心思想** — 它是什么？为什么重要？（2-3 句话）
        2. **关键公式 / 定理** — 用 LaTeX（$...$）给出精确的数学表述
        3. **直观理解** — 用一个具体的类比或几何解释
        4. **适用场景** — 什么类型的题目会用到这个概念？
        5. **迷你示例** — 一个简短的演示计算
        6. **常见误区** — 学生常犯的 2-3 个错误

        所有数学表达式使用 $行内公式$ 和 $$块级公式$$。
        主题：{topic}
        问题：{problem_text}

        始终使用中文回答。"""

        if ltm_hint:
            base += f"\n\n个性化要求 — 请针对该学生的具体情况调整讲解：\n{ltm_hint}"
        return base

    def _profile_prompt(self, problem_text: str, profile: dict) -> str:
        """
        Prompt for meta-cognitive queries ("我有哪些知识点不会" / 我的学习情况).
        Feeds the student's FULL learning profile from Redis to the LLM so it
        can answer honestly — including saying "no records yet" when empty.
        """
        weak    = profile.get("weak_topics") or {}
        strong  = profile.get("strong_topics") or {}
        patterns = profile.get("mistake_patterns") or []
        history = profile.get("episodic_history") or []
        best    = profile.get("best_strategy")
        avg     = profile.get("avg_attempts")

        def _fmt_counter(d: dict) -> str:
            items = sorted(d.items(), key=lambda kv: -kv[1])
            return "、".join(f"{t}（{c} 次）" for t, c in items if c > 0) or "（暂无记录）"

        weak_str   = _fmt_counter(weak)
        strong_str = _fmt_counter(strong)

        pat_lines = [
            f"- {p.get('topic', '?')}：{p.get('pattern', '')}（{p.get('count', 1)} 次）"
            for p in patterns[:5]
        ]
        pat_str = "\n".join(pat_lines) if pat_lines else "（暂无记录）"

        hist_lines = []
        for h in history:
            oc = "、".join(f"{k}×{v}" for k, v in (h.get("outcomes") or {}).items())
            hist_lines.append(f"- {h['topic']}：共 {h['count']} 次（{oc}）")
        hist_str = "\n".join(hist_lines) if hist_lines else "（暂无记录）"

        strat_str = (
            f"{best}（平均 {avg:.1f} 次尝试）" if best and avg is not None
            else best or "（暂无记录）"
        )

        return f"""你是一位资深数学老师。学生想了解自己的学习情况——哪些知识点掌握得不好、哪些掌握得好。

以下是该学生的真实学习档案（来自长期记忆数据，请严格以此为准，不要编造）：

## 薄弱知识点（按出错次数排序）
{weak_str}

## 掌握较好的知识点
{strong_str}

## 常见错误模式
{pat_str}

## 历史学习记录（按主题聚合）
{hist_str}

## 最适合该学生的解题方法
{strat_str}

学生的问题：{problem_text}

回答要求：
1. 如实基于以上档案数据回答；档案中没有的内容不要说"有"。
2. 先直接回答学生的问题（哪些不会/掌握情况），再给出有针对性的学习建议。
3. 如果档案几乎为空（薄弱知识点和强项都没有记录），如实告诉学生：
   "目前你的学习记录还很少。建议多问我几道题，或者直接告诉我'我不会某个知识点'，
   我才能记住并帮你查漏补缺。"
4. 用清晰的列表或分点呈现，始终使用中文回答。"""

    def _hint_prompt(self, problem_text: str, topic: str) -> str:
        return f"""你是一位资深数学老师，正在给出苏格拉底式提示。
        学生想要一个提示方向 — 而不是完整解答。

        回答结构（content 字段）：
        1. **思路方向** — 正确的技巧或方法（1-2 句话，不要泄露答案）
        2. **关键洞察** — 学生需要注意到的一个关键点
        3. **第一步** — 只描述第一步该怎么做

        不要解完整道题。不要直接给出答案。要给予鼓励。
        主题：{topic}
        题目：{problem_text}

        始终使用中文回答。"""

    def _formula_prompt(self, problem_text: str, topic: str) -> str:
        return f"""你是一位资深数学老师。学生想要一个公式或定理的表述。

        回答结构（content 字段）：
        1. **公式** — 用 LaTeX（$$...$$）给出精确表述
        2. **变量说明** — 每个符号代表什么含义
        3. **适用条件** — 这个公式在什么情况下成立？
        4. **使用方法** — 在什么类型的题目中使用？
        5. **相关公式** — 1-2 个密切相关的结果

        主题：{topic}
        请求：{problem_text}

        始终使用中文回答。"""

    def _research_prompt(self, problem_text: str, topic: str, web_context: str) -> str:
        ctx_block = (
            f"\n\n联网搜索到的信息（作为主要参考来源）：\n{web_context}\n"
            if web_context
            else "\n\n（没有可用的联网结果 — 请使用你的知识回答。）\n"
        )
        return f"""你是一位资深数学教育专家，正在回答一个研究性或知识性问题。
        {ctx_block}
        回答结构（content 字段）：
        1. **概述** — 这是什么？（2-3 句话）
        2. **核心内容** — 实质性内容
        3. **重要性** — 相关性与实际应用
        4. **关联** — 与这个领域其他概念的关系
        5. **延伸阅读** — 1-2 个值得深入研究的主题

        数学表达式使用 $行内$ 或 $$块级$$ 格式。
        主题：{topic}
        问题：{problem_text}

        始终使用中文回答。"""

    def _generate_prompt(
        self,
        problem_text: str,
        topic:        str,
        difficulty:   str,
        web_context:  str,
    ) -> str:
        ctx_block = (
            f"\n\n联网上下文（用于参考当前的题型和风格）：\n{web_context}\n"
            if web_context else ""
        )
        return f"""你是一位资深数学教育专家，正在为学生生成练习题。
        请根据学生的请求生成高质量题目或示例。
        {ctx_block}
        content 字段的要求：
        - 难度：{difficulty}
        - 主题：{topic}
        - 每道题必须信息完整、可独立解答
        - 选择题提供 4 个选项（A/B/C/D），并在答案中标注正确答案
        - 方程式使用 $$块级公式$$
        - 最后附上一份简短的答案

        格式：
        **题目 1：**
        [题目内容]

        **题目 2：**
        [题目内容]

        ---
        **参考答案：**
        1. [答案]
        2. [答案]

        学生的请求：{problem_text}

        始终使用中文出题和解答。"""
    def direct_response_agent(self, state: AgentState) -> dict:
        try:
            parsed        = state.get("parsed_data") or {}
            problem_text  = parsed.get("problem_text") or ""
            topic         = parsed.get("topic") or "general"
 
            plan          = state.get("solution_plan") or {}
            intent_type   = plan.get("intent_type", "explain")
            difficulty    = plan.get("difficulty") or "medium"
            strategy      = plan.get("solver_strategy") or ""
 
            ltm_context   = state.get("ltm_context") or {}
            ltm_hint      = format_ltm_for_explainer(ltm_context, topic)

            tool_signals.clear()  # ← wipe signals from any previous call

            # ── 元认知查询：学生询问自己的学习档案 / 弱项知识点 ───────────────
            # "我有哪些知识点不会"这类问题不是讲解请求，而是查询长期记忆档案。
            # 走 _profile_prompt，把完整档案喂给 LLM 让其如实总结。
            _PROFILE_QUERY_SIGNALS = (
                "哪些知识点不会", "哪些不会", "不会什么", "不会的知识点",
                "哪些没学会", "哪里不会", "哪里没掌握", "掌握了什么",
                "弱项", "薄弱", "学习情况", "学习档案", "学得怎么样",
                "掌握情况", "我学过什么", "历史学习", "我的学习",
                "记不住", "不会的", "没掌握",
            )
            is_profile_query = (
                intent_type in ("explain", "research")
                and any(sig in problem_text for sig in _PROFILE_QUERY_SIGNALS)
            )
            profile = None
            if is_profile_query:
                profile = build_student_profile(
                    state.get("student_id") or "anonymous"
                )
                logger.info(
                    f"[DirectResponse] profile query detected — "
                    f"weak={len(profile.get('weak_topics') or {})} "
                    f"episodes={len(profile.get('episodic_history') or [])}"
                )
 
            # ── Web search for research / generate intents ────────────────────
            web_context  = ""
            search_query = ""
 
            if intent_type == "research":
                search_query = f"{topic} {problem_text[:100]}"
                logger.info(f"[DirectResponse] Tavily search (research): {search_query[:100]}")
                tool_signals.append({"name": "web_search_tool", "args": {"query": search_query}})
                web_context = _web_search(search_query)
                logger.info(f"[DirectResponse] Web context: {len(web_context)} chars")
 
            elif intent_type == "generate":
                search_query = f"{topic} {difficulty} practice problems {problem_text[:60]}"
                logger.info(f"[DirectResponse] Tavily search (generate): {search_query[:100]}")
                tool_signals.append({"name": "web_search_tool", "args": {"query": search_query}})
                web_context = _web_search(search_query)
                logger.info(f"[DirectResponse] Web context: {len(web_context)} chars")
 
            # ── Build prompt ──────────────────────────────────────────────────
            if is_profile_query:
                prompt = self._profile_prompt(problem_text, profile or {})
            elif intent_type == "explain":
                prompt = self._explain_prompt(problem_text, topic, ltm_hint)
            elif intent_type == "hint":
                prompt = self._hint_prompt(problem_text, topic)
            elif intent_type == "formula_lookup":
                prompt = self._formula_prompt(problem_text, topic)
            elif intent_type == "research":
                prompt = self._research_prompt(problem_text, topic, web_context)
            elif intent_type == "generate":
                prompt = self._generate_prompt(problem_text, topic, difficulty, web_context)
            else:
                logger.warning(f"[DirectResponse] Unknown intent '{intent_type}' — defaulting to explain")
                prompt = self._explain_prompt(problem_text, topic, ltm_hint)
 
            _SYSTEM = (
                "Respond using EXACTLY these XML tags — nothing outside them:\n"
                "<content>\n"
                "YOUR FULL MARKDOWN RESPONSE HERE\n"
                "</content>\n"
                "Rules:\n"
                "- Write all your explanation inside <content>...</content>.\n"
                "- Use markdown freely inside <content> (headers, bold, LaTeX $...$, lists).\n"
                "- Do NOT add any text before <content>."
            )

            raw_response = self.reserve_llm.invoke(
                [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
            )
            raw_text = (raw_response.content or "").strip()
            content = _parse_xml_response(raw_text)

            if not content:
                content = (
                    "I wasn't able to generate a response for your request. "
                    "Please try rephrasing."
                )

            intent_headers = {
                "explain":        "## 📖 Explanation",
                "hint":           "## 💡 Hint",
                "formula_lookup": "## 📐 Formula",
                "research":       "## 🔬 Research",
                "generate":       "## 📝 Practice Problems",
            }
            header   = intent_headers.get(intent_type, "## 📘 Response")
            final_md = f"{header}\n\n{content}"

 
            # ── Stubs so downstream nodes (store_ltm etc.) don't crash ─────────
            stub_solver = {
                "solution":         content,
                "final_answer":     "",
                "rag_context_used": False,
                "calculator_used":  False,
                "web_search_used":  bool(web_context),
            }
            stub_verifier = {
                "status":        "correct",
                "verdict":       f"Direct response — intent={intent_type}. No verification needed.",
                "suggested_fix": None,
                "confidence":    1.0,
                "hitl_reason":   None,
            }
 
            # ── Payload ───────────────────────────────────────────────────────
            payload(
                state, "direct_response_agent",
                summary=(
                    f"intent={intent_type} | topic={topic} | difficulty={difficulty} | "
                    f"web={'yes' if web_context else 'no'} | "
                    f"personalised={'yes' if ltm_hint else 'no'} | "
                    f"profile={'yes' if is_profile_query else 'no'}"
                ),
                fields={
                    "Intent":          intent_type,
                    "Topic":           topic,
                    "Difficulty":      difficulty,
                    "Strategy":        strategy[:100] if strategy else "n/a",
                    "Web search":      f"yes — {search_query[:80]}" if web_context else "no",
                    "Web result size": f"{len(web_context)} chars" if web_context else "0",
                    "LTM hint":        ltm_hint[:80] if ltm_hint else "none",
                    "Profile query":   "yes" if is_profile_query else "no",
                    "Response size":   f"{len(content)} chars",
                    "Preview":         content[:120],
                },
            )
            logger.info(
                f"[DirectResponse] done | intent={intent_type} topic={topic} "
                f"web={bool(web_context)} ltm={bool(ltm_hint)}"
            )
 
            return {
                "messages":           [HumanMessage(content=problem_text or prompt), AIMessage(content=content)],
                "solver_output":      stub_solver,
                "verifier_output":    stub_verifier,
                "safety_passed":      True,
                "explainer_output":   None,
                "final_response":     final_md,
                "conversation_log":   [final_md],
                "hitl_required":      True,
                "hitl_type":          "satisfaction",
                "hitl_reason":        "Response delivered — was this helpful?",
                "follow_up_question": None,
                "student_satisfied":  None,
                "agent_payload_log":          state.get("agent_payload_log") or [],
                "direct_response_tool_calls": tool_signals,
            }
 
        except Exception as e:
            logger.error(f"[DirectResponse] failed: {e}")
            raise Agent_Exception(e, sys)