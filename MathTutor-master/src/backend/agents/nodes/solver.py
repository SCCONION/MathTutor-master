from backend.agents import *
from backend.agents.nodes import *
from backend.agents.nodes.tools.tools import has_store, rag_tool as _rag_tool_base
from backend.agents.nodes.memory.memory_manager import (
    trim_messages_if_needed,
    format_ltm_for_solver,
)
from langchain_core.tools import tool as lc_tool

_SCOPED_RAG_CACHE: dict = {}

def _make_scoped_rag(thread_id: str):
    # just scoping it to provide the right index to our rag_tool with the correct thread_id
    if thread_id in _SCOPED_RAG_CACHE:
        return _SCOPED_RAG_CACHE[thread_id]
    @lc_tool
    def rag_tool(query: str) -> str:
        """
        在已上传的 PDF 文档中检索与查询相关的段落。

        强制要求：当存在 PDF 时，每个新问题都必须首先调用此工具。

        关键查询规则：
        查询必须针对概念 / 主题 / 定理 / 公式的相似性，而不是题目文本的相似性。

        目标：即使题目措辞与学生笔记完全不同，也要检索到学生写下的确切公式或讲解。

        示例：
          题目："一个袋子里有 3 个红球和 2 个蓝球，取出两个球..."
          好的查询 → "贝叶斯定理"
          好的查询 → "贝叶斯定理公式"
          好的查询 → "条件概率"

          题目："求 ∫ x² sin(x) dx"
          好的查询 → "分部积分公式"
          好的查询 → "分部积分"

        不好的查询（不要使用）：
          - 完整题目原文
          - "求两个球都是红球的概率"
          - 任何包含具体数字或题目原文的内容

        保持查询简短（最多 3-8 个词），只关注数学概念。
        如果工具返回"未找到相关段落"，不要再调用它。
        """
        if not has_store(thread_id):
            return (
                "CRAG: 当前会话没有索引任何文档。"
                "不要再调用 rag_tool — 改用 web_search_tool。"
            )
        return _rag_tool_base.invoke({"query": query, "thread_id": thread_id})

    _SCOPED_RAG_CACHE[thread_id] = rag_tool
    return rag_tool

_TOOLS_NO_RAG = [calculator_tool, web_search_tool]


class SolverAgent(BaseAgent):

    _SYSTEM_PROMPT = """\
        你是一名专业的中文数学辅导老师。

        你的任务是帮助学生解决数学问题。

        回答要求：

        1. 始终使用中文回答。
        2. 解题过程必须清晰分步骤。
        3. 不只给答案，要解释每一步为什么这样做。
        4. 根据学生水平调整讲解深度。
        5. 如果学生容易犯错，需要指出原因。

        解题格式：

        已知：
        列出题目中的条件。

        求：
        说明要求什么。

        解：

        步骤1：
        说明使用的方法，并展示计算过程。

        步骤2：
        继续推导。

        最终答案：
        给出结果。

        注意：
        - 数学公式保持规范 LaTeX 格式。
        - 不要输出英文解释。

        严格工具规则：
        每次回复必须要么是结构化的工具调用，要么是完整的推导过程。
        绝不能同时包含两者。绝不要在文本中写 <function=...> 标签。

        工具
        {tool_guide}

        解答格式：
        已知：[用题目中确切的变量名重述已知条件]
        求：[要求解什么]

        步骤 1 — [标题，例如"分部积分法"]：
            [逐行推导，每行以新表达式结尾]
            ∴  [该步骤的结果]

        步骤 N — ...

        ∴ 最终答案：[精确结果 — 与推导符号一致 — 需要时带单位]

        策略：{strategy}
        第 {attempt} 次尝试，共 {max_attempts} 次。{feedback_block}{ltm_block}"""

    _TOOL_GUIDE_WITH_RAG = """\
        强制第一步：先调用 rag_tool，再开始写解答。

        查询策略（关键）：
        你不是在搜索与当前题目相似的题目。
        你是在搜索学生笔记中写下的确切概念 / 定理 / 公式 / 方法。

        构造查询的步骤：
        1. 阅读题目，找出解题所需的一个核心数学思想。
        2. 用简短的概念名称描述它（定理名、公式名、方法名）。
        3. 用它作为查询词。

        好的例子：
          - "贝叶斯定理"
          - "贝叶斯定理公式"
          - "条件概率公式"
          - "分部积分公式"
          - "洛必达法则"
          - "矩阵对角化方法"

        不好的例子（绝对不要这样做）：
          - 完整的题目原文
          - 任何包含数字或题目特定措辞的句子

        这样可以保证：即使学生的笔记中只有纯公式（如"P(A|B) = ..."）而没有例题，
        也能被检索到。

        用这个简短的概念查询词调用 rag_tool，只调用一次。
        收到结果后，结合返回的段落和自己的知识写出完整解答。不要再调用 rag_tool。"""

    _TOOL_GUIDE_NO_RAG = """\
        - symbolic_calculator 符号计算器 → 仅用于大阶乘、高精度小数、
                                或大型矩阵运算。不要用于基础概率、
                                简单积分或常规代数。
        - web_search_tool 联网搜索 → 需要查阅的公式或理论。"""

    # Removing rag tool from second iteration reasoning
    _TOOL_GUIDE_RETRY = """\
        RAG 已在上一次尝试中使用过 — 不要再调用 rag_tool。
        直接根据上面的校验反馈修正你的方法。
        - symbolic_calculator 符号计算器 → 仅用于大阶乘、高精度小数、
                                或大型矩阵运算。不要用于基础概率、
                                简单积分或常规代数。
        - web_search_tool 联网搜索 → 需要查阅的公式或理论。"""

    def _build_system(
        self,
        strategy:      str,
        attempt:       int,
        max_attempts:  int,
        rag_available: bool,
        is_retry:      bool,
        feedback:      str,
        ltm_hint:      str,
    ) -> SystemMessage:
        feedback_block = (
            f"\n\n上一次尝试不正确。\n校验反馈：{feedback}"
            if feedback else ""
        )
        ltm_block = (
            f"\n\n学生背景（来自过往会话）：\n{ltm_hint}"
            if ltm_hint else ""
        )

        if is_retry:
            tool_guide = self._TOOL_GUIDE_RETRY
        elif rag_available:
            tool_guide = self._TOOL_GUIDE_WITH_RAG
        else:
            tool_guide = self._TOOL_GUIDE_NO_RAG

        return SystemMessage(content=self._SYSTEM_PROMPT.format(
            tool_guide     = tool_guide,
            strategy       = strategy or "choose the most direct method",
            attempt        = attempt,
            max_attempts   = max_attempts,
            feedback_block = feedback_block,
            ltm_block      = ltm_block,
        ))

    def _bind_tools(self, rag_available: bool, thread_id: str,
                    rag_already_called: bool = False, is_retry: bool = False):
        """
        Tool binding per phase:

        - Retry (iteration > 0)            : [calc, web], tool_choice="auto"
                                             RAG context already injected as HumanMessage;
                                             LLM can use calc/web or just reason directly.
        - No PDF uploaded                  : [calc, web], tool_choice="auto"
        - Attempt 1, RAG not yet called    : [rag, calc, web], tool_choice={"type":"function","function":{"name":"rag_tool"}}
                                             Forces the FIRST call to be rag_tool specifically.
        - Attempt 1, RAG already returned  : [calc, web], tool_choice="auto"
                                             RAG is done; LLM writes solution, optionally
                                             using calc or web if it needs to.
        """
        if is_retry or not rag_available:
            return self.reserve_llm.bind_tools(_TOOLS_NO_RAG, tool_choice="auto")

        scoped_rag = _make_scoped_rag(thread_id)

        if rag_already_called:
            # RAG returned — drop it from the tool list, free the LLM to write
            return self.reserve_llm.bind_tools(_TOOLS_NO_RAG, tool_choice="auto")

        # First entry, RAG not yet called — force rag_tool as the next action
        return self.reserve_llm.bind_tools(
            [scoped_rag, calculator_tool, web_search_tool],
            tool_choice={"type": "function", "function": {"name": "rag_tool"}},
        )

    def _extract_final_answer(self, text: str) -> str:
        for marker in ("∴ Final Answer:", "Final Answer:", "FINAL ANSWER:"):
            if marker in text:
                return text.split(marker)[-1].strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else text

    def solver_agent(self, state: AgentState) -> dict:
        try:
            parsed        = state.get("parsed_data") or {}
            problem_text  = parsed.get("problem_text") or ""
            plan          = state.get("solution_plan") or {}
            strategy      = plan.get("solver_strategy", "")
            intent_type   = plan.get("intent_type", "solve")
            difficulty    = plan.get("difficulty", "medium")
            topic         = parsed.get("topic") or ""
            iteration     = state.get("solve_iterations", 0)
            thread_id     = state.get("thread_id") or ""
            human_fb      = state.get("human_feedback") or ""
            prev_verifier = state.get("verifier_output") or {}
            ltm_context   = state.get("ltm_context") or {}
            existing_msgs = state.get("messages") or []

            if state.get("solve_iterations", 0) == 0 and not any(
                isinstance(m, ToolMessage) for m in existing_msgs
            ):
                existing_msgs = []

            existing_msgs = trim_messages_if_needed(   # trimming
                messages  = list(existing_msgs),   
                thread_id = thread_id,
                llm       = self.reserve_llm,
            )

            ltm_hint = format_ltm_for_solver(ltm_context, topic) if ltm_context else ""

            feedback = prev_verifier.get("suggested_fix") or ""
            if human_fb and not existing_msgs:
                feedback = f"{feedback}\nHuman feedback: {human_fb}".strip()

            rag_available = has_store(thread_id)
            is_retry = iteration > 0

            _RAG_SENTINEL = "[RAG context retrieved from student's notes]"
            rag_already_called = any(
                (isinstance(m, ToolMessage) and "rag" in (m.name or "").lower())
                or (isinstance(m, HumanMessage) and _RAG_SENTINEL in (m.content or ""))
                for m in existing_msgs
            )

            # Tracks whether RAG was executed inline THIS invocation.
            # Needed because inline RAG never writes a ToolMessage to history,
            # so all_msgs inspection alone would miss it and report rag_used=False.
            _rag_ran_inline = False
            rag_query  = ""   
            rag_result = "" 

            system = self._build_system(
                strategy      = strategy,
                attempt       = iteration + 1,
                max_attempts  = 3,
                rag_available = rag_available,
                is_retry      = is_retry,
                feedback      = feedback,
                ltm_hint      = ltm_hint,
            )

            if not existing_msgs:
                messages = [
                    system,
                    HumanMessage(content=f"Solve this problem:\n\n{problem_text}"),
                ]
            else:
                messages = (
                    [system] + list(existing_msgs[1:])
                    if isinstance(existing_msgs[0], SystemMessage)
                    else [system] + list(existing_msgs)
                )

            response = self._bind_tools(
                rag_available, thread_id, rag_already_called, is_retry
            ).invoke(messages)

            updates: dict = {"messages": [response]}

            if getattr(response, "tool_calls", None):
                tool_names = [tc["name"] for tc in response.tool_calls]
                logger.info(f"[Solver] Tool calls: {tool_names}")

                has_only_rag = all("rag" in n.lower() for n in tool_names)
                if not has_only_rag:
                    return {
                        **updates,
                        "agent_payload_log": state.get("agent_payload_log") or [],
                    }

                # ---- RAG was called: execute it directly here ---------------
                rag_call        = response.tool_calls[0]
                rag_query       = rag_call["args"].get("query", "")
                rag_result      = _make_scoped_rag(thread_id).invoke({"query": rag_query})
                _rag_ran_inline = True  # Bug 2 fix: flag so rag_used is correct below

                logger.info(f"[Solver] RAG executed inline for query: '{rag_query}'")
               
                rag_context_block = (    
                    f"[RAG context retrieved from student's notes]\n{rag_result}\n"
                    f"[End of RAG context]\n\n"
                    f"Now write the full solution to the problem using the context above "
                    f"plus your own knowledge."
                )

                messages_for_call2 = [
                    system,
                    HumanMessage(content=f"Solve this problem:\n\n{problem_text}"),
                    HumanMessage(content=rag_context_block),
                ]

                response2 = self._bind_tools(
                    rag_available=False,   # forces _TOOLS_NO_RAG regardless of PDF
                    thread_id=thread_id,
                    rag_already_called=True,
                    is_retry=False,
                ).invoke(messages_for_call2)

                if getattr(response2, "tool_calls", None):
                    tool_names2 = [tc["name"] for tc in response2.tool_calls]
                    logger.info(f"[Solver] LLM Call 2 tool calls: {tool_names2}")
                    return {
                        "messages": messages_for_call2 + [response2],
                        "agent_payload_log": state.get("agent_payload_log") or [],
                    }

                # Use response2 as our final response going forward
                response = response2

            solution_text = response.content or ""
            if not solution_text.strip():
                logger.warning("[Solver] Empty solution text after tool loop — forcing retry signal")
                return {
                    "solve_iterations": iteration + 1,
                    "agent_payload_log": state.get("agent_payload_log") or [],
                    "solver_output": {
                        "solution":         "",
                        "final_answer":     "",
                        "rag_context_used": False,
                        "calculator_used":  False,
                        "web_search_used":  False,
                    },
                }

            all_msgs = list(existing_msgs) + [response]

            calc_used = any(
                isinstance(m, ToolMessage) and "calculator" in (m.name or "").lower()
                for m in all_msgs
            )
            rag_used = _rag_ran_inline or rag_already_called or any(
                isinstance(m, ToolMessage) and "rag" in (m.name or "").lower()
                for m in all_msgs
            )
            web_used = any(
                isinstance(m, ToolMessage) and "web_search" in (m.name or "").lower()
                for m in all_msgs
            )

            final_answer = self._extract_final_answer(solution_text)
            payload(
                state, "solver_agent",
                summary=(
                    f"Solution produced | attempt {iteration + 1} | "
                    f"topic={topic} | difficulty={difficulty}"
                ),
                fields={
                    "Topic":         topic,
                    "Intent":        intent_type,
                    "Difficulty":    difficulty,
                    "Attempt":       str(iteration + 1),
                    "RAG used":      str(rag_used),
                    "RAG query":     rag_query,
                    "RAG preview":   rag_result[:500] if isinstance(rag_result, str) else str(rag_result)[:500],
                    "RAG source":    "student's uploaded PDF" if rag_query else "",
                    "Calc used":     str(calc_used),
                    "Web used":      str(web_used),
                    "RAG indexed":   str(rag_available),
                    "LTM hint":      ltm_hint[:80] if ltm_hint else "none",
                    "Final answer":  final_answer[:80] if final_answer else "unknown",
                    "Preview":       solution_text[:120],
                },
            )

            logger.info(
                f"[Solver] Working done | iter={iteration + 1} "
                f"| calc={calc_used} | rag={rag_used} | web={web_used}"
            )
            return {
                "messages":         [response],
                "solve_iterations": iteration + 1,
                "agent_payload_log": state.get("agent_payload_log") or [],
                "solver_output": {
                    "solution":         solution_text,
                    "final_answer":     final_answer,
                    "rag_context_used": rag_used,
                    "calculator_used":  calc_used,
                    "web_search_used":  web_used,
                },
            }

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e) or "tokens" in str(e).lower():
                logger.warning(f"[Solver] Token/rate limit hit: {e}")
                return {
                    "solve_iterations": iteration + 1,
                    "hitl_required":    True,
                    "hitl_type":        "verification",
                    "hitl_reason":      "Token limit reached. Please try again in a moment.",
                    "agent_payload_log": state.get("agent_payload_log") or [],                                              
                    "solver_output": {
                        "solution":         "",
                        "final_answer":     "",
                        "rag_context_used": False,
                        "calculator_used":  False,
                        "web_search_used":  False,
                    }
                }
            logger.error(f"[Solver] failed: {e}")
            raise Agent_Exception(e, sys)