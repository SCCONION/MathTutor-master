from backend.agents import *
from backend.agents.nodes import *
from backend.agents.nodes.memory.memory_manager import format_ltm_for_explainer


class ExplainerAgent(BaseAgent):

    def _build_explanation_prompt(
        self,
        problem_text:     str,
        solution_text:    str,
        verifier_verdict: str,
        topic:            str,
        difficulty:       str,
        ltm_hint:         str,
    ) -> str:
        base = f"""你是一位资深的中学数学老师，正在撰写一份标准答案式讲解。

        学生提交了这道题目，解题代理已经给出了经过校验的正确解答。
        你的任务是将解答组织成清晰、严谨的讲解，让做错这道题的学生
        能够理解答案，更能掌握方法。

        数学符号规则（严格遵守）：
        - 使用题目中的原始变量名，绝不改名。
        - 积分：写 ∫ f(x) dx — 始终包含微分符号。
        - 分数：(分子)/(分母) — 全部加括号。
        - 幂次：与题目保持一致（x^2 或 x²）。
        - 根式：统一使用 √(表达式)，不要混用 sqrt()。
        - 精确形式：答案用分数/根式/π/e/ln，不用小数。
        - 极限：lim_{{x → a}} 带箭头。
        - 求和：Σ_{{k=1}}^{{n}} 带明确上下界。
        - 向量：→a 表示向量，|→a| 表示模长。
        - final_answer：始终是非空字符串 — 写 "0" 而不是 0，不要用布尔值。
          如果答案是零写 "0"，如果不存在这样的点写 "0个点"。

        步骤结构：
        - 每一步都要逐步展示完整的代数推导。
        - 每次变形单独一行，以 = 新表达式 结尾。
        - step.result 是该步骤圈出的结果表达式 — 只含数学内容。
        - step.why 只用于解释非显而易见的操作（例如意外的换元选择）。
        - step.inline_diagram：仅当该步骤确实需要图示时才加入
          简单的 ASCII/Unicode 示意图（数轴、三角形、文氏图）。
          代数步骤留空。

        题目（主题：{topic} | 难度：{difficulty}）：
        {problem_text}

        已校验的正确解答（以此为准，不要重新计算）：
        {solution_text}

        校验备注：
        {verifier_verdict}

        始终使用中文讲解。"""

        if ltm_hint:
            base += f"""

        学生个性化信息（据此调整讲解方式）：
        {ltm_hint}

        根据以上信息，请让 common_mistakes 直接针对该学生已知的错误模式，
        并在 key_concepts 中优先讲解学生历史上掌握薄弱的重点概念。

        始终使用中文讲解。"""

        return base

    def explainer_agent(self, state: AgentState) -> dict:
        try:
            parsed        = state.get("parsed_data") or {}
            problem_text  = parsed.get("problem_text") or ""
            topic         = parsed.get("topic") or "mathematics"

            plan          = state.get("solution_plan") or {}
            difficulty    = plan.get("difficulty") or "medium"
            intent_type   = plan.get("intent_type", "solve")

            solver_out    = state.get("solver_output") or {}
            solution_text = solver_out.get("solution", "")
            rag_used      = solver_out.get("rag_context_used", False)
            calc_used     = solver_out.get("calculator_used", False)
            web_used      = solver_out.get("web_search_used", False)

            verifier_out  = state.get("verifier_output") or {}
            verdict       = verifier_out.get("verdict", "")
            verifier_conf = verifier_out.get("confidence", 0.0)

            ltm_context = state.get("ltm_context") or {}
            ltm_hint    = format_ltm_for_explainer(ltm_context, topic)

            prompt = self._build_explanation_prompt(
                problem_text     = problem_text,
                solution_text    = solution_text,
                verifier_verdict = verdict,
                topic            = topic,
                difficulty       = difficulty,
                ltm_hint         = ltm_hint,
            )

            result: ExplainerOutput = self.llm.with_structured_output(
                ExplainerOutput, method="function_calling"
            ).invoke([HumanMessage(content=prompt)])

            final_md      = render_md(result, problem_text)
            explainer_dict = result.model_dump()

            payload(
                state, "explainer_agent",
                summary=(
                    f"{len(result.steps)} steps | "
                    f"difficulty={result.difficulty_rating} | "
                    f"intent={intent_type} | "
                    f"personalised={'yes' if ltm_hint else 'no'}"
                ),
                fields={
                    "Topic":           topic,
                    "Intent":          intent_type,
                    "Steps":           str(len(result.steps)),
                    "Key formulae":    str(len(result.key_formulae)),
                    "Key concepts":    str(len(result.key_concepts)),
                    "Common mistakes": str(len(result.common_mistakes)),
                    "Difficulty":      result.difficulty_rating,
                    "Verifier conf":   f"{verifier_conf:.0%}",
                    "RAG used":        str(rag_used),
                    "Calc used":       str(calc_used),
                    "Web used":        str(web_used),
                    "LTM hint":        ltm_hint[:80] if ltm_hint else "none",
                    "Preview":         final_md[:120],
                },
            )
            logger.info(
                f"[Explainer] done | steps={len(result.steps)} "
                f"difficulty={result.difficulty_rating} "
                f"ltm_personalised={bool(ltm_hint)}"
            )

            return {
                "explainer_output":  explainer_dict,
                "final_response":    final_md,
                "conversation_log":  [final_md],
                "hitl_required":     True,
                "hitl_type":         "satisfaction",
                "hitl_reason":       "Explanation delivered — awaiting student feedback.",
                "follow_up_question": None,
                "student_satisfied":  None,
                "agent_payload_log": state.get("agent_payload_log") or [],
            }

        except Exception as e:
            logger.error(f"[Explainer] failed: {e}")
            raise Agent_Exception(e, sys)