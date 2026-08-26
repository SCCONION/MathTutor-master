from backend.agents import *
from backend.agents.nodes import *

class VerifierAgent(BaseAgent):

    def verifier_agent(self, state: AgentState) -> AgentState:
        try:
            parsed       = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""
            solver_out   = state.get("solver_output") or {}
            iteration    = state.get("solve_iterations", 1)
            solution     = solver_out.get("solution", "")
            final_answer = solver_out.get("final_answer", "")

            if not solution.strip():
                logger.warning("[Verifier] Empty solution received — routing back to solver")
                return {
                    "verifier_output": {
                        "status":        "incorrect",
                        "verdict":       "解题代理没有生成任何解答文本。",
                        "suggested_fix": "解题代理必须写出完整的文字解答。调用工具后必须跟有完整的推导过程。",
                        "confidence":    0.0,
                    }
                }

            _VERIFIER_PROMPT = """你是一位严格的中学数学校验员。

                按下述三个标准核对下面的解答与题目：
                1. 正确性 — 每一步代数变形是否都有效？如果有问题，指出步骤编号。
                2. 单位/定义域 — 最终答案是否在正确的定义域/范围/单位内？
                3. 边界情况 — 除零、对数/根式无定义、空集等。

                状态规则：
                - 'correct' 正确          → 三项检查全部通过。
                - 'partially_correct' 部分正确 → 方法正确但存在计算/符号错误。
                - 'incorrect' 错误         → 方法本身有误。
                - 'needs_human' 需人工确认 → 你确实无法判断正确性。
                                          仅当确实缺少领域知识，或问题需要超出
                                          中学数学范围的专业知识时使用。

                不正确时：suggested_fix 必须指明具体步骤和具体错误。
                不要说"检查第3步" — 要写清楚哪里错了、应该怎么做。

                需要人工确认时：hitl_reason 必须用通俗的语言向学生清楚说明：
                - 为什么自动化校验器无法确认这个解答
                - 具体哪部分不确定（例如"第3步使用了超出中学教学大纲的定理，
                  无法自动校验"）
                - 人工审查者应该检查什么
                把 hitl_reason 写成直接对学生说的话 — 不要写成内部笔记。

                题目：
                {problem_text}

                解答（第 {iteration} 次尝试）：
                {solution}

                声称的最终答案：{final_answer}"""
            
            result: VerifierOutput = self.llm.with_structured_output(
                VerifierOutput, method="function_calling"
            ).invoke([HumanMessage(content=_VERIFIER_PROMPT.format(
                problem_text      = problem_text,
                iteration         = iteration,
                solution          = solution,
                final_answer      = final_answer,
            ))])
 
            updates: dict = {"verifier_output": result.model_dump()}
 
            if result.status == "needs_human":
                updates["hitl_required"] = True
                updates["hitl_type"]     = "verification"
                updates["hitl_reason"]   = (
                    result.hitl_reason or result.verdict or "校验器无法确认该解答的正确性。"
                )
 
            payload(
                state, "verifier_agent",
                summary = f"{result.status.upper()} | {result.confidence:.0%} confidence",
                fields  = {
                    "Status":  result.status,
                    "Verdict": result.verdict[:180],
                    "Fix":     result.suggested_fix[:120] if result.suggested_fix else None,
                },
            )
            logger.info(f"[Verifier] status={result.status} confidence={result.confidence:.2f}")
            return {**updates, "agent_payload_log": state.get("agent_payload_log") or []}

        except Exception as e:
            logger.error(f"[Verifier] failed: {e}")
            raise Agent_Exception(e, sys)