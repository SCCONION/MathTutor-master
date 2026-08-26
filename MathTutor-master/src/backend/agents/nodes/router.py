from backend.agents import *
from backend.agents.nodes import *


class IntentRouterAgent(BaseAgent):

    def intent_router_agent(self, state: AgentState) -> AgentState:
        try:
            parsed       = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""

            _ROUTER_PROMPT = """\
                你在为一名中学数学学生分类输入，以决定如何回应。

                返回四项内容：

                1. topic — 最匹配的数学领域（algebra 代数 | probability 概率 | calculus 微积分 |
                           linear_algebra 线性代数 | geometry 几何 | trigonometry 三角 | statistics 统计 | number_theory 数论）
                2. difficulty — easy 简单 | medium 中等 | hard 困难（中学标准；非题目类输入用 medium）
                3. solver_strategy — 最直接的解题路径（1-2 句话）。
                                     对于非解题类意图，描述讲解或生成的方法。
                4. intent_type — 用且仅用以下之一对学生意图进行分类：

                    solve 解题        → 学生想要某个具体数值/代数题目的完整解题过程
                                     默认 — 不确定但输入包含具体题目时使用
                    explain 讲解      → 学生想要概念、定理或方法的讲解
                                     触发词："什么是"、"解释"、"为什么"、"怎么理解"、"告诉我关于"、
                                     "描述"、"有什么区别"
                    hint 提示        → 学生想要解题的小提示/线索
                                     触发词："提示"、"线索"、"告诉我从哪里开始"、"我的思路对吗"
                    formula_lookup 公式查询 → 学生只想要公式或定理表述
                                     触发词："公式"、"写出定理"、"这个公式是什么"
                    research 研究    → 学生想了解最新进展、历史或一般数学知识
                                     不属于具体题目求解，也不属于简单概念讲解
                                     触发词："最新"、"发现"、"历史"、"谁发现了"、
                                     "应用"、"现实应用"、"讲点有趣的"
                    generate 出题    → 学生想要练习题目、考试题或示例
                                     触发词："给我"、"创建"、"出"、"生成"、"一些题"、
                                     "练习题"、"例题"、"考考我"、"出几道题"

                关键规则：
                - 如果输入包含需要求解的具体数字、方程或表达式 → 一律 "solve"
                - "什么是贝叶斯定理？" → "explain"（不是 "solve"）
                - "分部积分公式是什么？" → "formula_lookup"
                - "给我出5道概率题（高中水平）" → "generate"
                - "数学界最近有什么新发现？" → "research"
                - "解释一下 AM-GM 不等式及使用时机" → "explain"
                - 在 explain 和 solve 之间拿不准时：如果没有具体数值题目 → "explain"

                学生输入：
                {problem_text}"""

            structured_llm = self.llm.with_structured_output(IntentRouterOutput, method="function_calling")
            result: IntentRouterOutput = structured_llm.invoke(
                [HumanMessage(content=_ROUTER_PROMPT.format(problem_text=problem_text))]
            )

            payload(
                state, "intent_router",
                summary=f"{result.topic.title()} | {result.difficulty.title()} | {result.intent_type}",
                fields={
                    "Topic":       result.topic,
                    "Difficulty":  result.difficulty,
                    "Intent":      result.intent_type,
                    "Strategy":    result.solver_strategy,
                },
            )

            logger.info(
                f"[Router] topic={result.topic} difficulty={result.difficulty} "
                f"intent={result.intent_type}"
            )
            return {"solution_plan": result.model_dump(), "agent_payload_log": state.get("agent_payload_log") or []}

        except Exception as e:
            logger.error(f"[Router] failed: {e}")
            raise Agent_Exception(e, sys)