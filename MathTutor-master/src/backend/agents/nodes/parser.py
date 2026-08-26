from backend.agents import *
from backend.agents.nodes import *

class ParserAgent(BaseAgent):

    def parser_agent(self, state: AgentState) -> AgentState:
        try:
            raw_text = (
                state.get("user_corrected_text")
                or state.get("ocr_text")
                or state.get("transcript")
                or state.get("raw_text")
            )

            if not raw_text:
                return {
                    "hitl_required": True,
                    "hitl_type":     "bad_input",
                    "hitl_reason":   "无法从输入中提取出题目文字。",
                }

            _PARSER_PROMPT = """你是一个数学题目解析器。你的任务是对原始输入进行清洗和结构化。
                步骤：
                1. 修正 OCR/语音识别噪声，规范数学符号（分数、指数、积分、希腊字母等）。
                2. 提取题目中的所有变量和显式约束条件。
                3. 仅当题目缺少关键信息无法求解时才设置 needs_clarification=true — \
                例如缺少变量定义、句子被截断、或存在两个矛盾的条件。 \
                不要因为题目不常见或难度大就设置。如果能够做出合理解释，就正常解析。
                4. 重要：概念性问题如"什么是X"、"解释X"、"定义X"、"X的应用有哪些" — \
                这些都是有效的请求。对这类请求一律设置 needs_clarification=false。 \
                它们不需要数值答案即可完整。
                5. 当 needs_clarification=true 时，clarification_reason 要写成清晰、
                具体、面向学生的消息，明确说明缺少或歧义的信息是什么。

                输入：
                {_raw_text}"""
 
            structured_llm = self.llm.with_structured_output(ParserOutput, method="function_calling")
            parsed: ParserOutput = structured_llm.invoke(
                [HumanMessage(content=_PARSER_PROMPT.format(_raw_text=raw_text))])

            updates: dict = {"parsed_data": parsed.model_dump()}
 
            if parsed.needs_clarification:
                updates["hitl_required"] = True
                updates["hitl_type"]     = "clarification" 
                updates["hitl_reason"]  = parsed.clarification_reason or "题目信息不明确，请补充。"

            payload(
                state, "parser_agent",
                summary = f"Topic: {parsed.topic or '?'}",
                fields  = {
                    "Problem": parsed.problem_text[:120],
                    "Topic": parsed.topic,
                    "Variables": ", ".join(parsed.variables) if parsed.variables else None,
                    "Constraints": ", ".join(parsed.constraints) if parsed.constraints else None,
                    "Clarification": parsed.clarification_reason if parsed.needs_clarification else None,
                },
            )
            logger.info(f"[Parser] topic={parsed.topic} needs_clarification={parsed.needs_clarification}")
            return {**updates, "agent_payload_log": state.get("agent_payload_log") or []}
        
        except Exception as e:
            logger.error(f"[Parser] failed: {e}")
            raise Agent_Exception(e, sys)

