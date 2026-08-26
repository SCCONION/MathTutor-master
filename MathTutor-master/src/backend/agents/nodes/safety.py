import yaml
from functools import lru_cache
from pathlib import Path
from backend.agents import *
from backend.agents.nodes import *
from backend.agents.nodes import _HARD_BLOCK_KEYWORDS  # 下划线名字不会被 import * 导出，需显式导入

_SAFETY_POLICY_DIR = Path(__file__).resolve().parent / "security_checks"

@lru_cache(maxsize=1)
def _load_output_policy() -> dict:
    path = _SAFETY_POLICY_DIR / "output_policy.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Validate we got the right file
    if not isinstance(data, dict) or "blocked_output_patterns" not in data:
        raise ValueError(
            f"[Safety] Wrong file loaded. Keys found: "
            f"{list(data.keys()) if isinstance(data, dict) else type(data)}. "
            f"Path: {path.resolve()}"
        )
    return data

class SafetyAgent(BaseAgent):
    
    def _keyword_check(self, text: str) -> tuple[bool, str]:
        """Returns (is_blocked, reason). Fast — no LLM."""
        lower = text.lower()
        for kw in _HARD_BLOCK_KEYWORDS:
            if kw in lower:
                return True, f"keyword match: '{kw}'"
        return False, ""

    def _build_safety_prompt(self, solution_text: str, policy: dict) -> str:
        blocked  = "\n".join(f"- {p}" for p in policy.get("blocked_output_patterns", []))
        allowed  = "\n".join(f"- {p}" for p in policy.get("always_allowed", []))

        return f"""你是中学数学助手的输出安全审查员。

            审查下面解题代理的输出，判断是否安全地展示给学生。
            阻止的模式（命中任意一条 → passed=false）：
            {blocked}

            始终允许（不要误报）：
            {allowed}

            规则：
            - 数学内容、LaTeX、逐步代数、定理 — 始终安全。
            - 只有在明确违反策略时才阻止。
            - 拿不准时放行 — 误杀会让学生反感。

            需要审查的解题输出：
            {solution_text[:2000]}"""

    def safety_agent(self, state: AgentState) -> dict:
        try:
            solver_out = state.get("solver_output") or {}
            solution_text = solver_out.get("solution", "")

            if not solution_text:
                return {"safety_passed": True}

            # ── Stage 1: keyword fast path ────────────────────────────────────
            is_blocked, kw_reason = self._keyword_check(solution_text)

            if is_blocked:
                policy  = _load_output_policy()
                message = policy.get("on_violation", {}).get(
                    "replacement_message",
                    "该解答因违反安全策略无法展示，请重新提问。",
                )
                payload(
                    state, "safety_agent",
                    summary = f"BLOCKED — keyword: {kw_reason}",
                    fields  = {"Reason": kw_reason},
                )
                logger.warning(f"[Safety] Keyword block | {kw_reason}")
                return {
                    "safety_passed":  False,
                    "safety_reason":  kw_reason,
                    "final_response": message,
                }

            # ── Stage 2: LLM policy check ─────────────────────────────────────
            policy = _load_output_policy()
            prompt = self._build_safety_prompt(solution_text, policy)

            result: SafetyOutput = self.llm.with_structured_output(
                SafetyOutput, method="function_calling"
            ).invoke([HumanMessage(content=prompt)])

            updates: dict = {
                "safety_passed": result.passed,
                "safety_reason": result.reason,
            }

            if not result.passed:
                replacement = policy.get("on_violation", {}).get(
                    "replacement_message",
                    "该解答因违反安全策略无法展示，请重新提问。",
                )
                updates["final_response"] = replacement
                logger.warning(
                    f"[Safety] LLM blocked output | "
                    f"violation={result.violation_type} | reason={result.reason}"
                )

            payload(
                state, "safety_agent",
                summary = f"{'PASSED' if result.passed else 'BLOCKED'} | {result.violation_type or 'ok'}",
                fields  = {
                    "Passed":    str(result.passed),
                    "Violation": result.violation_type,
                    "Reason":    result.reason,
                },
            )
            logger.info(f"[Safety] passed={result.passed}")
            return {**updates, "agent_payload_log": state.get("agent_payload_log") or []}

        except Exception as e:
            logger.error(f"[Safety] failed: {e}")
            raise Agent_Exception(e, sys)