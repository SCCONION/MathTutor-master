import json
import tempfile
import os
import re 
import sys
from typing import Optional, Tuple
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
#from groq import Groq#语音识别

from backend.exceptions import Agent_Exception
from backend.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

def _coerce_bools(data: dict, bool_fields: set = None) -> dict:
    """
    LLMs (especially via Groq tool-calling) sometimes return 'true'/'false'
    strings instead of JSON booleans.  This coerces them before Pydantic
    validation so we never get a schema mismatch error.
    """
    bool_like = {"true": True, "false": False, "1": True, "0": False}
    for k, v in data.items():
        if bool_fields and k not in bool_fields:
            continue
        if isinstance(v, str) and v.lower() in bool_like:
            data[k] = bool_like[v.lower()]
    return data


def _log_payload(state: dict, node: str, summary: str, fields: dict) -> None:
    """
    Append a structured payload summary to state["agent_payload_log"].
    Called by every agent after it writes its result so the activity panel
    in app.py can display what the agent decided / produced.

    fields: a flat dict of key → value to show — keep values short strings.
    """
    log: list = state.get("agent_payload_log") or []
    log.append({
        "node":    node,
        "summary": summary,
        "fields":  {k: str(v)[:200] for k, v in fields.items() if v not in (None, "", [], {})},
    })
    state["agent_payload_log"] = log
    return log


def _render_markdown(result, problem_text: str) -> str:
    """
    Convert ExplainerOutput into a rich markdown string for the chat bubble.
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "## 📘 Solution",
        "",
        f"**Problem:** {problem_text}",
        "",
        "---",
        "",
    ]

    # ── Approach summary ──────────────────────────────────────────────────────
    lines += [
        "### 💡 Approach",
        "",
        result.approach_summary,
        "",
    ]

    # ── Key formulae ──────────────────────────────────────────────────────────
    if result.key_formulae:
        lines += ["### 📐 Formulae Used", ""]
        for f in result.key_formulae:
            lines += [
                f"- $${f}$$",
            ]
        lines.append("")

    lines += ["---", ""]

    # ── Step-by-step working ──────────────────────────────────────────────────
    lines += ["### ✍️ Working", ""]

    for step in result.steps:
        # Step heading
        lines += [
            f"**Step {step.step_number} — {step.heading}**",
            "",
        ]

        # Working lines — each on its own line, indented
        for wl in step.working.strip().splitlines():
            wl = wl.strip()
            if wl:
                lines.append(f"&emsp;{wl}")
        lines.append("")

        # Result on its own line in a LaTeX block
        lines += [
            "&emsp;∴ &nbsp; Result:",
            "",
            f"$$${step.result}$$$",
            "",
        ]

        # Optional inline diagram
        if step.inline_diagram:
            lines += [
                "```",
                step.inline_diagram.strip(),
                "```",
                "",
            ]

        # Optional why note
        if step.why:
            lines += [
                f"> 📎 *{step.why}*",
                "",
            ]

        lines.append("")

    lines += ["---", ""]

    # ── Final answer ──────────────────────────────────────────────────────────
    lines += [
        "### ✅ Final Answer",
        "",
        "$$" + result.final_answer + "$$",
        "",
        "---",
        "",
    ]

    # ── Key concepts ──────────────────────────────────────────────────────────
    if result.key_concepts:
        lines += ["### 🧠 Key Concepts", ""]
        for c in result.key_concepts:
            lines.append(f"- {c}")
        lines.append("")

    # ── Common mistakes ───────────────────────────────────────────────────────
    if result.common_mistakes:
        lines += ["### ⚠️ Common Mistakes", ""]
        for m in result.common_mistakes:
            lines.append(f"- {m}")
        lines.append("")

    # ── Difficulty badge ──────────────────────────────────────────────────────
    badge = {
        "easy":   "🟢 Easy",
        "medium": "🟡 Medium",
        "hard":   "🔴 Hard",
    }.get(result.difficulty_rating.lower(), result.difficulty_rating)
    lines.append(f"*Difficulty: {badge}*")

    return "\n".join(lines)


def _parse_xml_response(raw_text: str) -> tuple:
    """
    Parse the XML-tagged response the LLM is asked to produce.

    Expected format:
        <content>
        ...markdown here, no escaping needed...
        </content>

    Falls back gracefully: if tags are missing, treats whole text as content.
    """
    content_match = re.search(r"<content>(.*?)</content>", raw_text, re.S)
    
    content = content_match.group(1).strip() if content_match else raw_text.strip()

    if not content_match:
        logger.warning("[DirectResponse] <content> tag missing — using raw text as content")

    return content


def _get_secret(key: str, default: str = "") -> str:
    """
    Read from st.secrets (Streamlit Cloud) first,
    fall back to os.getenv / .env (local development).
    """
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


class MediaProcessor:

    def __init__(self):
        self.vision_llm: Optional[ChatOpenAI] = None
        self.groq_client: Optional[Groq] = None
        self._initialize_clients()


    def _initialize_clients(self) -> None:
        # DeepSeek Vision───────────────────────────────────────────────
       
        try:
            api_key = _get_secret("DEEPSEEK_API_KEY")

            if not api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY not set"
                )

            self.vision_llm = ChatOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                model="deepseek-v4-flash-vision-exp",
                temperature=0.1,
            )

            logger.info(
                "[MediaProcessor] DeepSeek Vision initialized"
            )

        except Exception as exc:
            logger.error(
                f"[MediaProcessor] DeepSeek Vision init failed: {exc}"
            )


        # ── Groq Whisper (ASR) ────────────────────────────────────────────────
        try:
            api_key = _get_secret("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            self.groq_client = Groq(api_key=api_key)
            logger.info("[MediaProcessor] Groq Whisper client initialised")
        except Exception as exc:
            logger.error(f"[MediaProcessor] Groq Whisper init failed: {exc}")



    def process_image(self, image_input) -> Tuple[str, float]:

        if not self.vision_llm:
            raise RuntimeError(
                "DeepSeek Vision client not initialized. "
                "Set DEEPSEEK_API_KEY."
            )

        try:

            # 读取图片
            if isinstance(image_input, (str, os.PathLike)):

                with open(image_input, "rb") as fh:
                    image_bytes = fh.read()

            else:
                image_bytes = bytes(image_input)


            import base64

            image_base64 = base64.b64encode(
                image_bytes
            ).decode("utf-8")


            response = self.vision_llm.invoke(
                [
                    {
                        "role": "system",
                        "content":
                        """
                        You are a mathematical OCR assistant.
                        Extract all text, formulas and symbols
                        from the image.
                        Return only the recognized problem text.
                        """
                    },
                    {
                        "role": "user",
                        "content":[
                            {
                                "type":"text",
                                "text":
                                "Read this math problem image."
                            },
                            {
                                "type":"image_url",
                                "image_url":{
                                    "url":
                                    f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
            )


            raw_text = response.content


            cleaned = self.clean_extracted_text(
                raw_text
            )


            confidence = 0.9


            logger.info(
                f"[DeepSeek Vision OCR] "
                f"{len(cleaned)} chars"
            )


            return cleaned, confidence


        except Exception as exc:

            logger.error(
                f"[Vision OCR] Error: {exc}"
            )

            raise Agent_Exception(exc, sys)


    def process_audio(self, audio_input) -> Tuple[str, float]:

        if not self.groq_client:
            raise RuntimeError(
                "Groq client not initialised. Ensure GROQ_API_KEY is set."
            )

        tmp_path: Optional[str] = None
        try:
            # Normalise to a file path (Groq SDK needs an open file handle)
            if isinstance(audio_input, (str, os.PathLike)):
                audio_path = str(audio_input)
                owns_tmp   = False
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(bytes(audio_input))
                tmp.close()
                audio_path = tmp.name
                owns_tmp   = True
                tmp_path   = audio_path

            with open(audio_path, "rb") as audio_fh:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=audio_fh,
                    model="whisper-large-v3",
                )

            transcript = transcription.text.strip()
            confidence = self._estimate_transcription_confidence(transcript)
            logger.info(f"[ASR] {len(transcript)} chars | conf={confidence:.2f}")
            return transcript, confidence

        except Agent_Exception:
            raise
        except Exception as exc:
            logger.error(f"[ASR] Error: {exc}")
            raise Agent_Exception(exc, sys)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


    confidence = 0.9

    def _estimate_transcription_confidence(self, transcript: str) -> float:
        """Heuristic confidence for Whisper (no per-word probs in Groq API)."""
        if not transcript:
            return 0.0
        score      = 0.9
        word_count = len(transcript.split())
        if word_count < 3:
            score -= 0.3
        elif word_count < 6:
            score -= 0.1
        for marker in ["?", "(unclear)", "(inaudible)", "[noise]", "[BLANK_AUDIO]"]:
            if marker.lower() in transcript.lower():
                score -= 0.1
        return round(max(0.0, min(1.0, score)), 3)


    def clean_extracted_text(self, text: str) -> str:
        """Remove common OCR noise and normalise whitespace."""
        if not text:
            return ""
        cleaned = text.strip()
        # Strip common OCR artifacts
        for artifact in ["|", "—", "•", "■", "□", "\x0c"]:
            cleaned = cleaned.replace(artifact, " ")
        # Collapse whitespace
        return " ".join(cleaned.split())