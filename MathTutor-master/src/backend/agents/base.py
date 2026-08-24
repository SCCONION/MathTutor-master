from __future__ import annotations

from langchain_openai import ChatOpenAI

from backend.agents.utils.helper import (
    _get_secret,
    MediaProcessor,
)


class BaseAgent:
    """
    Shared base for all agent nodes.

    self.llm
        General reasoning model.
        Used by router/parser/guardrail/direct response.

    self.reserve_llm
        Heavy reasoning model.
        Used by solver/verifier/explainer.

    Updated:
        Groq Llama-3.3-70B
        ->
        DeepSeek API compatible models
    """

    def __init__(self):

        # DeepSeek API Key
        key = _get_secret("DEEPSEEK_API_KEY")

        if not key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set — add it to .env "
                "(local) or Streamlit secrets."
            )


        # General purpose LLM
        #
        # Replace:
        # ChatGroq + llama-3.3-70b
        #
        # With:
        # DeepSeek-V3 / DeepSeek Chat
        #

        self.llm = ChatOpenAI(
            api_key=key,
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.1,
            max_tokens=2048,
            max_retries=2,
        )


        # Reserve model
        #
        # Original project used:
        #
        # GROQ_API_KEY_2
        #
        # for avoiding rate limit.
        #
        # DeepSeek API does not need a second key
        # during development, so reuse the same key.
        #

        self.reserve_llm = ChatOpenAI(
            api_key=key,
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.1,
            max_tokens=4096,
            max_retries=2,
        )


        # Media processor
        #
        # Image OCR / ASR entry.
        #
        # We will replace Google Vision
        # inside helper.py later.
        #

        self.media_processor = MediaProcessor()