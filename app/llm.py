"""Plain HTTP client for any OpenAI-compatible chat endpoint (Groq, OpenRouter, Ollama, ...).

No LangChain or SDKs: one POST request. Any failure returns None, and the caller
falls back to a deterministic template, so the API never breaks because of the LLM.
"""
from __future__ import annotations

import json
import logging
import re

import requests

logger = logging.getLogger("currency_api.llm")


class LLMClient:
    def __init__(self, base_url: str | None, api_key: str | None, model: str | None, timeout: float = 20.0,
                 extra_body: dict | None = None):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.extra_body = extra_body or {}  # e.g. {"reasoning_effort": "low"} for reasoning models

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def chat(self, system: str, user: str, max_tokens: int = 1000) -> str | None:
        if not self.enabled:
            return None
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    **self.extra_body,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content.strip() if content else None
        except Exception as exc:  # network error, bad key, rate limit, unexpected JSON ...
            logger.warning("LLM call failed: %s", exc)
            return None

    def chat_json(self, system: str, user: str) -> dict | None:
        text = self.chat(system, user, max_tokens=800)
        if not text:
            return None
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
