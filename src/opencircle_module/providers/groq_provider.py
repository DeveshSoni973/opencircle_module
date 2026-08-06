"""Groq provider wrapper."""

import os
import re
from .base import BaseProvider


class GroqProvider(BaseProvider):
    """Groq API provider."""
    
    def __init__(self, api_key=None):
        super().__init__(api_key or os.environ.get("GROQ_API_KEY"))
        try:
            from groq import AsyncGroq, Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")
        self._async_client = AsyncGroq(api_key=self.api_key)
        self._sync_client = Groq(api_key=self.api_key)
    
    def _strip_reasoning(self, text):
        """Strip reasoning/thinking blocks from response."""
        # Strip Qwen ## Reasoning blocks
        text = re.sub(r'\n*## Reasoning\n.*?(?=\n##|\Z)', '', text, flags=re.DOTALL)
        # Strip <think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Strip leading whitespace
        text = text.strip()
        return text
    
    async def chat(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return self._strip_reasoning(response.choices[0].message.content)
    
    def sync_chat(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        response = self._sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return self._strip_reasoning(response.choices[0].message.content)
    
    def get_name(self):
        return "groq"