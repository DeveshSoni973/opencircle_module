"""OpenAI provider wrapper."""

import os
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key=None, base_url=None):
        super().__init__(api_key or os.environ.get("OPENAI_API_KEY"))
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
        self._sync_client = OpenAI(api_key=self.api_key, base_url=base_url)
    
    async def chat(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
    
    def sync_chat(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        response = self._sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
    
    def get_name(self):
        return "openai"