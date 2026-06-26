"""OpenRouter provider wrapper."""

import os
from .base import BaseProvider


class OpenRouterProvider(BaseProvider):
    """OpenRouter unified API provider. OpenAI-compatible."""
    
    def __init__(self, api_key=None, base_url=None, http_referer=None, app_name=None):
        super().__init__(api_key or os.environ.get("OPENROUTER_API_KEY"))
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.http_referer = http_referer or os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost")
        self.app_name = app_name or os.environ.get("OPENROUTER_APP_NAME", "opencircle_module")
        
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        
        headers = {
            "HTTP-Referer": self.http_referer,
            "X-Title": self.app_name,
        }
        self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=headers)
        self._sync_client = OpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=headers)
    
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
        return "openrouter"