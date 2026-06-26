"""Anthropic/Claude provider wrapper."""

import os
from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key=None):
        super().__init__(api_key or os.environ.get("ANTHROPIC_API_KEY"))
        try:
            from anthropic import AsyncAnthropic, Anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        self._async_client = AsyncAnthropic(api_key=self.api_key)
        self._sync_client = Anthropic(api_key=self.api_key)
    
    async def chat(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        response = await self._async_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=chat_messages,
            **kwargs
        )
        return response.content[0].text
    
    def sync_chat(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        response = self._sync_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=chat_messages,
            **kwargs
        )
        return response.content[0].text
    
    def get_name(self):
        return "anthropic"