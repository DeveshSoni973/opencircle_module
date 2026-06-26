"""Google Gemini provider wrapper."""

import os
from .base import BaseProvider


class GoogleProvider(BaseProvider):
    """Google Gemini API provider."""
    
    def __init__(self, api_key=None):
        super().__init__(api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
        genai.configure(api_key=self.api_key)
        self._genai = genai
    
    async def chat(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                chat_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_messages.append({"role": "model", "parts": [msg["content"]]})
        
        model_obj = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=system_msg
        )
        
        import asyncio
        loop = asyncio.get_event_loop()
        chat = model_obj.start_chat(history=chat_messages[:-1] if len(chat_messages) > 1 else [])
        response = await loop.run_in_executor(
            None,
            lambda: chat.send_message(
                chat_messages[-1]["parts"][0] if chat_messages else "",
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
            )
        )
        return response.text
    
    def sync_chat(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                chat_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_messages.append({"role": "model", "parts": [msg["content"]]})
        
        model_obj = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=system_msg
        )
        chat = model_obj.start_chat(history=chat_messages[:-1] if len(chat_messages) > 1 else [])
        response = chat.send_message(
            chat_messages[-1]["parts"][0] if chat_messages else "",
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
        )
        return response.text
    
    def get_name(self):
        return "google"