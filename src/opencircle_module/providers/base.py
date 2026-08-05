"""Base provider interface."""


class BaseProvider:
    """Abstract base for all LLM providers. Subclass and override all methods."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    async def chat(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        """Send chat completion request. Return response text."""
        raise NotImplementedError("Subclass must implement chat()")
    
    def sync_chat(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        """Synchronous chat completion."""
        raise NotImplementedError("Subclass must implement sync_chat()")
    
    def get_name(self):
        """Provider name identifier."""
        raise NotImplementedError("Subclass must implement get_name()")