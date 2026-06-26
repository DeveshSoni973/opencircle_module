"""Provider factory."""

from .groq_provider import GroqProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .nvidia_provider import NvidiaProvider
from .openrouter_provider import OpenRouterProvider


class ProviderFactory:
    """Factory for creating provider instances."""
    
    _providers = {
        "groq": GroqProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,
        "nvidia": NvidiaProvider,
        "nim": NvidiaProvider,
        "openrouter": OpenRouterProvider,
    }
    
    def create(self, provider_name, api_key=None, **kwargs):
        """Create a provider instance by name."""
        name = provider_name.lower()
        if name not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")
        return self._providers[name](api_key=api_key, **kwargs)
    
    def available(self):
        """List available provider names."""
        return list(self._providers.keys())