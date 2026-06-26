"""
opencircle_module - A decentralized multi-agent group chat system.

Agents self-decide whether to speak. No external selector.
Parallel single-phase execution with dynamic system prompts.
"""

from .models import Agent, Message, RoundResult
from .orchestrator import GroupChat
from .registry import AgentRegistry
from .context_builder import ContextBuilder
from .system_prompts.loader import SystemPromptLoader

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Message",
    "RoundResult",
    "GroupChat",
    "AgentRegistry",
    "ContextBuilder",
    "SystemPromptLoader",
]