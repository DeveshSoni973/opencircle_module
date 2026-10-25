"""Core data models for opencircle_module."""

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TurnStatus(str, Enum):
    """Outcome of a single agent turn."""
    SPOKE = "spoke"
    SILENT = "silent"
    ERROR = "error"


@dataclass
class TurnResult:
    """What one agent did on one turn. Passed to on_round_complete."""
    agent_id: str
    status: TurnStatus
    content: Optional[str] = None  # set when status is SPOKE ([DONE] already stripped)
    error: Optional[str] = None    # set when status is ERROR
    done: bool = False             # agent said [DONE]: it stops for the rest of this run


class Message:
    """A single message in the canonical history."""
    
    def __init__(self, sender_id: str, content: str, msg_id: Optional[str] = None):
        self.sender_id = sender_id
        self.content = content
        self.msg_id = msg_id or str(uuid.uuid4())
    
    @property
    def is_user(self) -> bool:
        return self.sender_id == "usr"
    
    @property
    def is_silent(self) -> bool:
        return self.content.strip() == "[SILENT]"
    
    def __repr__(self) -> str:
        return f"Message(sender_id={self.sender_id[:8]!r}, content={self.content[:40]!r})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Message):
            return NotImplemented
        return self.msg_id == other.msg_id


class Agent:
    """An LLM agent in the group chat."""
    
    def __init__(
        self,
        name: str,
        model: str,
        provider: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_rounds: Optional[int] = None,
        base_url: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        self.name = name
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url                       # custom endpoint (proxy, self-hosted, gateway)
        self.agent_id = agent_id or str(uuid.uuid4())  # pass your own id to map turns onto your DB
        self.formatted_history = []
        self.history_cursor = -1
        self.rounds_taken = 0
        self.done = False              # set when this agent says [DONE]; reset every run
        self.max_rounds = max_rounds   # None = use GroupChat.max_rounds
        self.task = None
    
    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={self.model!r}, provider={self.provider!r})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Agent):
            return NotImplemented
        return self.agent_id == other.agent_id

