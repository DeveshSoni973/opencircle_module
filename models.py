"""Core data models for opencircle_module."""

import uuid
from typing import Optional


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
    
    @property
    def is_done(self) -> bool:
        return "[DONE]" in self.content
    
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
    ):
        self.name = name
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.agent_id = str(uuid.uuid4())
    
    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={self.model!r}, provider={self.provider!r})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Agent):
            return NotImplemented
        return self.agent_id == other.agent_id


class RoundResult:
    """Result of one round of group chat."""
    
    def __init__(
        self,
        round_num: int,
        responses: dict[str, Optional[str]],
        messages_added: list[Message],
        terminated: bool,
        termination_reason: Optional[str] = None,
    ):
        self.round_num = round_num
        self.responses = responses
        self.messages_added = messages_added
        self.terminated = terminated
        self.termination_reason = termination_reason
    
    def __repr__(self) -> str:
        return (
            f"RoundResult(round={self.round_num}, "
            f"responses={len(self.responses)}, "
            f"terminated={self.terminated})"
        )