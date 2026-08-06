"""Agent registry — UUID to name mapping and lookups."""

from typing import Optional
from .models import Agent


class AgentRegistry:
    """Bidirectional mapping between agent UUIDs and names."""
    
    def __init__(self):
        self._by_id = {}
        self._by_name = {}
        self._user_label = "usr"
    
    def register(self, agent: Agent) -> None:
        """Register an agent."""
        self._by_id[agent.agent_id] = agent
        self._by_name[agent.name] = agent
    
    def get_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get agent by UUID."""
        return self._by_id.get(agent_id)
    
    def get_by_name(self, name: str) -> Optional[Agent]:
        """Get agent by name."""
        return self._by_name.get(name)
    
    def get_name(self, agent_id: str) -> str:
        """Get display name for an agent_id. Returns 'User' for user."""
        if agent_id == self._user_label:
            return "User"
        agent = self._by_id.get(agent_id)
        return agent.name if agent else agent_id[:8]
    
    def get_id(self, name: str) -> Optional[str]:
        """Get UUID for a name."""
        agent = self._by_name.get(name)
        return agent.agent_id if agent else None
    
    def all_agents(self) -> list[Agent]:
        """Return all registered agents."""
        return list(self._by_id.values())
    
    def all_ids(self) -> list[str]:
        """Return all agent IDs."""
        return list(self._by_id.keys())
    
    def all_names(self) -> list[str]:
        """Return all agent names."""
        return list(self._by_name.keys())
    
    def __len__(self) -> int:
        return len(self._by_id)
    
    def __repr__(self) -> str:
        names = ", ".join(self.all_names())
        return f"AgentRegistry({len(self)} agents: {names})"