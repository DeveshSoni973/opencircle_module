"""Build LLM context from canonical history."""


class ContextBuilder:
    """Transforms canonical UUID-based history into provider-ready messages."""
    
    def __init__(self, registry):
        self.registry = registry
    
    def build(self, history, current_agent, system_prompt, include_silent=False, max_messages=None):
        """
        Build context for a single agent.
        Returns list of {"role": str, "content": str} for LLM API.
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        hist = history
        if max_messages and len(hist) > max_messages:
            hist = hist[-max_messages:]
        
        built = []
        for msg in hist:
            if not include_silent and msg.is_silent:
                continue
            
            role = self._resolve_role(msg, current_agent)
            content = self._resolve_content(msg, current_agent)
            built.append({"role": role, "content": content})
        
        merged = self._merge_consecutive_roles(built)
        messages.extend(merged)
        
        return messages
    
    def _resolve_role(self, msg, current_agent):
        """Determine role: assistant if msg is from current agent, else user."""
        if msg.sender_id == current_agent.agent_id:
            return "assistant"
        return "user"
    
    def _resolve_content(self, msg, current_agent):
        """Resolve sender name and format content."""
        if msg.sender_id == current_agent.agent_id:
            return msg.content
        
        name = self.registry.get_name(msg.sender_id)
        return f"{name}: {msg.content}"
    
    def _merge_consecutive_roles(self, messages):
        """Merge consecutive messages with same role."""
        if not messages:
            return []
        
        merged = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg)
        
        return merged