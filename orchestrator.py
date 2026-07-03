"""Main group chat orchestrator with parallel single-phase execution."""

import asyncio
import logging
import os
from .models import Message, RoundResult
from .registry import AgentRegistry
from .context_builder import ContextBuilder
from .system_prompts.loader import SystemPromptLoader
from .providers.factory import ProviderFactory

from langsmith import traceable

logger = logging.getLogger(__name__)


class GroupChat:
    """
    Decentralized multi-agent group chat.
    Agents self-decide whether to speak via parallel single-phase execution.
    """
    
    def __init__(
        self,
        agents,
        system_prompt="default",
        max_rounds=5,
        prompts_dir=None,
        on_round_complete=None,
        on_message=None,
        include_silent_in_history=False,
        max_history_messages=None,
        langchain_api_key=None,
        langchain_project="opencircle"
    ):
        self.agents = agents
        self.system_prompt_name = system_prompt
        self.max_rounds = max_rounds
        self.on_round_complete = on_round_complete
        self.on_message = on_message
        self.include_silent = include_silent_in_history
        self.max_history = max_history_messages
        
        self.registry = AgentRegistry()
        for agent in agents:
            self.registry.register(agent)
        
        self.builder = ContextBuilder(self.registry)
        self.prompt_loader = SystemPromptLoader(prompts_dir)
        
        self._providers = {}
        self.history = []
        self.current_query = ""
        self.round_num = 0
        self.terminated = False
        if langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = langchain_project
    
    def _get_provider(self, agent):
        """Get or create provider for an agent."""
        key = f"{agent.provider}:{agent.api_key or 'env'}"
        if key not in self._providers:
            self._providers[key] = ProviderFactory().create(
                agent.provider,
                api_key=agent.api_key
            )
        return self._providers[key]
    
    def _build_system_prompt(self, agent):
        """Build system prompt for an agent from file template."""
        other_names = [a.name for a in self.agents if a.agent_id != agent.agent_id]
        return self.prompt_loader.load(
            self.system_prompt_name,
            agent_name=agent.name,
            other_agents=", ".join(other_names),
            user_query=self.current_query,
        )
    
    @traceable(run_type="chain", name="agent_turn")
    async def _agent_respond(self, agent):
        """Single agent: build context, call LLM, return response or None."""
        system = self._build_system_prompt(agent)

        if agent.history_cursor==-1 or self.max_history is not None:
            agent.formatted_history = self.builder.build(
                self.history,
                agent,
                include_silent=self.include_silent,
                max_messages=self.max_history,
                prepend_system=False,
            )
        else:
            new_msgs=self.history[agent.history_cursor + 1:]
            if new_msgs:
                built = self.builder.build(
                    new_msgs,
                    agent,
                    include_silent=self.include_silent,
                    prepend_system=False,
                )

                if (agent.formatted_history and built
                    and agent.formatted_history[-1]["role"] == built[0]["role"]):
                    agent.formatted_history[-1]["content"] += "\n\n" + built[0]["content"]
                    agent.formatted_history.extend(built[1:])
                else:
                    agent.formatted_history.extend(built)
        
        agent.history_cursor = len(self.history) - 1
        messages = [{"role": "system", "content": system}] + agent.formatted_history
        
        provider = self._get_provider(agent)
        
        try:
            response = await provider.chat(
                model=agent.model,
                messages=messages,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
            )
            
            if response.strip() == "[SILENT]":
                return agent.agent_id, None
            return agent.agent_id, response
            
        except Exception as e:
            logger.error(f"Response failed for {agent.name}: {e}")
            return agent.agent_id, None
    
    async def _run_round(self):
        """Execute one round: all agents parallel, gather responses."""
        self.round_num += 1
        
        tasks = [self._agent_respond(agent) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = {}
        messages_added = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
                continue
            agent_id, response = result
            responses[agent_id] = response
            
            if response is not None:
                msg = Message(sender_id=agent_id, content=response)
                self.history.append(msg)
                messages_added.append(msg)
                
                if self.on_message:
                    self.on_message(msg)
        
        terminated = False
        reason = None
        
        if not messages_added:
            terminated = True
            reason = "all_silent"
        elif any("[DONE]" in (r or "") for r in responses.values()):
            terminated = True
            reason = "agent_done"
        
        return RoundResult(
            round_num=self.round_num,
            responses=responses,
            messages_added=messages_added,
            terminated=terminated,
            termination_reason=reason,
        )
    
    async def run(self, query):
        """Run the group chat on a user query, continuing prior history. Returns full history."""
        self.current_query = query
        self.history.append(Message(sender_id="usr", content=query))
        self.round_num = 0
        self.terminated = False
        
        for _ in range(self.max_rounds):
            if self.terminated:
                break
            
            result = await self._run_round()
            
            if self.on_round_complete:
                self.on_round_complete(result)
            
            if result.terminated:
                self.terminated = True
                break
        
        return self.history
    
    def run_sync(self, query):
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(query))
    
    def get_transcript(self):
        """Get human-readable conversation transcript."""
        lines = []
        for msg in self.history:
            name = self.registry.get_name(msg.sender_id)
            lines.append(f"{name}: {msg.content}")
        return "\n".join(lines)
    
    def reset(self):
        """Clear history to start a new conversation with the same agents."""
        self.history = []
        self.round_num = 0
        self.terminated = False
        self.current_query = ""

        for agent in self.agents:
            agent.formatted_history = []
            agent.history_cursor = -1