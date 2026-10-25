"""Main group chat orchestrator with parallel single-phase execution."""

import asyncio
import logging
import os
from .models import Message, TurnResult, TurnStatus
from .registry import AgentRegistry
from .context_builder import ContextBuilder
from .system_prompts.loader import SystemPromptLoader
from .providers.factory import ProviderFactory
import re
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
        self.max_rounds = max_rounds  # default for agents that don't set their own
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
        if langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = langchain_project
    
    def _get_provider(self, agent):
        """Get or create provider for an agent."""
        kwargs = {"base_url": agent.base_url} if agent.base_url else {}
        key = (agent.provider, agent.api_key or "env", agent.base_url)
        if key not in self._providers:
            self._providers[key] = ProviderFactory().create(
                agent.provider,
                api_key=agent.api_key,
                **kwargs,
            )
        return self._providers[key]

    def _build_system_prompt(self, agent):
        """Build system prompt for an agent from file template."""
        other_names = [a.name for a in self.agents if a.agent_id != agent.agent_id]
        return self.prompt_loader.load(
            self.system_prompt_name,
            agent_name=agent.name,
            other_agents=", ".join(other_names),
        )

    @traceable(run_type="chain", name="agent_turn")
    async def _agent_respond(self, agent) -> TurnResult:
        """Single agent: build context, call LLM, return what happened."""
        system = self._build_system_prompt(agent)
        
        # Snapshot history at start — won't see concurrent additions
        snapshot = self.history[:len(self.history)]
        
        if agent.history_cursor == -1 or self.max_history:
            # Full rebuild (first time or sliding window active)
            agent.formatted_history = self.builder.build(
                snapshot,
                agent,
                include_silent=self.include_silent,
                max_messages=self.max_history,
                prepend_system=False,
            )
        else:
            # Incremental: only new messages since last snapshot
            new_msgs = snapshot[agent.history_cursor + 1:]
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
            else:
                agent.formatted_history.append({
                    "role": "user",
                    "content": (
                        "No new messages have been added since your last turn. "
                        "If you have more to contribute, continue naturally. "
                        "Otherwise respond with [SILENT]."
                    )
                })
        
        agent.history_cursor = len(snapshot) - 1
        
        messages = [{"role": "system", "content": system}] + agent.formatted_history

        try:
            provider = self._get_provider(agent)  # inside try: a bad provider config is this agent's error, not the run's
            response = await provider.chat(
                model=agent.model,
                messages=messages,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
            )
        except Exception as e:
            logger.error(f"Response failed for {agent.name}: {e}")
            return TurnResult(agent.agent_id, TurnStatus.ERROR, error=str(e))

        response = (response or "").strip()
        done = re.search(r"\[done\]", response, flags=re.IGNORECASE) is not None
        if done:
            response = re.sub(r"\s*\[done\]\s*", " ", response, flags=re.IGNORECASE).strip()
        match = re.search(r"\[silent\]", response, flags=re.IGNORECASE)
        if match:
            response = response[:match.start()].rstrip()
        if not response:
            return TurnResult(agent.agent_id, TurnStatus.SILENT, done=done)
        return TurnResult(agent.agent_id, TurnStatus.SPOKE, content=response, done=done)

    async def run(self, query):
        self.current_query = query
        self.history.append(Message(sender_id="usr", content=query))
        for agent in self.agents:
            agent.rounds_taken = 0
            agent.done = False

        self._launch_ready_agents()

        while any(a.task is not None for a in self.agents):
            pending_tasks = [a.task for a in self.agents if a.task is not None]
            done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                agent = next(a for a in self.agents if a.task is task)
                agent.task = None
                try:
                    result = await task
                except Exception as e:
                    logger.error(f"Task failed for {agent.name}: {e}")
                    result = TurnResult(agent.agent_id, TurnStatus.ERROR, error=str(e))

                if result.status == TurnStatus.SPOKE:
                    msg = Message(sender_id=result.agent_id, content=result.content)
                    self.history.append(msg)
                    agent.rounds_taken += 1
                    if self.on_message:
                        self.on_message(msg)
                if result.done:
                    agent.done = True  # only this agent stops; others keep going

                if self.on_round_complete:
                    self.on_round_complete(agent, result)

            self._launch_ready_agents()

        return self.history

    def _launch_ready_agents(self):
        for agent in self.agents:
            if agent.task is not None or agent.done:
                continue
            limit = agent.max_rounds if agent.max_rounds is not None else self.max_rounds
            if agent.rounds_taken >= limit:
                continue
            new_from_others = [m for m in self.history[agent.history_cursor + 1:] if m.sender_id != agent.agent_id]
            if agent.history_cursor == -1 or new_from_others:
                agent.task = asyncio.create_task(self._agent_respond(agent))
    
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
        self.current_query = ""

        for agent in self.agents:
            agent.formatted_history = []
            agent.history_cursor = -1
            agent.rounds_taken = 0
            agent.done = False