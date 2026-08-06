import os
import sys
from opencircle_module.models import Agent, Message
from opencircle_module.orchestrator import GroupChat

def on_message_handler(msg: Message):
    """Callback triggered whenever a new message is added to the history."""
    if not msg.is_user:
        # Display agent responses as they come in
        print(f"\n[{msg.sender_id}] >>> {msg.content}")

def on_round_complete_handler(agent: Agent, response: str | None):
    """Callback triggered when an agent finishes their turn (including when they choose to be silent)."""
    if response is None or "[SILENT]" in response:
        print(f"  (Agent {agent.name} decided to remain silent)")

def main():
    # Example using Anthropic and Gemini
    agent_claude = Agent(
        name="Claude",
        model="claude-3-haiku-20240307",
        provider="anthropic",
        temperature=0.7
    )
    
    agent_gemini = Agent(
        name="Gemini",
        model="gemini-1.5-flash",
        provider="google",
        temperature=0.7
    )

    chat = GroupChat(
        agents=[agent_claude, agent_gemini],
        system_prompt="debate", # Uses the 'debate.txt' prompt
        max_rounds=5,
        on_message=on_message_handler,
        on_round_complete=on_round_complete_handler
    )

    print("Starting Debate...")
    query = "Is it better to use strict type hinting in Python, or rely on duck typing?"
    print(f"User: {query}\n")
    
    print("Agents are thinking...")
    # Chat output is handled dynamically by our callbacks!
    chat.run_sync(query)

if __name__ == "__main__":
    main()
