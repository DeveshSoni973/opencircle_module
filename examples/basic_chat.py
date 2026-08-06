import os
from dotenv import load_dotenv

load_dotenv()

from opencircle_module.models import Agent
from opencircle_module.orchestrator import GroupChat

def main():
    # Make sure you have your API keys set in your environment!
    # For example:
    # os.environ["GROQ_API_KEY"] = "your_key_here"

    agent1 = Agent(
        name="Alice (Creative)",
        model="openai/gpt-oss-120b",
        provider="groq",
        temperature=0.8
    )
    agent2 = Agent(
        name="Bob (Technical)",
        model="qwen/qwen3.6-27b",
        provider="groq",
        temperature=0.3,
        max_tokens=4096
    )
    chat = GroupChat(
        agents=[agent1, agent2],
        system_prompt="default",
        max_rounds=3
    )

    print("Starting Group Chat...")
    print("-" * 50)
    query = "What is the best way to structure a modern Python project?"
    print(f"User: {query}\n")
    history = chat.run_sync(query)
    print("-" * 50)
    print("Chat finished! Here is the transcript:")
    print(chat.get_transcript())

if __name__ == "__main__":
    main()