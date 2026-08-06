import os
from opencircle_module.models import Agent
from opencircle_module.orchestrator import GroupChat

def main():
    # Make sure you have your API keys set in your environment!
    # For example:
    # os.environ["GROQ_API_KEY"] = "your_key_here"
    # os.environ["OPENAI_API_KEY"] = "your_key_here"

    # Create a couple of agents with different roles
    # Note: These use Groq and OpenAI providers, adjust according to what you have available
    agent1 = Agent(
        name="Alice (Creative)",
        model="llama3-8b-8192",
        provider="groq",
        temperature=0.8
    )
    
    agent2 = Agent(
        name="Bob (Technical)",
        model="gpt-3.5-turbo",
        provider="openai",
        temperature=0.3
    )
    
    # Initialize the group chat
    chat = GroupChat(
        agents=[agent1, agent2],
        system_prompt="default", # Or "creative", "technical", "debate"
        max_rounds=3
    )

    print("Starting Group Chat...")
    print("-" * 50)
    
    query = "What is the best way to structure a modern Python project?"
    print(f"User: {query}\n")
    
    # Run the chat synchronously
    # We can also iterate over the transcript
    history = chat.run_sync(query)
    
    print("-" * 50)
    print("Chat finished! Here is the transcript:")
    print(chat.get_transcript())

if __name__ == "__main__":
    main()
