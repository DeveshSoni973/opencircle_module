# OpenCircle Module

> **A decentralized, multi-agent group chat system where AI agents self-decide when to speak.**

Traditional multi-agent frameworks rely on a central "manager" or "router" LLM to decide which agent should speak next. **OpenCircle** flips this paradigm: all agents execute in parallel, observe the same conversation history, and independently decide whether to contribute or stay `[SILENT]`. 

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Concepts](#core-concepts)
   - [Agents](#agents)
   - [The Orchestrator (GroupChat)](#the-orchestrator-groupchat)
   - [System Prompts](#system-prompts)
6. [Supported Providers](#supported-providers)
7. [Advanced Usage](#advanced-usage)
   - [Callbacks & Streaming (UI Integration)](#callbacks--streaming-ui-integration)
   - [Context Window Management](#context-window-management)
   - [Observability & Tracing (LangSmith)](#observability--tracing-langsmith)

---

## Overview & Architecture

OpenCircle is built on a **Parallel Single-Phase Execution** model. 

In every conversational round, the Orchestrator takes a snapshot of the canonical history and broadcasts it to all agents simultaneously. Every agent processes the context and decides to either return a message or output `[SILENT]`. If an agent (or all agents) feel the user's query is fully resolved, they can output `[DONE]` to terminate the loop.

### How it Works (Data Flow)

```text
[ User Query ] ──> Added to Canonical History (UUID-based)
                            │
        ┌───────────────────┴───────────────────┐
        │       ROUND START (Parallel)          │
        ├──────────────┬──────────────┬─────────┤
        │              │              │         │
    ┌───▼───┐      ┌───▼───┐      ┌───▼───┐     │
    │ Agent │      │ Agent │      │ Agent │ ... │
    │   A   │      │   B   │      │   C   │     │
    └───┬───┘      └───┬───┘      └───┬───┘     │
        │              │              │         │
    [SILENT]      "I agree!"      [SILENT]      │
        │              │              │         │
        └──────────────┼──────────────┘         │
                       │                        │
             Append to History ─────────────────┘
                       │
             Next Round or Exit if [DONE]
```

---

## Key Features

- **Decentralized Control**: No "Manager LLM" bottleneck. Agents act autonomously.
- **True Parallel Execution**: Fast response times. All agents are queried asynchronously via `asyncio`.
- **Provider-Agnostic**: Mix and match models from OpenAI, Anthropic, Google, Groq, NVIDIA, and OpenRouter in the same chat.
- **Canonical UUID History**: Clean internal data structure. Names are resolved at runtime via the `AgentRegistry`, preventing LLMs from getting confused by naming collisions.
- **Dynamic Context Building**: Automatically handles provider-specific quirks (e.g., Anthropic's strict user/assistant alternation requirements).

---

## Installation

Ensure you are using Python 3.9+.

```bash
# Create a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

---

## Quick Start

Here is a simple script putting two agents (Llama and Qwen) in a room to debate a topic.

```python
import asyncio
import os
from dotenv import load_dotenv
from opencircle_module import Agent, GroupChat

load_dotenv()

async def main():
    # 1. Define your agents
    agents = [
        Agent(
            name="Llama",
            model="llama-3.3-70b-versatile",
            provider="groq",
            api_key=os.environ.get("GROQ_API_KEY"),
        ),
        Agent(
            name="Qwen",
            model="qwen-32b",
            provider="groq",
            api_key=os.environ.get("GROQ_API_KEY"),
        ),
    ]
    
    # 2. Initialize the chat room
    chat = GroupChat(
        agents=agents, 
        system_prompt="debate",  # Uses system_prompts/debate.txt
        max_rounds=3
    )
    
    # 3. Run the conversation
    print("Starting chat...")
    history = await chat.run("Should I learn Machine Learning or Backend Engineering first?")
    
    # 4. Print the human-readable transcript
    print("\n--- TRANSCRIPT ---")
    print(chat.get_transcript())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Core Concepts

### Agents
An `Agent` represents a single AI persona. You configure them with a `name`, a specific `model`, and the API `provider`. Each agent operates with independent `temperature` and `max_tokens` settings.

### The Orchestrator (GroupChat)
The `GroupChat` class manages the lifecycle of the conversation. It holds the `AgentRegistry`, maintains the `history` of `Message` objects, and handles the `asyncio` loop that queries agents.

### System Prompts
System prompts dictate how agents behave in the group setting. They are stored as `.txt` files in `opencircle_module/system_prompts/`. 

The `SystemPromptLoader` dynamically injects variables into these text files at runtime:
- `$agent_name`: The current agent's name.
- `$other_agents`: A comma-separated list of the *other* agents in the room.
- `$user_query`: The initial prompt that started the conversation.

**Included Templates:**
- `default`: Basic conversational rules.
- `debate`: Instructs agents to play devil's advocate.
- `creative`: Encourages brainstorming and "yes, and..." thinking.
- `technical`: Focuses on implementation details and edge cases.

*To add your own, simply drop a new `my_prompt.txt` into the `system_prompts` directory and pass `system_prompt="my_prompt"` to `GroupChat`.*

---

## 🔌 Supported Providers

OpenCircle abstracts the complexities of different LLM SDKs. To use a provider, simply set its environment variable and pass its string identifier to the `Agent`.

| Provider String | Required Environment Variable | Notes |
| :--- | :--- | :--- |
| `openai` | `OPENAI_API_KEY` | Standard OpenAI compatible. |
| `anthropic` | `ANTHROPIC_API_KEY` | Uses `anthropic` SDK. Extracts system prompts automatically. |
| `groq` | `GROQ_API_KEY` | Auto-strips `<think>` tags from reasoning models. |
| `google` | `GOOGLE_API_KEY` *(or GEMINI_API_KEY)* | Uses `google-generativeai`. Maps roles to `user`/`model`. |
| `nvidia` | `NVIDIA_API_KEY` | Uses standard OpenAI SDK pointing to NVIDIA NIM URL. |
| `openrouter` | `OPENROUTER_API_KEY` | Includes standard OpenRouter HTTP headers. |

---

## Advanced Usage

### Callbacks & Streaming (UI Integration)
If you are building a frontend (like a Gradio, Streamlit, or web app) and want to see messages as they are generated, you can pass callback functions to `GroupChat`:

```python
def handle_new_message(msg):
    # msg is a Message object
    print(f"[{msg.sender_id}] just spoke: {msg.content}")

def handle_round_complete(result):
    print(f"Round {result.round_num} finished. {len(result.messages_added)} messages added.")

chat = GroupChat(
    agents=agents,
    on_message=handle_new_message,
    on_round_complete=handle_round_complete
)
```

### Context Window Management
To prevent long conversations from crashing the LLM APIs, you can limit the amount of history sent to the agents:

```python
chat = GroupChat(
    agents=agents,
    max_history_messages=10,        # Only send the last 10 messages to the LLM
    include_silent_in_history=False # Keep [SILENT] messages out of the context window
)
```


### Observability & Tracing (LangSmith)

OpenCircle has built-in support for **LangSmith** to provide deep visibility into every agent's thought process, API latency, and token usage. This allows you to debug exactly why an agent chose to speak or why it stayed silent.

#### 1. Setup
Install the LangSmith client and set your environment variables:

```bash
uv pip install langsmith
```

```text
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=opencircle_dev
```

#### 2. Usage
Tracing is automatically enabled if the environment variables are present. You can also pass the configuration explicitly when initializing the chat:

```python
chat = GroupChat(
    agents=agents,
    langchain_api_key="ls__your_key_here", # Optional: overrides environment
    langchain_project="my_gc_project"
)
```

#### 3. What you see in the dashboard
Once enabled, every round of chat creates a structured trace:

- `agent_turn`: A high-level span showing which specific agent is attempting to respond.

- `llm_[provider]`: A detailed span showing the exact raw JSON history sent to the provider (OpenAI/Groq/etc.) and the raw text response received.

This makes it easy to spot formatting errors, token limits, or logic issues in real-time.