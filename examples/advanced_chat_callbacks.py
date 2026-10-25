import asyncio
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from opencircle_module import Agent, GroupChat, TurnStatus

load_dotenv()


class Colors:
    USER = "\033[92m"
    SYSTEM = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def setup_logging():
    """Create logs directory with timestamped session folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("logs") / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    return {
        "session_dir": session_dir,
        "full_history": session_dir / "full_history.jsonl",
        "per_agent_dir": session_dir / "agents",
    }


def log_message(log_files, msg, registry):
    """Append message to full history and per-agent log."""
    name = registry.get_name(msg.sender_id)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "sender_id": msg.sender_id,
        "name": name,
        "content": msg.content,
    }

    # Full history (JSONL)
    with open(log_files["full_history"], "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Per-agent log (text)
    if msg.sender_id != "usr":
        agent_log = log_files["per_agent_dir"] / f"{name}.txt"
        with open(agent_log, "a", encoding="utf-8") as f:
            f.write(f"[{entry['timestamp']}] {entry['content']}\n\n")


def on_message_factory(log_files, registry):
    """Create on_message callback with logging."""
    def on_message(msg):
        name = registry.get_name(msg.sender_id)
        print(f"{Colors.BOLD}{name}:{Colors.RESET} {msg.content}")
        log_message(log_files, msg, registry)
    return on_message


def on_round_complete_factory(log_files, registry):
    """Create on_round_complete callback with logging.

    GroupChat calls on_round_complete(agent, result) once per agent turn,
    where result is a TurnResult with status spoke / silent / error.
    """
    def on_round_complete(agent, result):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent.name,
            "status": result.status.value,
            "content": result.content,
            "error": result.error,
        }

        round_log = log_files["session_dir"] / "rounds.jsonl"
        with open(round_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if result.status == TurnStatus.SILENT:
            print(f"{Colors.SYSTEM}({agent.name} stayed silent){Colors.RESET}")
        elif result.status == TurnStatus.ERROR:
            print(f"{Colors.SYSTEM}({agent.name} failed: {result.error}){Colors.RESET}")
    return on_round_complete


async def terminal_gc():
    agents = [
        Agent(name="Llama", model="openai/gpt-oss-120b", provider="groq"),
        Agent(name="Qwen", model="qwen/qwen3.6-27b", provider="groq", max_tokens=4096),
    ]

    chat = GroupChat(
        agents=agents,
        system_prompt="default",
        max_rounds=15,
    )

    log_files = setup_logging()
    log_files["per_agent_dir"].mkdir(exist_ok=True)

    # Set callbacks after init so we can capture registry
    chat.on_message = on_message_factory(log_files, chat.registry)
    chat.on_round_complete = on_round_complete_factory(log_files, chat.registry)

    print(f"{Colors.BOLD}⭕ OpenCircle Group Chat Started{Colors.RESET}")
    print(f"{Colors.SYSTEM}Participants: {', '.join(a.name for a in agents)}{Colors.RESET}")
    print(f"{Colors.SYSTEM}Logs: {log_files['session_dir']}{Colors.RESET}\n")

    while True:
        try:
            user_msg = input(f"{Colors.USER}{Colors.BOLD}You:{Colors.RESET} ")
            if user_msg.lower() in ("exit", "quit"):
                break
            if not user_msg.strip():
                continue

            # Log user message
            from opencircle_module.models import Message
            user_entry = Message(sender_id="usr", content=user_msg)
            log_message(log_files, user_entry, chat.registry)

            await chat.run(user_msg)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n{Colors.BOLD}Error:{Colors.RESET} {e}")


if __name__ == "__main__":
    try:
        asyncio.run(terminal_gc())
    except KeyboardInterrupt:
        pass