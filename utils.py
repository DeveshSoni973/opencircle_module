"""Utility functions."""

import tiktoken


def count_tokens(text, model="gpt-4"):
    """Estimate token count for text."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_messages_tokens(messages, model="gpt-4"):
    """Estimate total tokens in message list."""
    total = 0
    for msg in messages:
        total += count_tokens(msg.content, model)
    return total


def summarize_conversation(messages, max_chars=500):
    """Create a brief summary of conversation for context window management."""
    if not messages:
        return ""
    
    summary_parts = []
    
    if messages:
        summary_parts.append(f"Started with: {messages[0].content[:100]}...")
    
    senders = set(m.sender_id for m in messages)
    summary_parts.append(f"Participants: {len(senders)}")
    
    if len(messages) >= 2:
        last = messages[-2:]
        summary_parts.append(f"Latest: {last[0].content[:80]}... -> {last[1].content[:80]}...")
    
    return " | ".join(summary_parts)