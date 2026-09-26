"""
Helpers for generating Utterances (for AI Speakers) and Supports (for Assistants) with convokit.genai.

A generation config is a dict with the keys:

- "prompt": the prompt that generates the message (required)
- "model": the model name, e.g. "gemini-2.5-flash" or "gpt-4o-mini" (required unless a client is given)
- "provider": the convokit.genai provider, "gemini" / "gpt" / "local" (inferred from "model" if missing)
- "temperature": sampling temperature (optional; the client's default is used if missing)
"""

import time
import uuid
from typing import Dict, Optional

from .aiUtil import transcript_sort_key


def infer_provider(model: str) -> str:
    model = model.lower()
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return "gpt"
    raise ValueError(
        "Cannot infer the provider for model {!r}; set config['provider'] "
        "to one of 'gemini', 'gpt', 'local'.".format(model)
    )


def get_client(config: Dict, client=None, config_manager=None):
    """
    Get the convokit.genai LLMClient to generate with: `client` if given, otherwise one built from
    config["model"] / config["provider"].
    """
    if client is not None:
        return client
    try:
        from convokit.genai import GenAIConfigManager, get_llm_client
    except ImportError as e:
        raise ImportError(
            "GenAI dependencies not available. Please install via `pip install convokit[genai]`."
        ) from e
    model = config.get("model")
    if not model:
        raise ValueError("config must have a 'model' to generate with, or pass a client.")
    provider = config.get("provider") or infer_provider(model)
    return get_llm_client(provider, config_manager or GenAIConfigManager(), model=model)


def call_llm(client, config: Dict, prompt: str) -> str:
    kwargs = {"temperature": config["temperature"]} if "temperature" in config else {}
    return client.generate(prompt, **kwargs).text.strip()


def build_prompt(prompt: str, transcript: Optional[str], instruction: str) -> str:
    parts = [prompt]
    if transcript is not None:
        parts.append("Conversation so far:\n" + transcript)
    parts.append(instruction)
    return "\n\n".join(parts)


def last_utterance_id(conversation) -> Optional[str]:
    utterances = list(conversation.iter_utterances())
    return max(utterances, key=transcript_sort_key).id if utterances else None


def last_support(conversation, reply_to: str):
    """
    The latest Support in the conversation replying to the utterance `reply_to`, or None.
    """
    supports = [s for s in conversation.supports if s.reply_to == reply_to]
    return max(supports, key=transcript_sort_key) if supports else None


def next_timestamp(conversation) -> int:
    """
    A timestamp that orders after every utterance and support in the conversation.
    """
    if conversation is not None:
        timestamps = [u.timestamp for u in conversation.iter_utterances()]
        timestamps += [s.timestamp for s in conversation.supports]
        numeric = [t for t in timestamps if isinstance(t, (int, float))]
        if numeric:
            return int(max(numeric)) + 1
    return int(time.time())


def new_id() -> str:
    return uuid.uuid4().hex
