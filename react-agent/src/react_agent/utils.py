"""Utility & helper functions."""

from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name."""
    provider, model = fully_specified_name.split("/", maxsplit=1)
    if provider == "ollama":
        return ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            format="json"  # Enable JSON mode for tool calling
        )
    raise ValueError(f"Unsupported provider: {provider}")
