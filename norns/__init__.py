"""Norns SDK — Python client for the Norns durable agent runtime."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("norns-sdk")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from norns.agent import Agent, tool
from norns.client import GardDestroyed, JoinError, KeyRevoked, Norns, NornsClient
from norns.models import (
    AgentResponse,
    ConversationResponse,
    EventResponse,
    MessageResult,
    RunResponse,
    StreamEvent,
    WaitingFor,
)

__all__ = [
    "__version__",
    "Norns",
    "NornsClient",
    "Agent",
    "tool",
    "GardDestroyed",
    "JoinError",
    "KeyRevoked",
    "AgentResponse",
    "ConversationResponse",
    "EventResponse",
    "MessageResult",
    "RunResponse",
    "StreamEvent",
    "WaitingFor",
]
