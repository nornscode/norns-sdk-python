"""Agent definition and tool decorator."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


@dataclass
class ToolDef:
    """A tool that an agent can call."""

    name: str
    description: str
    input_schema: dict
    handler: Callable
    side_effect: bool = False

    def to_registration(self) -> dict:
        """Convert to the wire format for worker registration."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "side_effect": self.side_effect,
        }


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effect: bool = False,
) -> ToolDef | Callable:
    """Decorator to define a tool from a function.

    Usage:
        @tool
        def search(query: str) -> str:
            '''Search the web.'''
            return results

        @tool(side_effect=True)
        def send_email(to: str, subject: str, body: str) -> str:
            '''Send an email.'''
            ...
    """

    def wrap(f: Callable) -> ToolDef:
        tool_name = name or f.__name__
        tool_desc = description or (f.__doc__ or "").strip() or tool_name
        schema = _infer_schema(f)

        return ToolDef(
            name=tool_name,
            description=tool_desc,
            input_schema=schema,
            handler=f,
            side_effect=side_effect,
        )

    if fn is not None:
        return wrap(fn)
    return wrap


def _infer_schema(fn: Callable) -> dict:
    """Infer JSON Schema from function type hints."""
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        type_hint = hints.get(param_name)
        json_type = _python_type_to_json(type_hint)
        properties[param_name] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json(type_hint: Any) -> str:
    """Map Python types to JSON Schema types."""
    if type_hint is None:
        return "string"

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # Handle Optional, Union, etc. by checking origin
    origin = getattr(type_hint, "__origin__", None)
    if origin is not None:
        # For List[X], Dict[X, Y], etc.
        return type_map.get(origin, "string")

    return type_map.get(type_hint, "string")


@dataclass
class Agent:
    """Declarative agent definition.

    Usage:
        agent = Agent(
            name="support-bot",
            model="anthropic/claude-sonnet-5",
            system_prompt="You are a support agent.",
            tools=[my_tool],
        )

    The model field is the model name as known to the LLM provider (e.g.
    "claude-sonnet-5", "gpt-4o"). Set the llm_provider on the
    Agent to tell the SDK which provider to use (defaults to "anthropic").
    """

    name: str
    model: str = "claude-sonnet-5"
    llm_provider: str = "anthropic"
    system_prompt: str = ""
    tools: list[ToolDef] = field(default_factory=list)
    mode: str = "task"
    checkpoint_policy: str = "on_tool_call"
    context_strategy: str = "sliding_window"
    context_window: int = 20
    max_steps: int = 50
    on_failure: str = "retry_last_step"
    # {"compact_at": input_tokens, "keep": messages}: fold older history into
    # a summary once a response reports compact_at tokens. Pair it with
    # context_strategy="none"; the summarisation is served by this worker.
    context_policy: dict | None = None

    def to_registration(self) -> dict:
        """Convert to the wire format for worker registration."""
        return {
            "name": self.name,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "mode": self.mode,
            "checkpoint_policy": self.checkpoint_policy,
            "context_strategy": self.context_strategy,
            "context_window": self.context_window,
            "max_steps": self.max_steps,
            "on_failure": self.on_failure,
            "context_policy": self.context_policy,
            "tools": [t.name for t in self.tools],
        }
