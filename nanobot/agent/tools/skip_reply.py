"""Internal tool to intentionally suppress a visible final reply."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.schema import tool_parameters_schema


@tool_parameters(tool_parameters_schema(required=[]))
class SkipReplyTool(Tool):
    def __init__(self) -> None:
        self._skip_in_turn_var: ContextVar[bool] = ContextVar("skip_reply_in_turn", default=False)

    def start_turn(self) -> None:
        self._skip_in_turn = False

    @property
    def _skip_in_turn(self) -> bool:
        return self._skip_in_turn_var.get()

    @_skip_in_turn.setter
    def _skip_in_turn(self, value: bool) -> None:
        self._skip_in_turn_var.set(value)

    @property
    def name(self) -> str:
        return "skip_reply"

    @property
    def description(self) -> str:
        return (
            "Suppress the visible final reply for the current conversation turn. "
            "Use this only when the message was processed but no user-facing reply is necessary."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameter_schema

    async def execute(self, **kwargs: Any) -> str:
        ctx = current_request_context()
        if ctx is None or ctx.enabled_tools is None or self.name not in ctx.enabled_tools:
            return "skip_reply unavailable in this turn"
        self._skip_in_turn = True
        return "Visible reply skipped for this turn"
