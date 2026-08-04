from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.skip_reply import SkipReplyTool
from nanobot.bus.queue import MessageBus
from nanobot.channels.slack.runtime import SlackChannel, SlackConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.runtime_context import RuntimeContextBlock, wrap_runtime_context_lines
from nanobot.session.manager import SessionManager
from nanobot.utils.llm_runtime import LLMRuntime


class DummyProvider:
    def __init__(self, response: LLMResponse):
        self._response = response

    async def generate_response(self, *args, **kwargs):
        return self._response

    def get_default_model(self) -> str:
        return "dummy"

    def supports_thinking(self) -> bool:
        return False

    def supports_concurrent_tools(self) -> bool:
        return False

    def can_resume_conversation_state(self, state, model):
        return False


@pytest.mark.asyncio
async def test_skip_reply_tool_hidden_without_flag():
    registry = ToolRegistry()
    registry.register(SkipReplyTool())
    token = bind_request_context(
        RequestContext(channel="slack", chat_id="C1", enabled_tools=set()),
    )
    try:
        names = [schema["function"]["name"] for schema in registry.get_definitions()]
    finally:
        reset_request_context(token)
    assert "skip_reply" not in names


@pytest.mark.asyncio
async def test_skip_reply_tool_visible_with_flag():
    registry = ToolRegistry()
    registry.register(SkipReplyTool())
    token = bind_request_context(
        RequestContext(channel="slack", chat_id="C1", enabled_tools={"skip_reply"}),
    )
    try:
        names = [schema["function"]["name"] for schema in registry.get_definitions()]
    finally:
        reset_request_context(token)
    assert "skip_reply" in names


@pytest.mark.asyncio
async def test_runner_stops_on_skip_reply(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(SkipReplyTool())
    provider = DummyProvider(
        LLMResponse(
            content="",
            tool_calls=[ToolCallRequest(id="call1", name="skip_reply", arguments={})],
        )
    )
    runtime = LLMRuntime.capture(provider, "dummy", context_window_tokens=1024)
    runner = AgentRunner()
    token = bind_request_context(
        RequestContext(channel="slack", chat_id="C1", enabled_tools={"skip_reply"}),
    )
    try:
        result = await runner.run(
            AgentRunSpec(
                initial_messages=[{"role": "user", "content": "hi"}],
                tools=registry,
                runtime=runtime,
                max_iterations=2,
                max_tool_result_chars=4000,
                workspace=tmp_path,
            )
        )
    finally:
        reset_request_context(token)
    assert result.stop_reason == "skip_reply"
    assert result.final_content is None


def test_slack_sender_context_and_participated_thread(tmp_path: Path):
    bus = MessageBus()
    sessions = SessionManager(tmp_path / "workspace")
    sessions.save(sessions.get_or_create("slack:C123:999.888"))
    channel = SlackChannel(
        SlackConfig(sender_names={"U123": "Nico"}, process_participated_threads=True),
        bus,
        session_manager=sessions,
    )

    blocks = channel._sender_context_blocks("U123")
    assert blocks == [
        RuntimeContextBlock(
            source="slack_sender",
            content=wrap_runtime_context_lines([
                "The following Slack message is from Nico.",
                "Slack sender ID: U123.",
            ]),
        )
    ]
    assert channel._has_participated_thread_session("C123", "999.888") is True
