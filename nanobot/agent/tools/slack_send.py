"""Slack send tool — send messages to Slack channels and register thread participation."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema

SLACK_API = "https://slack.com/api"


def _slack_post(token: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SLACK_API}/{endpoint}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def _resolve_channel_id(token: str, name: str) -> str:
    """Resolve a channel name (with or without #) to its Slack channel ID."""
    name = name.lstrip("#")
    cursor: str | None = None
    while True:
        params: dict[str, str] = {
            "limit": "200",
            "exclude_archived": "true",
            "types": "public_channel,private_channel",
        }
        if cursor:
            params["cursor"] = cursor
        query = "&".join(f"{k}={urllib.parse.quote(v)}" for k, v in params.items())
        req = urllib.request.Request(
            f"{SLACK_API}/conversations.list?{query}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data: dict[str, Any] = json.loads(resp.read())
        if not data.get("ok"):
            raise RuntimeError(f"conversations.list error: {data.get('error')}")
        for ch in data.get("channels") or []:
            if ch.get("name") == name:
                return str(ch["id"])
        cursor = (data.get("response_metadata") or {}).get("next_cursor") or ""
        if not cursor:
            break
    raise RuntimeError(f"Slack channel '#{name}' not found")


@tool_parameters(
    tool_parameters_schema(
        channel=StringSchema(
            "Slack channel name with or without '#' (e.g. '#general' or 'general')"
        ),
        message=StringSchema("Message text to send"),
        required=["channel", "message"],
    )
)
class SlackSendTool(Tool):
    """Send a message to a Slack channel and register the thread so replies are processed."""

    def __init__(self, bot_token: str = "", sessions: Any = None) -> None:
        self._bot_token = bot_token
        self._sessions = sessions

    @classmethod
    def _load_slack_config(cls) -> dict[str, Any]:
        """Read Slack config from the nanobot config file."""
        import os
        config_path = os.path.expanduser("~/.nanobot/config.json")
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            return cfg.get("channels", {}).get("slack", {})
        except Exception:
            return {}

    @classmethod
    def create(cls, ctx: ToolContext) -> "SlackSendTool":
        slack_cfg = cls._load_slack_config()
        bot_token: str = slack_cfg.get("botToken", "") or slack_cfg.get("bot_token", "")
        return cls(bot_token=bot_token, sessions=ctx.sessions)

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        slack_cfg = cls._load_slack_config()
        return bool(slack_cfg.get("enabled", False))

    @property
    def name(self) -> str:
        return "slack_send"

    @property
    def description(self) -> str:
        return (
            "Send a message to a Slack channel. "
            "The bot will be registered as a participant in the resulting thread, "
            "so replies to that message will be processed automatically."
        )

    async def execute(self, channel: str, message: str, **kwargs: Any) -> str:  # type: ignore[override]
        if not self._bot_token:
            return ToolResult.error("Slack bot token not configured")

        try:
            channel_id = _resolve_channel_id(self._bot_token, channel)
        except RuntimeError as e:
            return ToolResult.error(str(e))

        try:
            result = _slack_post(
                self._bot_token,
                "chat.postMessage",
                {"channel": channel_id, "text": message},
            )
        except Exception as e:
            return ToolResult.error(f"Slack API error: {e}")

        if not result.get("ok"):
            return ToolResult.error(f"Slack error: {result.get('error')}")

        ts: str = result.get("ts", "")

        # Register thread participation so replies are processed by the bot.
        # The canonical session key matches what SlackChannel uses for threads:
        # slack:{channel_id}:{thread_ts}
        if ts and self._sessions is not None:
            session_key = f"slack:{channel_id}:{ts}"
            try:
                self._sessions.get_or_create(session_key)
            except Exception:
                pass  # Non-fatal: message was sent, participation tracking failed

        channel_name = channel.lstrip("#")
        return f"Message sent to #{channel_name} (ts: {ts})"
