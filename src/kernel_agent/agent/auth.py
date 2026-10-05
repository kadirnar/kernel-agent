"""How the agent sessions authenticate (``--auth``), and Claude usage limits.

* ``subscription``: every session runs on the Claude Code login (a Claude
  subscription). The variables that would make Claude Code use an API key or a
  cloud provider instead (:data:`SCRUBBED`) are blanked in the agent
  environment, a login must exist before the run starts (:func:`preflight`:
  the credentials file, ``CLAUDE_CODE_OAUTH_TOKEN`` or the macOS keychain item,
  checked for presence only), and a session whose init message reports an
  ``apiKeySource`` other than ``none`` is stopped before its first request
  (:class:`AuthError`). USD in ``costs.json`` is then notional.
* ``api``: an API key, bearer token or cloud provider must be set; a session that
  falls back to the Claude Code login is stopped.
* ``auto`` (default): whatever Claude Code finds, as before; nothing is checked.

Every session's ``apiKeySource`` and billing (:func:`billing`) go to ``costs.json``.

A session that ends at a usage or rate limit (:class:`LimitWatch`: a rejected
``rate_limit_event``, an assistant message with ``error: rate_limit``, an error
result with HTTP 429 or the limit text) is not a failure: the orchestrator waits
until the limit resets (:func:`wait_seconds`) and resumes it. Nothing here reads
or prints a credential; only variable names and file paths are reported.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AUTO, SUBSCRIPTION, API = "auto", "subscription", "api"
MODES = (AUTO, SUBSCRIPTION, API)

API_KEY_VARS = ("ANTHROPIC_API_KEY",)
#: Credentials without an API key: Claude Code reports ``apiKeySource: none`` for them too.
TOKEN_VARS = ("ANTHROPIC_AUTH_TOKEN",)
PROVIDER_VARS = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    "CLAUDE_CODE_USE_MANTLE",
)
#: Blanked in the agent environment with ``--auth subscription``.
SCRUBBED = API_KEY_VARS + TOKEN_VARS + PROVIDER_VARS
OAUTH_TOKEN_VAR = "CLAUDE_CODE_OAUTH_TOKEN"  # a subscription token (`claude setup-token`)
CREDENTIALS_FILE = ".credentials.json"
KEYCHAIN_SERVICE = "Claude Code-credentials"

#: Usage limits: the first wait without a reset time, its cap, the slack after a reset
#: time, and the waits per session before the run stops starting agents.
BACKOFF_S = 60.0
MAX_BACKOFF_S = 1800.0
RESET_SLACK_S = 60.0
MAX_LIMIT_WAITS = 8
RESUME_PROMPT = (
    "The usage limit has reset. Continue the task you were working on when it was reached; "
    "do not repeat work that is already complete."
)
_LIMIT_TEXT = re.compile(
    r"usage limit|hit your (?:[\w'-]+ ){0,3}limit|rate[ _-]?limit|too many requests"
    r"|(?:session|weekly|opus|sonnet|5-hour|five[ _-]hour|seven[ _-]day) limit",
    re.I,
)
_LEGACY_RESET = re.compile(r"limit reached\|(\d{9,11})")  # "Claude AI usage limit reached|<t>"


class AuthError(SystemExit):
    """A session would not authenticate the way ``--auth`` asks: the run stops."""


# ------------------------------------------------------------------ environment


def present(env: Mapping[str, str], names: tuple[str, ...]) -> list[str]:
    """The ``names`` set (non-empty) in ``env``; never their values."""
    return [n for n in names if env.get(n)]


def scrub(env: dict[str, str], environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """``env`` with every :data:`SCRUBBED` variable set in ``environ`` (default: this
    process) or ``env`` blanked. The Claude Agent SDK starts Claude Code with this
    process's environment updated by ``env``, so a blank value is what removes one."""
    environ = os.environ if environ is None else environ
    return {**env, **{n: "" for n in present({**environ, **env}, SCRUBBED)}}


def _session_env(env: Mapping[str, str]) -> dict[str, str]:
    """What a Claude Code session started with the agent environment ``env`` sees."""
    return {**os.environ, **env}


def login(environ: Mapping[str, str] | None = None, home: Path | None = None) -> str | None:
    """Where a Claude Code login exists (a variable name, a file path or the keychain), or
    None. Only presence is checked: nothing is read."""
    environ = os.environ if environ is None else environ
    if environ.get(OAUTH_TOKEN_VAR):
        return OAUTH_TOKEN_VAR
    config = environ.get("CLAUDE_CONFIG_DIR") or str((home or Path.home()) / ".claude")
    credentials = Path(config).expanduser() / CREDENTIALS_FILE
    if credentials.is_file():
        return str(credentials)
    if sys.platform == "darwin" and _keychain_item():
        return "macOS keychain"
    return None


def _keychain_item() -> bool:
    """Whether the macOS keychain has Claude Code's item (without ``-w``: no secret)."""
    try:
        done = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def preflight(mode: str, environ: Mapping[str, str] | None = None, home: Path | None = None) -> str:
    """Check before a run starts that sessions can authenticate as ``mode`` asks; returns
    a line for the log. Raises SystemExit with what to do otherwise."""
    environ = os.environ if environ is None else environ
    if mode not in MODES:
        raise SystemExit(f"--auth must be one of {', '.join(MODES)}, not {mode!r}")
    if mode == SUBSCRIPTION:
        where = login(environ, home)
        if where is None:
            raise SystemExit(
                "--auth subscription: no Claude Code login found (~/.claude/"
                f"{CREDENTIALS_FILE}, {OAUTH_TOKEN_VAR} or the macOS keychain). Run `claude` "
                "and /login with your Claude subscription (or `claude setup-token` and set "
                f"{OAUTH_TOKEN_VAR}), or use --auth api."
            )
        ignored = present(environ, SCRUBBED)
        note = f"; ignoring {', '.join(ignored)} from the environment" if ignored else ""
        return f"auth: subscription (Claude Code login: {where}){note}"
    if mode == API:
        found = present(environ, API_KEY_VARS + TOKEN_VARS + PROVIDER_VARS)
        if not found:
            raise SystemExit(
                f"--auth api: none of {', '.join(API_KEY_VARS + TOKEN_VARS)} or "
                "CLAUDE_CODE_USE_* is set; export one, or use --auth subscription"
            )
        return f"auth: api ({', '.join(found)})"
    found = present(environ, SCRUBBED)
    return f"auth: auto ({', '.join(found) or 'the Claude Code login'}, as Claude Code finds it)"


def session_problem(mode: str, source: str | None, env: Mapping[str, str]) -> str | None:
    """Why a session whose init message reports ``apiKeySource`` ``source`` may not run as
    ``mode`` asks (None: it may); ``env``: the agent environment."""
    if source is None:  # not reported: nothing to check
        return None
    if mode == SUBSCRIPTION and source != "none":
        return (
            f"--auth subscription: Claude Code reports API key source {source!r} instead of "
            "the Claude Code login; the session was stopped before its first request. Remove "
            "that key or use --auth api."
        )
    if mode == API and billing(source, env) == SUBSCRIPTION:
        return (
            "--auth api: Claude Code reports no API key (apiKeySource 'none'): the session "
            "would run on the Claude Code login; it was stopped before its first request"
        )
    return None


def billing(source: str | None, env: Mapping[str, str]) -> str | None:
    """What a session with ``apiKeySource`` ``source`` is billed to: ``subscription`` (the
    Claude Code login), ``api`` (an API key, bearer token or cloud provider) or None
    (not reported); ``env``: the agent environment."""
    if source is None:
        return None
    if source != "none":
        return API
    return API if present(_session_env(env), TOKEN_VARS + PROVIDER_VARS) else SUBSCRIPTION


def usd_note(costs: Mapping[str, Any]) -> str:
    """``notional (subscription)`` when sessions of ``costs.json`` ran on the Claude
    subscription (their USD is Claude Code's estimate, not a bill), else ``""``."""
    rows = [c for c in costs.values() if isinstance(c, dict)]
    subs = sum(c.get("billing") == SUBSCRIPTION for c in rows)
    if not subs:
        return ""
    some = "" if subs == len(rows) else f" for {subs} of {len(rows)} sessions"
    return "notional (subscription)" + some


# ------------------------------------------------------------------ usage limits


@dataclass
class UsageLimit:
    """A session stopped at a usage or rate limit."""

    message: str  # what Claude Code said
    resets_at: float | None = None  # unix time the limit resets, when Claude Code says
    kind: str | None = None  # five_hour, seven_day, ... (the rate_limit_event's type)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LimitWatch:
    """What a session's messages say about usage limits. A rejected ``rate_limit_event``
    alone is no limit (extra usage may cover it): the session must also end in an error."""

    hit: bool = False
    errored: bool = False
    resets_at: float | None = None
    kind: str | None = None
    texts: list[str] = field(default_factory=list)

    def see(self, message: Any) -> None:
        # here, not at the top: status and the dashboard import this module (usd_note)
        from claude_agent_sdk import AssistantMessage, RateLimitEvent, ResultMessage, TextBlock

        if isinstance(message, RateLimitEvent):
            info = message.rate_limit_info
            if info.status == "rejected":
                self.hit = True
                self.resets_at = info.resets_at or self.resets_at
                self.kind = info.rate_limit_type or self.kind
        elif isinstance(message, AssistantMessage) and message.error == "rate_limit":
            self.hit = True
            self.texts += [b.text for b in message.content if isinstance(b, TextBlock)]
        elif isinstance(message, ResultMessage) and message.is_error:
            self._error(message.api_error_status, [message.result or "", *(message.errors or [])])

    def failed(self, exc: BaseException) -> None:
        """The SDK raised ``exc`` (it does after an error result: Claude Code exits 1)."""
        self._error(getattr(exc, "api_error_status", None), [str(exc)])

    def _error(self, status: int | None, texts: list[str]) -> None:
        self.errored = True
        texts = [t for t in texts if t]
        if status == 429 or any(_LIMIT_TEXT.search(t) for t in texts):
            self.hit = True
            self.texts += texts

    def limit(self) -> UsageLimit | None:
        """The limit the session stopped at, or None."""
        if not (self.hit and self.errored):
            return None
        resets = self.resets_at
        for text in self.texts:
            if resets is None and (m := _LEGACY_RESET.search(text)):
                resets = float(m.group(1))
        message = next((t for t in self.texts if t.strip()), "usage limit reached")
        return UsageLimit(message=message.strip()[:300], resets_at=resets, kind=self.kind)


def wait_seconds(limit: UsageLimit, waits: int, now: float) -> float:
    """Seconds to wait before resuming a session stopped at ``limit`` after ``waits``
    earlier waits: until its reset time (plus :data:`RESET_SLACK_S`) when known, and at
    least an exponential back-off from :data:`BACKOFF_S` capped at :data:`MAX_BACKOFF_S`."""
    backoff = min(BACKOFF_S * 2**waits, MAX_BACKOFF_S)
    if limit.resets_at is None:
        return backoff
    return max(float(limit.resets_at) - now + RESET_SLACK_S, backoff)
