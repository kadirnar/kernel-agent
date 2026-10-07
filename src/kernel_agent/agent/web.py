"""Documentation lookups of agent sessions (issue #125): the domains ``WebFetch`` may
reach, and the record of every lookup.

Agent sessions have ``WebFetch`` / ``WebSearch`` unless ``--no-web``
(``runner.WEB_TOOLS``). :class:`Lookups` keeps them to documentation and records them:

* :meth:`Lookups.guard`: a PreToolUse hook (like ``runner.write_guard``) that lets
  ``WebFetch`` reach only the documentation / code / paper domains of :data:`DOMAINS`
  plus ``--web-domain`` (a host and its subdomains), and restricts ``WebSearch`` to the
  same domains (its ``allowed_domains``), so every result it shows can be fetched.
* :meth:`Lookups.see`: reads the session's messages and records each lookup: time,
  tool, URL or query, outcome (``ok``, ``denied``, ``error``), the HTTP code, and the
  size and sha256 of what the agent got back.

The orchestrator appends a session's lookups to ``research/sources.jsonl`` of the run
(:func:`record`), counts them in ``costs.json`` (:func:`summary`) and lists the sources
used in ``report.md`` (:func:`report_lines`). A lookup is a GET of a public page or a
search query; the prompts (``prompts.web_note``) say that pages are untrusted data and
that nothing of the run goes into a URL or a query.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from claude_agent_sdk import (
    AssistantMessage,
    HookJSONOutput,
    HookMatcher,
    Message,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from kernel_agent.workspace import append_jsonl, read_jsonl

FETCH, SEARCH = "WebFetch", "WebSearch"
#: Hosts ``WebFetch`` may reach by default, each with its subdomains: documentation,
#: reference code and papers (``knowledge/sources.md`` uses only these).
DOMAINS = (
    "docs.nvidia.com",
    "developer.nvidia.com",
    "nvidia.github.io",
    "github.com",
    "raw.githubusercontent.com",
    "triton-lang.org",
    "tilelang.com",
    "pytorch.org",
    "docs.flashinfer.ai",
    "arxiv.org",
    "openreview.net",
    "opencompute.org",
    "crfm.stanford.edu",
    "huggingface.co",
)
SOURCES_FILE = Path("research") / "sources.jsonl"  # in the run directory
REPORT_ROWS = 25  # sources listed in report.md


def _host(raw: str) -> str:
    """A domain as written by a person: lower case, no scheme, path or leading ``*.`` /
    ``www.``."""
    host = str(raw).strip().lower().split("://", 1)[-1].split("/", 1)[0]
    return host.removeprefix("*.").removeprefix("www.").rstrip(".")


def domains(extra: Iterable[str] = ()) -> list[str]:
    """:data:`DOMAINS` plus ``extra`` (``--web-domain``), normalised (:func:`_host`)."""
    return list(dict.fromkeys(h for h in map(_host, (*DOMAINS, *extra)) if h))


def allowed(url: str, hosts: Iterable[str]) -> bool:
    """Whether ``url`` is an http(s) URL on one of ``hosts`` or a subdomain of one."""
    try:
        parts = urlsplit(str(url).strip())
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not host or parts.username or parts.password:
        return False
    return any(host == h or host.endswith("." + h) for h in hosts)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _text(content: Any) -> str:
    """The text of a tool result's content (a string or a list of content blocks)."""
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts)


def _links(result: Any) -> int | None:
    """The number of result links of a WebSearch ``tool_use_result`` (None: unknown)."""
    if not isinstance(result, dict) or not isinstance(result.get("results"), list):
        return None
    n = 0
    for entry in result["results"]:
        content = entry.get("content") if isinstance(entry, dict) else None
        n += len(content) if isinstance(content, list) else 0
    return n


class Lookups:
    """The WebFetch / WebSearch calls of one agent session, appended to ``items``."""

    def __init__(self, items: list[dict[str, Any]], extra_domains: Iterable[str] = ()) -> None:
        self.items = items
        self.hosts = domains(extra_domains)
        self.pending: dict[str, dict[str, Any]] = {}
        self.denied: dict[str, str] = {}  # tool_use_id -> why the guard refused it

    def guard(self) -> HookMatcher:
        """PreToolUse hook: WebFetch to :attr:`hosts` only; WebSearch restricted to them."""
        hosts = self.hosts

        async def hook(data: Any, tool_use_id: str | None, context: Any) -> HookJSONOutput:
            tool_input = dict(data.get("tool_input") or {})
            if data.get("tool_name") == SEARCH:  # results only from fetchable domains
                asked = [_host(d) for d in tool_input.get("allowed_domains") or []]
                within = [d for d in asked if any(d == h or d.endswith("." + h) for h in hosts)]
                tool_input["allowed_domains"] = within or list(hosts)
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "updatedInput": tool_input,
                    }
                }
            url = str(tool_input.get("url") or "")
            if allowed(url, hosts):
                return {}
            why = (
                f"WebFetch may reach documentation, code and paper domains only: "
                f"{', '.join(hosts)} (and their subdomains); not {url or 'an empty URL'}"
            )
            if tool_use_id:
                self.denied[tool_use_id] = why
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": why,
                }
            }

        return HookMatcher(matcher=f"{FETCH}|{SEARCH}", hooks=[hook])

    def see(self, message: Message) -> None:
        """Record the lookups a message starts (tool calls) or finishes (tool results)."""
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock) and block.name in (FETCH, SEARCH):
                    item: dict[str, Any] = {"time": _now(), "tool": block.name}
                    if block.name == FETCH:
                        item["url"] = str(block.input.get("url") or "")
                        if block.input.get("offset"):
                            item["offset"] = block.input["offset"]
                    else:
                        item["query"] = str(block.input.get("query") or "")
                    item["status"] = "no result"  # until its result arrives
                    self.pending[block.id] = item
                    self.items.append(item)
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    found = self.pending.pop(block.tool_use_id, None)
                    if found is not None:
                        self._finish(found, block, message.tool_use_result)

    def _finish(self, item: dict[str, Any], block: ToolResultBlock, result: Any) -> None:
        text = _text(block.content)
        item["chars"] = len(text)
        item["sha256"] = hashlib.sha256(text.encode()).hexdigest()
        if why := self.denied.pop(block.tool_use_id, None):
            item.update(status="denied", reason=why[:300])
            return
        item["status"] = "error" if block.is_error else "ok"
        if block.is_error:
            item["reason"] = text[:300]
        if isinstance(result, dict):
            if isinstance(result.get("code"), int):
                item["code"] = result["code"]
                if not 200 <= result["code"] < 300:
                    item["status"] = "error"
            if isinstance(result.get("bytes"), int):
                item["bytes"] = result["bytes"]
        if item["tool"] == SEARCH and (links := _links(result)) is not None:
            item["results"] = links


def summary(items: list[dict[str, Any]]) -> dict[str, int]:
    """A session's lookups for ``costs.json``: fetches, searches, denied fetches and the
    distinct pages fetched."""
    fetches = [i for i in items if i.get("tool") == FETCH]
    return {
        "fetches": len(fetches),
        "searches": sum(i.get("tool") == SEARCH for i in items),
        "denied": sum(i.get("status") == "denied" for i in items),
        "pages": len({i.get("url") for i in fetches if i.get("status") == "ok"}),
    }


def record(run_root: Path, agent: str, items: list[dict[str, Any]]) -> None:
    """Append a session's lookups to ``research/sources.jsonl`` of the run."""
    for item in items:
        append_jsonl(run_root / SOURCES_FILE, {"agent": agent, **item})


def _cited(run_root: Path) -> dict[str, str]:
    """The text of the files where agents cite sources (relative path -> text)."""
    texts: dict[str, str] = {}
    patterns = (
        "targets/*/research.md",
        "targets/*/plan.md",
        "targets/*/NOTES.md",
        "targets/*/workers/*/NOTES.md",
        "transforms/NOTES.md",
        "plan.json",
    )
    for pattern in patterns:
        for path in sorted(run_root.glob(pattern)):
            try:
                texts[path.relative_to(run_root).as_posix()] = path.read_text(errors="replace")
            except OSError:
                continue
    return texts


def report_lines(run_root: Path) -> list[str]:
    """``## Sources used`` of report.md: the pages agents fetched, by whom and where they
    are cited; searches and refused fetches in a line each ([] without lookups)."""
    items = read_jsonl(run_root / SOURCES_FILE)
    if not items:
        return []
    fetched: dict[str, dict[str, Any]] = {}
    for i in items:
        if i.get("tool") == FETCH and i.get("status") == "ok" and i.get("url"):
            entry = fetched.setdefault(str(i["url"]).split("#")[0], {"agents": [], "n": 0})
            entry["n"] += 1
            if i.get("agent") not in entry["agents"]:
                entry["agents"].append(i.get("agent"))
    searches = [i for i in items if i.get("tool") == SEARCH]
    denied = [i for i in items if i.get("status") == "denied"]
    sessions = len({i.get("agent") for i in items})
    lines = [
        "",
        "## Sources used",
        "",
        f"* {len(items)} lookups in {sessions} agent sessions: {len(fetched)} pages fetched, "
        f"{len(searches)} searches, {len(denied)} fetches refused (outside the allowed "
        f"domains); every lookup is in `{SOURCES_FILE.as_posix()}`",
    ]
    if fetched:
        texts = _cited(run_root)
        lines += ["", "| source | fetched by | cited in |", "|---|---|---|"]
        for url, entry in list(fetched.items())[:REPORT_ROWS]:
            cited = [name for name, text in texts.items() if url in text]
            agents = ", ".join(f"`{a}`" for a in entry["agents"][:4])
            more = f" (+{len(entry['agents']) - 4})" if len(entry["agents"]) > 4 else ""
            lines.append(
                f"| <{url}> | {agents}{more} | "
                + (", ".join(f"`{c}`" for c in cited[:4]) or "—")
                + " |"
            )
        if len(fetched) > REPORT_ROWS:
            lines.append(f"\n… and {len(fetched) - REPORT_ROWS} more pages")
    if searches:
        queries = "; ".join(f"“{str(i.get('query'))[:80]}”" for i in searches[:8])
        lines += ["", f"Searches: {queries}" + (" …" if len(searches) > 8 else "")]
    if denied:
        urls = ", ".join(f"<{str(i.get('url'))[:100]}>" for i in denied[:5])
        lines += ["", f"Refused fetches: {urls}"]
    return [*lines, ""]
