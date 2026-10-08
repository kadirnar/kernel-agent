"""The blackboard of a run: ``board.jsonl`` (docs/MULTIAGENT.md §3.4, issue #187).

Sessions that run beside each other (``improve --agents N``, ``coordinator.py``) and after
each other share **conclusions** here, never progress: why a kept result wins, a dead end and
the error that proves it, a fact about the GPU or a compiler, a module a session is about to
change, a question for another arm (KernelArc's wins and traps, KernelAgent's avoid / try
patterns: docs/MULTIAGENT-LITERATURE.md R4). The board is advisory: nothing reads it for a
score, and the ledger stays the truth.

* **Store**: ``<run>/board.jsonl``, append-only, one entry per line, numbered ``id`` 1, 2, ...
  in file order. Only the coordinator process writes it (its sessions' MCP tools and its own
  posts), under an in-process lock and an ``flock`` of the file (:meth:`Board.post`).
* **Writes**: agents post with the ``post_note`` tool (:meth:`Board.note`): the kinds of
  :data:`AGENT_KINDS`, at most :data:`POSTS_PER_SESSION` per session, the text cut to
  :data:`TEXT_CHARS`, a text already on the board (same kind and target) refused, a
  ``winner`` note only for a kept result. kernel-agent posts as :data:`COORDINATOR`: a
  ``winner`` on every new best (``agent/tools.record_candidate`` / ``record_e2e_result``:
  :func:`kernel_winner`, :func:`e2e_winner`), each ``integration`` and ``round``
  (``improve.py``), and the ``claim`` / ``release`` of the modules of every concurrent
  session (``coordinator.py``; an agent's own ``claim`` joins its session's claims).
* **Reads**, all bounded and filtered by the reader's subscription (:func:`relevant`: its
  role, arm, module class and precision; what is addressed to it): a slice digest's
  ``## Board`` (:meth:`Board.section`, which sets the session's cursor), the first lines of
  what is new since the cursor on every evaluation result (:meth:`Board.piggyback`), the
  full text with ``read_board`` (:meth:`Board.read`).
* A research session's evidence has its target's board history (:func:`history_lines`), the
  librarian's prompt the run's insights and traps (:func:`lessons_lines`), and the report
  what the board held and how many posted winners another session built on
  (:func:`report_lines`).

``improve --board auto`` (default) keeps a board when ``--agents`` is above 1 (``--agents 1``
stays the loop as before); ``on`` / ``off`` (:func:`enabled`).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent import ledger
from kernel_agent.workspace import RunDir, read_json

FILE = "board.jsonl"
COORDINATOR = "coordinator"  # the author of kernel-agent's own posts
INSIGHT, TRAP, WINNER, CLAIM, QUESTION = "insight", "trap", "winner", "claim", "question"
RELEASE, INTEGRATION, ROUND = "release", "integration", "round"
#: What an agent may post (``post_note``)
AGENT_KINDS = (INSIGHT, TRAP, WINNER, CLAIM, QUESTION)
KINDS = (*AGENT_KINDS, RELEASE, INTEGRATION, ROUND)
NOTES = (INSIGHT, TRAP, QUESTION)  # conclusions that are not about one kept result
#: The roles whose sessions have the board's tools, its prompt section and its digest section
ROLES = ("kernel", "systems", "native", "research")
TOOLS = ("post_note", "read_board")
SYSTEMS, NATIVE = "systems", "native"  # the arms that are not kernel targets (scheduler.py)

TEXT_CHARS = 1500  # of a note
REFS, REF_CHARS = 8, 200  # refs of a note, characters of one
POSTS_PER_SESSION = 6  # an agent session's notes
MAX_ENTRIES = 5000  # beyond this no agent note is taken (kernel-agent's own posts go on)
LINE_CHARS = 160  # an entry's first line, as a piggyback shows it
PIGGYBACK_ITEMS = 5  # first lines on one evaluation result
DIGEST_ITEMS = 10  # entries in a digest's ``## Board`` (+ those addressed to the session)
ADDRESSED_ITEMS = 5
DIGEST_CHARS = 400  # of an entry's text in a digest or a research session's evidence
READ_ITEMS, READ_MAX, READ_CHARS = 20, 50, 12000  # read_board: default, most, characters
HISTORY_ITEMS = 30  # a research session's board history
LESSONS_CHARS = 4000  # the board's insights and traps in the librarian's prompt
_ADDRESS = re.compile(r"^(arm|role|session):[A-Za-z0-9_.#:-]{1,80}$")
_EXP = re.compile(r"^exp:(\d+)$")


def enabled(mode: str, agents: int) -> bool:
    """Whether ``improve`` keeps a board: ``--board on``, or ``auto`` with ``--agents`` > 1."""
    return mode == "on" or (mode == "auto" and agents > 1)


def _role(name: str) -> str:
    """The role of a session name or label (``kernel-attn-w2#3`` → ``kernel``)."""
    return name.split("#", 1)[0].split("-", 1)[0].lower()


# ------------------------------------------------------------------ readers


@dataclass(frozen=True)
class Reader:
    """Who reads the board: a session (``label``: its notes and results are not news to it),
    its ``role`` and ``arm`` (a kernel target, ``systems`` or ``native``; a research
    session's target), the module class and reduced precision of its target."""

    label: str
    role: str
    arm: str | None = None
    module_class: str | None = None
    precision: str | None = None

    @classmethod
    def of(cls, run: RunDir, label: str, role: str, arm: str | None = None) -> Reader:
        module_class = precision = None
        if arm and arm not in (SYSTEMS, NATIVE):
            spec = read_json(run.target(arm) / "spec.json", {}) or {}
            module_class = spec.get("module_class") or None
            precision = spec.get("precision") or None
        return cls(label, role, arm, module_class, precision)

    @classmethod
    def of_session(cls, run: RunDir, bound: Any) -> Reader:
        """The reader of an agent session from its tools' binding (``SessionBinding``)."""
        role = str(bound.role or _role(bound.label))
        arm = role if role in (SYSTEMS, NATIVE) else bound.target_id
        return cls.of(run, str(bound.label), role, arm)

    def addresses(self) -> set[str]:
        """The ``to`` values that address it."""
        out = {f"role:{self.role}", f"session:{self.label}"}
        return out | ({f"arm:{self.arm}"} if self.arm else set())


def relevant(entry: dict[str, Any], reader: Reader) -> bool:
    """Whether ``entry`` is for ``reader`` (its subscription, docs/MULTIAGENT.md §3.4).

    Never its own note or its own result, never a ``release`` (it only closes a claim:
    :func:`open_claims`); an addressed entry only for its addressee; an integration or a new
    round for everyone. Else what is about its arm or its module class, and by role: kernel
    and research sessions the notes that hold for every target (no target) or for their
    precision; systems every ``winner`` and ``claim`` and the general notes; native every
    ``winner`` and every note."""
    tags = entry.get("tags") or {}
    kind = entry.get("kind")
    if kind == RELEASE or (
        reader.label and reader.label in (entry.get("author"), tags.get("session"))
    ):
        return False
    if to := entry.get("to"):
        return to in reader.addresses()
    if kind in (INTEGRATION, ROUND):
        return True
    targets = tags.get("targets") or []
    cls = tags.get("module_class")
    if (reader.arm and reader.arm in targets) or (
        reader.module_class and cls == reader.module_class
    ):
        return True
    general = not targets and not cls
    if reader.role == NATIVE:
        return kind in (WINNER, *NOTES)
    if reader.role == SYSTEMS:
        return kind in (WINNER, CLAIM) or (general and kind in NOTES)
    if reader.role in ("kernel", "research"):
        same = bool(reader.precision) and tags.get("precision") == reader.precision
        return kind in NOTES and (general or same)
    return kind == WINNER


# ------------------------------------------------------------------ rendering


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def head(entry: dict[str, Any]) -> str:
    """``#12 winner [attn] (coordinator)``: id, kind, targets, addressee and author."""
    targets = (entry.get("tags") or {}).get("targets") or []
    out = f"#{entry.get('id')} {entry.get('kind')}"
    out += f" [{', '.join(map(str, targets))}]" if targets else ""
    out += f" → {entry['to']}" if entry.get("to") else ""
    return f"{out} ({entry.get('author')})"


def line(entry: dict[str, Any], chars: int = LINE_CHARS) -> str:
    """An entry as its head and the first line of its text (a piggyback item)."""
    first = str(entry.get("text") or "").strip().split("\n", 1)[0]
    return _cut(f"{head(entry)}: {_flat(first)}", chars)


def brief(entry: dict[str, Any], chars: int = DIGEST_CHARS) -> str:
    """An entry as a digest bullet: its head, its text (cut; kernel-agent's own posts their
    first line, which holds their facts) and its refs."""
    text = str(entry.get("text") or "")
    if entry.get("author") == COORDINATOR:
        text = text.strip().split("\n", 1)[0]
    refs = entry.get("refs") or []
    tail = f" (refs: {', '.join(f'`{r}`' for r in refs)})" if refs else ""
    return f"* {head(entry)}: {_cut(_flat(text), chars)}{tail}"


def open_claims(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``entries`` without the claims a later ``release`` closed and without the releases:
    a claim of a session (kernel-agent's names it, an agent's is its author's) closes with
    that session's release (the islands of a target run at once, ``workers.py``), one of no
    session with the next release of its arm."""
    by_arm: dict[str, int] = {}  # arm -> its last release
    by_session: dict[str, int] = {}  # session -> its last release
    for e in entries:
        if e.get("kind") == RELEASE:
            tags = e.get("tags") or {}
            by_arm[str(tags.get("arm"))] = int(e["id"])
            if tags.get("session"):
                by_session[str(tags["session"])] = int(e["id"])

    def closed(e: dict[str, Any]) -> bool:
        tags = e.get("tags") or {}
        author = e.get("author")
        session = tags.get("session") or (author if author != COORDINATOR else None)
        if session:
            return by_session.get(str(session), 0) > int(e["id"])
        return by_arm.get(str(tags.get("arm")), 0) > int(e["id"])

    return [
        e
        for e in entries
        if e.get("kind") != RELEASE and not (e.get("kind") == CLAIM and closed(e))
    ]


def compact(entry: dict[str, Any]) -> dict[str, Any]:
    """An entry as ``read_board`` returns it (its full text)."""
    tags = entry.get("tags") or {}
    out: dict[str, Any] = {k: entry[k] for k in ("id", "time", "author", "kind") if k in entry}
    for key in ("targets", "module_class", "precision", "arm"):
        if tags.get(key):
            out[key] = tags[key]
    for key in ("to", "reply_to"):
        if entry.get(key):
            out[key] = entry[key]
    out["text"] = entry.get("text", "")
    if entry.get("refs"):
        out["refs"] = entry["refs"]
    return out


def _key(kind: str, tags: dict[str, Any], to: str | None, text: str) -> str:
    """What two entries share when they say the same (dedup): kind, target, addressee and
    the text up to case and whitespace."""
    about = ",".join(sorted(map(str, tags.get("targets") or []))) or str(tags.get("module_class"))
    blob = "\0".join((kind, about, to or "", _flat(text).lower()))
    return hashlib.sha1(blob.encode()).hexdigest()


# ------------------------------------------------------------------ the board


class Refused(ValueError):
    """A note the board does not take (the message says why)."""


class Board:
    """``board.jsonl`` of one run in the coordinator process: its entries (in memory, synced
    with the file), the sessions' cursors (the last entry each was shown) and posts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._entries: list[dict[str, Any]] = []
        self._offset = 0  # bytes of the file read so far
        self._keys: dict[str, int] = {}  # dedup key -> entry id
        self._posts: dict[str, int] = {}  # agent author -> its notes
        self.cursors: dict[str, int] = {}  # session label -> the last entry id it was shown
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.sync()

    # -------------------------------------------------------- store

    @property
    def last_id(self) -> int:
        with self._lock:
            return int(self._entries[-1]["id"]) if self._entries else 0

    def sync(self) -> None:
        """Read the lines appended to the file since the last read (another process's too).
        A last line without its newline is left for later; a line that is not an entry
        (a crashed writer's) is skipped."""
        with self._lock:
            if not self.path.exists():
                return
            with self.path.open("rb") as fh:
                fh.seek(self._offset)
                data = fh.read()
            end = data.rfind(b"\n") + 1
            for raw in data[:end].splitlines():
                with contextlib.suppress(ValueError):
                    entry = json.loads(raw)
                    if isinstance(entry, dict) and isinstance(entry.get("id"), int):
                        self._index(entry)
            self._offset += end

    def _index(self, entry: dict[str, Any]) -> None:
        self._entries.append(entry)
        tags = entry.get("tags") or {}
        self._keys.setdefault(
            _key(str(entry.get("kind")), tags, entry.get("to"), str(entry.get("text"))),
            int(entry["id"]),
        )
        if (author := str(entry.get("author"))) != COORDINATOR:
            self._posts[author] = self._posts.get(author, 0) + 1

    def entries(self, since: int = 0) -> list[dict[str, Any]]:
        """The entries after id ``since``, oldest first."""
        with self._lock:
            self.sync()
            return [e for e in self._entries if int(e["id"]) > since]

    def post(
        self,
        author: str,
        kind: str,
        text: str,
        *,
        tags: dict[str, Any] | None = None,
        to: str | None = None,
        refs: Iterable[str] = (),
        reply_to: int | None = None,
    ) -> dict[str, Any]:
        """Append an entry (as given: :meth:`note` checks an agent's) and return it; an entry
        that says what one on the board says already is not appended again (that one is
        returned). The id is taken and the line written under the file's ``flock``, so
        writers in other threads and processes never share an id or tear a line."""
        return self._append(author, kind, text, tags, to, list(refs), reply_to)[0]

    def _append(
        self,
        author: str,
        kind: str,
        text: str,
        tags: dict[str, Any] | None,
        to: str | None,
        refs: list[str],
        reply_to: int | None,
    ) -> tuple[dict[str, Any], bool]:
        """:meth:`post`; and whether the entry is new (False: a duplicate's)."""
        tags = {k: v for k, v in (tags or {}).items() if v not in (None, "", [])}
        key = _key(kind, tags, to, text)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+b") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    self.sync()  # what other writers appended: the next id and the dedup keys
                    if (same := self._keys.get(key)) is not None:
                        return next(e for e in self._entries if e["id"] == same), False
                    if size := fh.seek(0, os.SEEK_END):
                        fh.seek(size - 1)
                        if fh.read(1) != b"\n":
                            fh.write(b"\n")  # a torn last line stays a line of its own
                    ts = ledger.clock()
                    entry: dict[str, Any] = {
                        "id": self.last_id + 1,
                        "ts": round(ts, 3),
                        "time": ledger.stamp(ts),
                        "author": author,
                        "kind": kind,
                        **({"to": to} if to else {}),
                        **({"tags": tags} if tags else {}),
                        "text": text,
                        **({"refs": list(refs)} if refs else {}),
                        **({"reply_to": reply_to} if reply_to else {}),
                    }
                    fh.write((json.dumps(entry, default=str) + "\n").encode())
                    fh.flush()
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            self.sync()
            listeners = list(self._listeners)
        for listener in listeners:  # the coordinator (an agent's claim); never fails a post
            with contextlib.suppress(Exception):
                listener(entry)
        return entry, True

    def listen(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Call ``listener`` with every new entry (from the thread that posted it)."""
        with self._lock:
            self._listeners.append(listener)

    def unlisten(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self._lock, contextlib.suppress(ValueError):
            self._listeners.remove(listener)

    # -------------------------------------------------------- reads

    def join(self, label: str) -> None:
        """A session starts (its tools are built): what is on the board now is not news to it
        (a cursor its digest set stays)."""
        with self._lock:
            self.sync()
            self.cursors.setdefault(label, self.last_id)

    def section(self, reader: Reader) -> list[str]:
        """``## Board`` of a session's digest ([] when nothing is for it): the entries
        addressed to it (the newest :data:`ADDRESSED_ITEMS`) and the newest
        :data:`DIGEST_ITEMS` others for it (claims only while open), oldest first. Sets its
        cursor: its evaluation results carry what is posted after."""
        with self._lock:
            self.sync()
            mine = [e for e in open_claims(self._entries) if relevant(e, reader)]
            self.cursors[reader.label] = self.last_id
        addresses = reader.addresses()
        addressed = [e for e in mine if e.get("to") in addresses][-ADDRESSED_ITEMS:]
        rest = [e for e in mine if e.get("to") not in addresses][-DIGEST_ITEMS:]
        shown = sorted([*addressed, *rest], key=lambda e: int(e["id"]))
        if not shown:
            return []
        lines = [
            "",
            f"## Board (`{FILE}`: what the other sessions concluded; advice, never scored)",
            f"The newest of {len(mine)} entries for you, oldest first; `read_board` has their "
            "full text, and the new ones ride on your evaluation results.",
            "",
        ]
        return lines + [brief(e) for e in shown]

    def news(self, reader: Reader) -> list[dict[str, Any]]:
        """The entries for ``reader`` posted since its cursor (which moves to the newest)."""
        with self._lock:
            self.sync()
            since = self.cursors.setdefault(reader.label, self.last_id)
            self.cursors[reader.label] = self.last_id
            return [e for e in self._entries if int(e["id"]) > since and relevant(e, reader)]

    def piggyback(self, reader: Reader) -> dict[str, Any]:
        """``{"board": {...}}`` for an evaluation result: how many entries are new for the
        session and the first lines of the newest :data:`PIGGYBACK_ITEMS` ({} when none)."""
        new = self.news(reader)
        if not new:
            return {}
        out: dict[str, Any] = {"new": len(new), "items": [line(e) for e in new[-PIGGYBACK_ITEMS:]]}
        out["read"] = f"read_board(since={int(new[0]['id']) - 1}) for the full text"
        return {"board": out}

    def query(
        self,
        reader: Reader | None = None,
        *,
        since: int = 0,
        after: float | None = None,
        kinds: Iterable[str] = (),
        target: str = "",
    ) -> list[dict[str, Any]]:
        """The entries after id ``since`` and posted at ``after`` (a time) or later, of
        ``kinds``, about ``target`` (a target id, module class or arm), for ``reader`` (its
        subscription, :func:`relevant`; None: every entry), oldest first."""
        kinds = set(kinds)
        return [
            e
            for e in self.entries(since)
            if (after is None or float(e.get("ts") or 0.0) >= after)
            and (not kinds or e.get("kind") in kinds)
            and (not target or _about(e, target))
            and (reader is None or relevant(e, reader))
        ]

    def read(self, reader: Reader, args: dict[str, Any]) -> dict[str, Any]:
        """``read_board``: the entries for the session (or by ``kinds`` / ``target``, or
        ``all``) after the id ``since`` and of the last ``minutes``, the newest ``limit`` (at
        most :data:`READ_MAX` and :data:`READ_CHARS`) with their full text. The session's own
        view (no filter) moves its cursor."""
        kinds = _kinds(args.get("kinds"))
        if bad := [k for k in kinds if k not in KINDS]:
            raise Refused(f"unknown kinds {bad}; kinds: {', '.join(KINDS)}")
        target = str(args.get("target") or "").strip()
        since = _int(args.get("since"), 0)
        minutes = _int(args.get("minutes"), 0)
        after = ledger.clock() - 60.0 * minutes if minutes > 0 else None
        limit = max(1, min(_int(args.get("limit"), READ_ITEMS), READ_MAX))
        mine = not (kinds or target or args.get("all"))
        with self._lock:
            pool = self.query(
                reader if mine else None, since=since, after=after, kinds=kinds, target=target
            )
            last = self.last_id
            if mine and after is None and since <= self.cursors.get(reader.label, 0):
                self.cursors[reader.label] = last  # it read everything new for it
        shown = [compact(e) for e in pool[-limit:]]
        while len(shown) > 1 and len(json.dumps(shown, default=str)) > READ_CHARS:
            shown.pop(0)  # the oldest first
        return {
            "entries": shown,
            "matching": len(pool),
            "shown": len(shown),
            "last_id": last,
            "note": "advice from other sessions, never scored: check a note before you build "
            "on it; the ledger is the truth",
        }

    # -------------------------------------------------------- an agent's note

    def note(self, run: RunDir, reader: Reader, args: dict[str, Any]) -> dict[str, Any]:
        """``post_note``: check an agent's note and post it (:class:`Refused` says why not):
        its kind, its text (cut to :data:`TEXT_CHARS`), :data:`POSTS_PER_SESSION` notes per
        session, a known target, addressee and reply, a ``winner`` only for a kept result
        (its ``exp:N`` or snapshot in ``refs``), a ``claim`` only of a target or module
        class, nothing already on the board."""
        if reader.role not in ROLES:
            raise Refused(f"the board is for the {', '.join(ROLES)} sessions")
        kind = str(args.get("kind") or "").strip().lower()
        if kind not in AGENT_KINDS:
            raise Refused(f"kind is one of {', '.join(AGENT_KINDS)}, not {kind!r}")
        text = str(args.get("text") or "").strip()
        if not text:
            raise Refused("text is required: the conclusion, with its numbers")
        cut = len(text) > TEXT_CHARS
        text = _cut(text, TEXT_CHARS)
        refs = [_cut(str(r).strip(), REF_CHARS) for r in _list(args.get("refs")) if str(r).strip()]
        refs = refs[:REFS]
        with self._lock:
            self.sync()
            posts = self._posts.get(reader.label, 0)
            if posts >= POSTS_PER_SESSION:
                raise Refused(
                    f"this session posted {posts} notes, the most a session may: post only "
                    "conclusions another session can act on"
                )
            if len(self._entries) >= MAX_ENTRIES:
                raise Refused(f"the board is full ({MAX_ENTRIES} entries)")
            reply_to = _int(args.get("reply_to"), 0) or None
            if reply_to is not None and not any(e["id"] == reply_to for e in self._entries):
                raise Refused(f"reply_to: no entry #{reply_to}")
        tags = _target_tags(run, str(args.get("target") or "").strip())
        to = _address(run, str(args.get("to") or "").strip())
        if kind == WINNER:
            row = _kept(run, refs)
            if row is None:
                raise Refused(
                    "a winner note says why a kept result wins: give its `exp:N` or its "
                    "snapshot (`history/...`) in refs; post anything else as an insight"
                )
            if not tags:  # about the result's target (or the end-to-end arm's)
                about = row["target"] if row["target"] != ledger.E2E else reader.arm
                tags = _target_tags(run, str(about or ""))
            tags.update(exp=row["exp"], snapshot=row["snapshot"])
            if f"exp:{row['exp']}" not in refs:
                refs = [*refs, f"exp:{row['exp']}"][:REFS]
        if kind == CLAIM and not (tags.get("targets") or tags.get("module_class")):
            raise Refused("a claim names what you will change: its target id or module class")
        tags |= {"arm": reader.arm} if reader.arm else {}
        entry, new = self._append(reader.label, kind, text, tags, to, refs, reply_to)
        if not new:
            raise Refused(f"already on the board as #{entry['id']} ({entry.get('author')})")
        with self._lock:
            left = POSTS_PER_SESSION - self._posts.get(reader.label, 0)
        out: dict[str, Any] = {"posted": entry["id"], "kind": kind, "notes_left": left}
        if cut:
            out["cut"] = f"the text was cut to {TEXT_CHARS} characters"
        return out


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


def _kinds(value: Any) -> list[str]:
    items = _list(value)
    if len(items) == 1 and isinstance(items[0], str):
        items = items[0].split(",")
    return [str(k).strip().lower() for k in items if str(k).strip()]


def _about(entry: dict[str, Any], target: str) -> bool:
    tags = entry.get("tags") or {}
    return target in (tags.get("targets") or []) or target in (
        tags.get("module_class"),
        tags.get("arm"),
    )


def _target_tags(run: RunDir, target: str) -> dict[str, Any]:
    """The tags of a note about ``target``: a target id (its module class and reduced
    precision), ``systems`` / ``native``, a module class of the run (its targets), or
    nothing (a note for every target)."""
    if not target:
        return {}
    ids = run.target_ids()
    if target in ids:
        spec = read_json(run.target(target) / "spec.json", {}) or {}
        return {
            "targets": [target],
            "module_class": spec.get("module_class"),
            "precision": spec.get("precision"),
        }
    if target in (SYSTEMS, NATIVE):
        return {"targets": [target]}
    classes = {
        t: (read_json(run.target(t) / "spec.json", {}) or {}).get("module_class") for t in ids
    }
    if target in classes.values():
        return {"module_class": target, "targets": [t for t, c in classes.items() if c == target]}
    known = ", ".join([*ids, SYSTEMS, NATIVE])
    raise Refused(f"unknown target {target!r}: a target id ({known}) or a module class")


def _address(run: RunDir, to: str) -> str | None:
    """``to`` as an address (``arm:<id>``, ``role:<role>``, ``session:<label>``; a bare arm
    id is ``arm:<id>``), or None."""
    if not to:
        return None
    if to in (*run.target_ids(), SYSTEMS, NATIVE):
        return f"arm:{to}"
    if not _ADDRESS.match(to):
        raise Refused(
            f'to is "arm:<target or systems or native>", "role:<role>" or '
            f'"session:<label>", not {to!r}'
        )
    return to


def _kept(run: RunDir, refs: list[str]) -> dict[str, Any] | None:
    """The kept ledger row a winner note's ``refs`` name (``exp:N`` or its snapshot)."""
    exps = {int(m[1]) for r in refs if (m := _EXP.match(r))}
    names = {Path(r).name for r in refs if not _EXP.match(r)}
    for row in reversed(ledger.rows(run)):
        if row["status"] == ledger.KEEP and (row["exp"] in exps or row["snapshot"] in names):
            return row
    return None


# ------------------------------------------------------------------ the run's board

_boards: dict[Path, Board] = {}
_boards_lock = threading.Lock()


def open_board(run: RunDir) -> Board:
    """The board of ``run`` in this process, from now on :func:`active` (until :func:`close`)."""
    key = run.root.resolve()
    with _boards_lock:
        found = _boards.get(key)
        if found is None:
            found = _boards[key] = Board(run.root / FILE)
        return found


def active(run: RunDir) -> Board | None:
    """The board of ``run`` while ``improve`` keeps one (None: no board)."""
    with _boards_lock:
        return _boards.get(run.root.resolve())


def close(run: RunDir) -> None:
    with _boards_lock:
        _boards.pop(run.root.resolve(), None)


@contextlib.contextmanager
def opened(run: RunDir) -> Iterator[Board]:
    """:func:`open_board` for the body, then :func:`close`."""
    try:
        yield open_board(run)
    finally:
        close(run)


def load(run: RunDir) -> list[dict[str, Any]]:
    """Every entry of ``run``'s board (its file when no board is active; [] without one)."""
    found = active(run)
    if found is not None:
        return found.entries()
    path = run.root / FILE
    return Board(path).entries() if path.exists() else []


# ------------------------------------------------------------------ kernel-agent's posts


def kernel_winner(
    run: RunDir, target_id: str, row: dict[str, Any], *, hypothesis: str, session: str | None
) -> None:
    """``winner``: a new best of a kernel target (its ledger ``row``), with the snapshot to
    build on and to run end to end (no-op without a board)."""
    found = active(run)
    if found is None:
        return
    spec = read_json(run.target(target_id) / "spec.json", {}) or {}
    snap = str(row["snapshot"])
    what = [str(row.get("backend") or "")]
    if spec.get("precision"):
        what.append(str(spec["precision"]))
    if row.get("pct_of_sol") is not None:
        what.append(f"{float(row['pct_of_sol']):.0f} % of SOL")
    if row.get("idea"):
        what.append(f"idea `{row['idea']}`")
    by = f", by `{session}`" if session else ""
    text = (
        f"`{target_id}`: new best {float(row['speedup'] or 0.0):.3f}x module speedup "
        f"({', '.join(w for w in what if w)}): `history/{snap}`{by}.\n"
        f"Hypothesis: {_flat(hypothesis)}\n"
        f'Build on it: `parent="history/{snap}"` in a session of `{target_id}`; '
        f'`"{target_id}=history/{snap}"` in evaluate_e2e kernels.'
    )
    found.post(
        COORDINATOR,
        WINNER,
        text,
        tags={
            "targets": [target_id],
            "module_class": spec.get("module_class"),
            "precision": spec.get("precision"),
            "backend": row.get("backend"),
            "arm": target_id,
            "session": session,
            "exp": row["exp"],
            "snapshot": snap,
        },
        refs=[f"targets/{target_id}/history/{snap}", f"exp:{row['exp']}"],
    )


def e2e_winner(
    run: RunDir,
    row: dict[str, Any],
    snaps: list[Path],
    kernels: list[str],
    *,
    hypothesis: str,
    session: str | None,
) -> None:
    """``winner``: a new end-to-end best of the systems or native agent (no-op without a
    board)."""
    found = active(run)
    if found is None:
        return
    arm = NATIVE if str(row.get("backend") or "").startswith(NATIVE) else SYSTEMS
    ms = "" if row.get("new_ms") is None else f", {float(row['new_ms']):.1f} ms"
    by = f", by `{session}`" if session else ""
    files = [f"transforms/history/{s.name}" for s in snaps]
    text = (
        f"end to end: new best {float(row['speedup'] or 0.0):.3f}x{ms} ({arm}: "
        f"`{row['snapshot']}`){by}.\nHypothesis: {_flat(hypothesis)}"
        + (f"\nKernels in it: {', '.join(f'`{k}`' for k in kernels)}" if kernels else "")
    )
    found.post(
        COORDINATOR,
        WINNER,
        text,
        tags={
            "targets": [arm, *(k.partition("=")[0] for k in kernels)],
            "backend": row.get("backend"),
            "arm": arm,
            "session": session,
            "exp": row["exp"],
            "snapshot": row["snapshot"],
        },
        refs=[*files, f"exp:{row['exp']}"][:REFS],
    )


def integration(run: RunDir, rec: dict[str, Any]) -> None:
    """``integration``: a measured re-integration's result (``improve.json`` record ``rec``;
    no-op without a board): the end-to-end speedup and what it accepted."""
    found = active(run)
    if found is None:
        return
    accepted = list(rec.get("accepted") or [])
    ms = f" ({float(rec['median_ms']):.1f} ms)" if rec.get("median_ms") else ""
    gain = "a real gain" if rec.get("gain") else "no real gain over the last one"
    text = (
        f"integration {rec.get('n')}: {float(rec.get('speedup') or 1.0):.3f}x end to end{ms}, "
        f"{gain}; accepted: {', '.join(f'`{a}`' for a in accepted) or 'nothing'}.\n"
        "The baseline your results integrate against moved: build on the accepted set."
    )
    ids = set(run.target_ids())
    found.post(
        COORDINATOR,
        INTEGRATION,
        text,
        tags={"targets": [a for a in accepted if a in ids], "n": rec.get("n")},
        refs=["integration.json"],
    )


def round_started(
    run: RunDir, n: int, median_ms: float, targets: list[str], live: list[str]
) -> None:
    """``round``: round ``n`` started from a re-profile of the optimised model (no-op without
    a board)."""
    found = active(run)
    if found is None:
        return
    text = (
        f"round {n}: the optimised model re-profiled at {median_ms:.1f} ms; new targets: "
        f"{', '.join(f'`{t}`' for t in targets) or 'none'}; live arms: "
        f"{', '.join(f'`{a}`' for a in live) or 'none'}."
    )
    found.post(COORDINATOR, ROUND, text, tags={"n": n}, refs=[f"rounds/{n}/profile/summary.md"])


def _modules(claims: list[tuple[str, str | None]]) -> str:
    return ", ".join(
        "the whole model" if cls == "*" else f"`{cls}`" + (f" (`{qual}`)" if qual else "")
        for cls, qual in claims
    )


def claim(
    run: RunDir, label: str, role: str, arm: str, claims: list[tuple[str, str | None]]
) -> bool:
    """``claim``: the session ``label`` (of ``role`` on ``arm``) starts on the modules of
    ``claims`` (``coordinator.Claim``); whether it was posted (a board and claims)."""
    found = active(run)
    if found is None or not claims:
        return False
    classes = [cls for cls, _ in claims if cls != "*"]
    found.post(
        COORDINATOR,
        CLAIM,
        f"`{label}` ({role} on `{arm}`) works on {_modules(claims)}.",
        tags={
            "targets": [arm] if arm not in (SYSTEMS, NATIVE) else [],
            "module_class": classes[0] if len(classes) == 1 else None,
            "arm": arm,
            "session": label,
            "modules": [list(c) for c in claims],
        },
    )
    return True


def release(run: RunDir, label: str, arm: str, claims: list[tuple[str, str | None]]) -> None:
    """``release``: the session ``label`` ended; its claimed modules are free (no-op without
    a board)."""
    found = active(run)
    if found is None or not claims:
        return
    classes = [cls for cls, _ in claims if cls != "*"]
    found.post(
        COORDINATOR,
        RELEASE,
        f"`{label}` ended: {_modules(claims)} released.",
        tags={
            "targets": [arm] if arm not in (SYSTEMS, NATIVE) else [],
            "module_class": classes[0] if len(classes) == 1 else None,
            "arm": arm,
            "session": label,
        },
    )


# ------------------------------------------------------------------ what others read of it


def history_lines(run: RunDir, target_id: str, items: int = HISTORY_ITEMS) -> list[str]:
    """``## Board`` of a research session's evidence: the conclusions about its target (and
    the general ones), newest :data:`HISTORY_ITEMS` ([] without any)."""
    reader = Reader.of(run, "", "research", target_id)
    found = [
        e for e in load(run) if e.get("kind") not in (CLAIM, RELEASE, ROUND) and relevant(e, reader)
    ]
    if not found:
        return []
    return [
        "",
        f"## Board: what the sessions concluded about this target (`{FILE}`)",
        f"The newest {min(items, len(found))} of {len(found)} entries, oldest first. They are "
        "the sessions' advice, not evidence: the ledger above is the truth. A trap the "
        "ledger confirms belongs under *Do not try*.",
        "",
        *(brief(e) for e in found[-items:]),
    ]


def lessons_lines(run: RunDir, chars: int = LESSONS_CHARS) -> list[str]:
    """The board's insights, traps and agents' winner notes for the librarian's prompt ([]
    without any): the newest that fit in ``chars``."""
    notes = [
        e
        for e in load(run)
        if e.get("kind") in (INSIGHT, TRAP)
        or (e.get("kind") == WINNER and e.get("author") != COORDINATOR)
    ]
    shown: list[str] = []
    for entry in reversed(notes):
        item = brief(entry)
        if sum(len(s) + 1 for s in shown) + len(item) > chars:
            break
        shown.insert(0, item)
    if not shown:
        return []
    return [
        "",
        f"# Board: the sessions' own conclusions (`{FILE}`)",
        "Insights, traps and why winners won, as the agents posted them during the run. Turn "
        "one into a rule only where the ledger and the notes above confirm it.",
        "",
        *shown,
    ]


def _built_on(winner: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    """Whether another session's evaluation after ``winner`` built on it: a kernel candidate
    with its snapshot as parent, an end-to-end run with its target's kernel or its files."""
    tags = winner.get("tags") or {}
    exp, who = int(tags.get("exp") or 0), tags.get("session")
    snap = str(tags.get("snapshot") or "")
    arm = tags.get("arm")
    for row in rows:
        if (row["exp"] or 0) <= exp or not row.get("session") or row.get("session") == who:
            continue
        if row["target"] == ledger.E2E:
            parts = set(str(row["snapshot"]).split("+"))
            if (arm not in (SYSTEMS, NATIVE) and arm in parts) or parts & set(snap.split("+")):
                return True
        elif row["target"] == arm and Path(str(row.get("parent") or "")).name == snap:
            return True
    return False


def report_lines(run: RunDir) -> list[str]:
    """The report's line on the board ([] without one): its entries by author and kind, and
    how many posted winners another session built on afterwards (:func:`_built_on`)."""
    entries = load(run)
    if not entries:
        return []

    def counts(items: list[dict[str, Any]]) -> str:
        by: dict[str, int] = {}
        for e in items:
            by[str(e.get("kind"))] = by.get(str(e.get("kind")), 0) + 1
        return ", ".join(f"{k} {v}" for k, v in by.items()) or "none"

    agents = [e for e in entries if e.get("author") != COORDINATOR]
    own = [e for e in entries if e.get("author") == COORDINATOR]
    winners = [e for e in own if e.get("kind") == WINNER]
    rows = ledger.rows(run)
    used = sum(_built_on(w, rows) for w in winners)
    return [
        f"* board (`{FILE}`): {len(entries)} entries; the agents' {len(agents)} "
        f"({counts(agents)}), kernel-agent's {len(own)} ({counts(own)}); "
        f"{used} of {len(winners)} posted winners built on by another session"
    ]
