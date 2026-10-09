"""The library scout's step in a run (issue #227): no agent, one GPU lease per target.

:func:`scout_target` runs once per target, after its capture (``optimize``'s kernels phase,
each ``improve`` loop pass before slices, like the library priors):

1. the probe (:mod:`kernel_agent.libscout.probe`, a subprocess): the op families of the
   reference's dominant case, every adapter's availability and decision, the candidate
   files (``candidates/libscout_<adapter>.py``) and the op bars;
2. every adapter that runs goes through ``sweep_candidate``'s machinery
   (:func:`kernel_agent.kernels.sweep.run_sweep`: its configs checked and timed with
   racing, the best one through the full evaluator with every anti-gaming guard) and is
   recorded like an agent's sweep (snapshot, ``results.jsonl``, a ledger row) with the
   hypothesis ``library scout: <what> (<package> <version>) [sweep: ...]``, the session
   ``libscout`` and the backend ``library:<package>@<version>``;
3. ``targets/<id>/libscout.json`` keeps what it found (:func:`summary`) and ``run.json`` →
   ``libscout.scouted`` that it ran.

The probe and the sweeps hold one GPU lock from start to end. The results are the **library
bar** (:func:`bar_lines`): the kernel engineer's digest and first prompt, and the planner's
round context, show it as the floor to beat. It is never a stop signal: scout rows extend no
streak (``budget.LIBRARY_HYPOTHESIS``), and a best the scout set does not retire an arm by
the speed-of-light or speedup-goal rules (``scheduler``).

``python -m kernel_agent.libscout CAPTURE`` runs the same on a capture outside a run (nothing
recorded; the table on stdout).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from kernel_agent.budget import LIBRARY_HYPOTHESIS
from kernel_agent.workspace import RunDir, read_json, write_json

SESSION = "libscout"  # the ledger's session column of scout rows
FILE = "libscout.json"  # in the target's directory
IDEA = "library-"  # + the adapter: the scout rows' idea_id
#: Op bars listed per target in a digest (the fastest library first)
DIGEST_BARS = 6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] libscout: {msg}", flush=True)


# ------------------------------------------------------------------ state


def scouted(run: RunDir, target_id: str) -> dict[str, Any] | None:
    """What :func:`remember` stored for a target (None: not scouted yet)."""
    found = ((run.load().get("libscout") or {}).get("scouted") or {}).get(target_id)
    return found if isinstance(found, dict) else None


def remember(run: RunDir, target_id: str, entry: Mapping[str, Any]) -> None:
    """``run.json`` → ``libscout.scouted[target_id]`` (the scout runs once per target)."""

    def put(data: dict[str, Any]) -> None:
        data.setdefault("libscout", {}).setdefault("scouted", {})[target_id] = dict(entry)

    run.update(put)


def summary(run: RunDir, target_id: str) -> dict[str, Any] | None:
    """``targets/<id>/libscout.json`` (None: never scouted)."""
    data = read_json(run.target(target_id) / FILE, None)
    return data if isinstance(data, dict) else None


# ------------------------------------------------------------------ the step


def scout_target(
    run: RunDir,
    target_id: str,
    *,
    keeper: Any,
    timeout: float = 300.0,
    backends: Mapping[str, bool] | None = None,
    race: bool = True,
    say: Callable[[str], None] = log,
    prober: Callable[..., dict[str, Any]] | None = None,
    sweeper: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scout one target (the module docstring) and return its summary. ``keeper``: the run's
    :class:`~kernel_agent.truth.Truth`; ``timeout``: the evaluation timeout; ``prober`` /
    ``sweeper``: :func:`probe.run_probe` / :func:`sweep.run_sweep` (fakes in tests)."""
    from kernel_agent import precisions
    from kernel_agent.gpulock import gpu_lock
    from kernel_agent.libscout import probe
    from kernel_agent.truth import TamperError

    target_dir = run.target(target_id)
    spec = read_json(target_dir / "spec.json", {}) or {}
    capture = run.capture_file(target_id)
    start = time.monotonic()
    out: dict[str, Any] = {"target": target_id, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not capture.exists():
        out["error"] = "no capture"
        return out
    try:
        sha256 = keeper.expect(capture)
    except TamperError as exc:
        out["error"] = str(exc)
        return out
    prober = prober or probe.run_probe
    with gpu_lock():  # one lease: the probe and every adapter's sweep
        info = prober(
            capture,
            out_dir=target_dir / "candidates",
            target=target_id,
            precision=precisions.of_spec(spec),
            backends=backends,
            sha256=sha256,
            timeout=2 * timeout,
        )
        if "error" in info:
            out.update(error=str(info["error"])[-1500:], seconds=_since(start))
            write_json(target_dir / FILE, out)
            say(f"{target_id}: probe failed: {str(info['error']).splitlines()[-1:]}")
            return out
        results = []
        for decision in info.get("decisions") or []:
            if not decision.get("run"):
                continue
            src = Path(info["candidates"][decision["adapter"]])
            results.append(
                _sweep_adapter(
                    run,
                    target_id,
                    src,
                    decision,
                    info,
                    keeper=keeper,
                    sha256=sha256,
                    timeout=timeout,
                    race=race,
                    sweeper=sweeper,
                )
            )
    out.update(
        case=info.get("case"),
        families=info.get("families"),
        described=info.get("described"),
        capability=info.get("capability"),
        adapters=results,
        skipped=[
            {k: d.get(k) for k in ("adapter", "package", "reason")}
            for d in info.get("decisions") or []
            if not d.get("run")
        ],
        op_bars=info.get("op_bars") or [],
        probe_s=info.get("seconds"),
        seconds=_since(start),
    )
    write_json(target_dir / FILE, out)
    ran = ", ".join(_short(r) for r in results) or "no adapter applies"
    say(f"{target_id}: {info.get('described')}: {ran} ({out['seconds']} s)")
    return out


def _since(start: float) -> float:
    return round(time.monotonic() - start, 1)


def _short(result: Mapping[str, Any]) -> str:
    if result.get("correct") and result.get("speedup") is not None:
        return f"{result['adapter']} {float(result['speedup']):.2f}x"
    return f"{result['adapter']} {result.get('status')}"


def _sweep_adapter(
    run: RunDir,
    target_id: str,
    src: Path,
    decision: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    keeper: Any,
    sha256: str | None,
    timeout: float,
    race: bool,
    sweeper: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    """One adapter's sweep, recorded like ``sweep_candidate``'s (snapshot, results.jsonl,
    ledger row)."""
    from kernel_agent.agent.tools import record_candidate, snapshot
    from kernel_agent.kernels import sweep as sweep_mod
    from kernel_agent.libscout.adapters import BY_NAME
    from kernel_agent.truth import sha256_file

    name = str(decision["adapter"])
    adapter = BY_NAME.get(name)
    version = ((info.get("available") or {}).get(name) or {}).get("version")
    configs = list(decision.get("configs") or [{}])
    snaps: list[tuple[Path, str]] = []

    def prepare(bound: Path) -> Path:  # the best config bound into the source: the snapshot
        snap = snapshot(run, bound, target_id)
        snaps.append((snap, sha256_file(snap)))
        return snap

    start = time.monotonic()
    data = (sweeper or sweep_mod.run_sweep)(
        run.capture_file(target_id),
        src,
        configs,
        timeout=timeout,
        capture_sha256=sha256,
        prepare=prepare,
        race=race,
    )
    result = data["evaluation"]
    snap, snap_sha256 = snaps[0]
    if result.get("status") == "tampered":
        keeper.alarm(run.capture_file(target_id), str(result.get("error")))
    elif sha256_file(snap) != snap_sha256:
        keeper.alarm(snap, "snapshot changed during its evaluation")
        result = {
            "status": "tampered",
            "correct": False,
            "error": f"{snap.name} changed while it was evaluated; result discarded",
        }
    sweep = data["sweep"]
    config = data["config"]
    result = {**result, "config": config, "sweep": sweep}
    title = adapter.title if adapter else name
    package = adapter.package if adapter else "?"
    tag = f"{sweep_mod.label(config)}; best of {sweep['passed']}/{sweep['configs']} configs"
    record, row = record_candidate(
        run,
        target_id,
        src,
        snap,
        result,
        hypothesis=f"{LIBRARY_HYPOTHESIS}{title} ({package} {version}) [sweep: {tag}]",
        eval_s=_since(start),
        snapshot_sha256=snap_sha256,
        keeper=keeper,
        idea=IDEA + name,
        session=SESSION,
        title=f"library {name}: {sweep_mod.label(config)}",  # its name in the ledger (#222)
    )
    table = [sweep_mod.compact_row(r) for r in sweep.get("table") or []]
    return {
        "adapter": name,
        "title": title,
        "package": package,
        "version": version,
        "backend": row.get("backend"),
        "snapshot": f"history/{snap.name}",
        "exp": row.get("exp"),
        "status": row.get("status"),
        "correct": bool(record.get("correct")),
        "speedup": record.get("speedup"),
        "pct_of_sol": record.get("pct_of_sol"),
        "custom_kernel_share": record.get("custom_kernel_share"),
        "config": config,
        "configs": table,
        "pruned": decision.get("pruned") or [],  # configs its op bars spared the sweep
        "error": _last_line(record.get("error")),
        "seconds": _since(start),
    }


def _last_line(text: Any) -> str | None:
    lines = str(text or "").strip().splitlines()
    return lines[-1][:240] if lines else None


# ------------------------------------------------------------------ the bar


def best(found: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The fastest correct adapter result of a :func:`summary` (None: none correct)."""
    correct = [
        r
        for r in (found or {}).get("adapters") or []
        if r.get("correct") and r.get("speedup") is not None
    ]
    return max(correct, key=lambda r: float(r["speedup"]), default=None)


def _config(config: Any) -> str:
    if not isinstance(config, Mapping) or not config:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in config.items())


def _sol(result: Mapping[str, Any]) -> str:
    pct = result.get("pct_of_sol")
    return f", {float(pct):.0f} % of SOL" if isinstance(pct, int | float) else ""


def _result_line(result: Mapping[str, Any]) -> str:
    """``torch-sdpa (BACKEND='cudnn'): 1.04x, 31 % of SOL (keep)``."""
    config = _config(result.get("config"))
    head = f"{result.get('adapter')}" + (f" ({config})" if config else "")
    if result.get("correct") and result.get("speedup") is not None:
        return f"{head}: {float(result['speedup']):.2f}x{_sol(result)} ({result.get('status')})"
    why = f": {result['error']}" if result.get("error") else ""
    # the best config failed its full evaluation: what the others measured in the sweep
    others = [
        f"{_config(row.get('config')) or 'default'} {float(row['speedup']):.2f}x"
        for row in result.get("configs") or []
        if row.get("correct")
        and row.get("speedup") is not None
        and _config(row.get("config")) != config
    ]
    seen = f" (sweep: {', '.join(others)})" if others else ""
    return f"{head}: {result.get('status')}{why}{seen}"


def bar_groups(bars: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Op bars grouped by call signature: ``{signature, calls, ref_us, libraries: [...]}``
    with the libraries fastest first (kernel time when the CUDA graph captured, else the
    eager loop's)."""
    groups: dict[str, dict[str, Any]] = {}
    for bar in bars:
        if "signature" not in bar:
            continue
        group = groups.setdefault(
            str(bar["signature"]),
            {"signature": bar["signature"], "calls": bar.get("calls"), "libraries": []},
        )
        graph = bar.get("us") is not None
        group.setdefault("kernel_time", graph)
        group["ref_us"] = bar.get("ref_us") if graph else bar.get("ref_eager_us")
        group["ref_eager_us"] = bar.get("ref_eager_us")
        group["libraries"].append(
            {
                "adapter": bar.get("adapter"),
                "config": bar.get("config"),
                "ok": bool(bar.get("ok")),
                "us": bar.get("us") if graph else bar.get("eager_us"),
                "eager_us": bar.get("eager_us"),
                "speedup": bar.get("speedup") if graph else bar.get("eager_speedup"),
                "error": bar.get("error"),
            }
        )
    for group in groups.values():
        group["libraries"].sort(
            key=lambda lib: (not lib["ok"], lib["us"] is None, lib["us"] or 0.0)
        )
    return list(groups.values())


def _bar_line(group: Mapping[str, Any]) -> str:
    libs = []
    for lib in group["libraries"]:
        who = _config(lib.get("config")) or lib.get("adapter")
        if not lib["ok"]:
            libs.append(f"{who} fails ({str(lib.get('error') or '')[:80]})")
        elif lib.get("us") is not None:
            libs.append(f"{who} {lib['us']:.1f} us ({float(lib['speedup']):.2f}x)")
    ref = group.get("ref_us")
    how = "kernel time, CUDA graph" if group.get("kernel_time") else "eager, host time included"
    line = f"`{group['signature']}`: reference {ref} us; " + "; ".join(libs) + f" [{how}]"
    eager = [
        f"{_config(lib.get('config')) or lib.get('adapter')} {lib['eager_us']:.1f}"
        for lib in group["libraries"]
        if lib["ok"] and lib.get("eager_us") is not None
    ]
    if group.get("kernel_time") and eager and group.get("ref_eager_us") is not None:
        line += f"; eager per call: reference {group['ref_eager_us']}, " + ", ".join(eager)
    return line


def bar_lines(run: RunDir, target_id: str) -> list[str]:
    """``## Library bar`` of a target for a kernel engineer ([] before the scout ran)."""
    found = summary(run, target_id)
    if found is None:
        return []
    lines = [
        "",
        "## Library bar (the library scout: library kernels with no agent; a floor to beat, "
        "never a ceiling)",
    ]
    if found.get("error"):
        return [*lines, f"* The scout failed on this target: {_last_line(found['error'])}"]
    top = best(found)
    if top is not None:
        lines.append(
            f"* Best: `{top['snapshot']}` ({top.get('title')}, `{top.get('backend')}`"
            + (f", {_config(top.get('config'))}" if _config(top.get("config")) else "")
            + f"): {float(top['speedup']):.2f}x{_sol(top)}. Copy it or call its library inside "
            "your own kernel, then beat it."
        )
    results = found.get("adapters") or []
    if results:
        lines.append("* Measured (module level, eager): " + "; ".join(map(_result_line, results)))
    elif found.get("described"):
        lines.append(f"* The reference calls {found['described']}; no library adapter applies.")
    groups = bar_groups(found.get("op_bars") or [])
    if groups:
        lines.append("* Op bars (the reference's op vs the library's on its recorded inputs):")
        lines += [f"  - {_bar_line(g)}" for g in groups[:DIGEST_BARS]]
    if skipped := found.get("skipped"):
        lines.append(
            "* Not run: " + "; ".join(f"{s.get('adapter')} ({s.get('reason')})" for s in skipped)
        )
    return lines


def prompt_note(run: RunDir, target_id: str) -> str:
    """:func:`bar_lines` as a section of the engineer's first prompt ("" before the scout)."""
    lines = bar_lines(run, target_id)
    return "\n".join(lines) + "\n" if lines else ""


def planner_lines(run: RunDir, target_ids: Iterable[str]) -> list[str]:
    """The library bar of every scouted target, one line each (the planner's round context
    and ``ceilings.md``)."""
    out = []
    for target_id in target_ids:
        found = summary(run, target_id)
        if found is None or found.get("error"):
            continue
        top = best(found)
        groups = bar_groups(found.get("op_bars") or [])
        fastest = [
            f"{g['libraries'][0].get('adapter')} {_config(g['libraries'][0].get('config'))} "
            f"{float(g['libraries'][0]['speedup']):.2f}x on `{g['signature']}`"
            for g in groups
            if g["libraries"]
            and g["libraries"][0]["ok"]
            and g["libraries"][0].get("speedup")
            and float(g["libraries"][0]["speedup"]) > 1.05
        ]
        bar = (
            f"{top['adapter']} {float(top['speedup']):.2f}x{_sol(top)}"
            if top
            else "no library candidate faster than the reference"
        )
        line = f"* `{target_id}` ({found.get('described') or 'no family'}): {bar}"
        if fastest:
            line += "; op bars: " + "; ".join(fastest[:3])
        out.append(line)
    return out


CEILINGS_HEAD = "## Library bars (libscout)"


def write_ceilings(run: RunDir, profile_dir: Path | None = None) -> None:
    """Replace (or append) the library bars section of ``<profile_dir>/ceilings.md``; by
    default of the run's profile and of the newest improve round's (what the next planner
    reads)."""
    dirs = [profile_dir] if profile_dir is not None else [run.profile_dir]
    if profile_dir is None:
        rounds = [p for p in (run.root / "rounds").glob("*/profile") if p.parent.name.isdigit()]
        if rounds:
            dirs.append(max(rounds, key=lambda p: int(p.parent.name)))
    lines = planner_lines(run, run.target_ids())
    for directory in dirs:
        path = directory / "ceilings.md"
        if not directory.is_dir():
            continue
        text = path.read_text() if path.is_file() else ""
        text = text.split("\n" + CEILINGS_HEAD)[0].split(CEILINGS_HEAD)[0].rstrip("\n")
        if lines:
            text += (
                f"\n\n{CEILINGS_HEAD}\n\nWhat library kernels reach on each target with no "
                "agent (module speedup of the best scout candidate; op bars: the library's op "
                "alone vs the reference's). A floor to beat, not a ceiling.\n\n" + "\n".join(lines)
            )
        path.write_text(text.lstrip("\n") + "\n")


# ------------------------------------------------------------------ export and doctor


def requirements(sources: Mapping[str, str]) -> list[dict[str, Any]]:
    """The libraries the exported files declare (``KA_LIBRARY`` / ``KA_LICENCE``): one entry
    per package with its exact version, licence and the files that need it."""
    from kernel_agent.libscout.template import library_of, licence_of

    found: dict[str, dict[str, Any]] = {}
    for file, text in sources.items():
        label = library_of(text)
        if not label:
            continue
        package, _, version = label.partition("@")
        entry = found.setdefault(
            package,
            {"package": package, "version": version, "licence": licence_of(text), "by": []},
        )
        entry["by"].append(file)
    return sorted(found.values(), key=lambda e: e["package"])


def requirements_text(entries: Iterable[Mapping[str, Any]]) -> str:
    """``requirements.txt`` of :func:`requirements`: exact versions (a local build tag such
    as ``+cu130`` named in the comment: pip resolves the public version)."""
    lines = ["# Libraries the exported kernels call (kernel-agent's library scout, #227)."]
    for e in entries:
        version = str(e.get("version") or "")
        public, _, local = version.partition("+")
        pin = f"{e['package']}=={public}" if public and public != "unknown" else e["package"]
        notes = [f"licence {e.get('licence') or 'unknown'}"]
        if local:
            notes.append(f"built as {version}")
        notes.append("used by " + ", ".join(e.get("by") or []))
        lines.append(f"{pin}  # {'; '.join(notes)}")
    return "\n".join(lines) + "\n"


def doctor_lines(
    capability: tuple[int, ...] | None, backends: Mapping[str, bool] | None
) -> list[str]:
    """``doctor``: every library the scout knows, its installed version (or why not), and
    which adapters can run on this GPU (the target decides the rest: its op families)."""
    from kernel_agent.libscout.adapters import ADAPTERS
    from kernel_agent.libscout.registry import availability

    probes = availability(ADAPTERS)
    lines = ["library scout (#227): libraries and the adapters that can run on this GPU"]
    by_package: dict[str, list[Any]] = {}
    for a in ADAPTERS:
        by_package.setdefault(a.package, []).append(a)
    for package, adapters in by_package.items():
        probe = probes[adapters[0].name]
        state = probe.version or "not installed"
        lines.append(f"  {package} {state} ({adapters[0].licence})")
        for a in adapters:
            mine = probes[a.name]
            why = (
                a.no_template
                or a.arch_reason(capability)
                or mine.reason
                or (
                    "no CUDA toolkit for its extension"
                    if a.compiles and backends is not None and not backends.get("cuda")
                    else None
                )
            )
            fams = "/".join(a.families)
            verified = "" if a.verified else ", not verified on a GPU"
            status = "runs here" if why is None else f"skipped: {why}"
            lines.append(f"    {a.name} [{fams}{verified}]: {status}")
    return lines


# ------------------------------------------------------------------ outside a run


def main(argv: list[str] | None = None) -> int:
    """Scout a capture outside a run: the probe and each adapter's sweep, nothing recorded."""
    from kernel_agent import toolchain
    from kernel_agent.gpulock import gpu_lock
    from kernel_agent.kernels import sweep as sweep_mod
    from kernel_agent.libscout import probe

    parser = argparse.ArgumentParser(
        prog="python -m kernel_agent.libscout",
        description="The library scout on one capture: op families, op bars, and every "
        "applicable library adapter swept and fully evaluated (nothing recorded).",
    )
    parser.add_argument("capture", type=Path)
    parser.add_argument("--precision", default=None, help="the target's precision (exact)")
    parser.add_argument("--out-dir", type=Path, default=None, help="keep the candidates here")
    parser.add_argument("--timeout", type=float, default=300.0, help="evaluation timeout (s)")
    parser.add_argument("--no-sweep", action="store_true", help="the probe and op bars only")
    parser.add_argument("--json", type=Path, default=None, help="write the result here")
    ns = parser.parse_args(argv)
    tc = toolchain.setup()
    out_dir = ns.out_dir or Path(tempfile.mkdtemp(prefix="ka-libscout-"))
    start = time.monotonic()
    result: dict[str, Any] = {}
    try:
        with gpu_lock():
            info = probe.run_probe(
                ns.capture,
                out_dir=out_dir,
                target=ns.capture.stem,
                precision=ns.precision,
                backends=tc.backends,
                timeout=2 * ns.timeout,
            )
            result["probe"] = info
            if "error" in info:
                print(info["error"], file=sys.stderr)
                return 1
            print(f"{ns.capture.name}: {info['described']} (probe {info['seconds']} s)")
            for d in info["decisions"]:
                if not d["run"]:
                    print(f"  skipped {d['adapter']}: {d['reason']}")
                for p in d.get("pruned") or []:
                    print(f"  not swept: {d['adapter']} {_config(p['config'])}: {p['why']}")
            for group in bar_groups(info.get("op_bars") or []):
                print("  op bar " + _bar_line(group))
            sweeps = result["sweeps"] = {}
            for d in info["decisions"] if not ns.no_sweep else []:
                if not d["run"]:
                    continue
                src = Path(info["candidates"][d["adapter"]])
                t0 = time.monotonic()
                data = sweep_mod.run_sweep(
                    ns.capture, src, d["configs"], timeout=ns.timeout, race=True
                )
                data["seconds"] = round(time.monotonic() - t0, 1)
                sweeps[d["adapter"]] = data
                print(f"\n{d['adapter']} ({data['seconds']} s)")
                print(sweep_mod.format_table(data))
        result["seconds"] = round(time.monotonic() - start, 1)
        print(f"\nscout: {result['seconds']} s")
        if ns.json:
            ns.json.write_text(json.dumps(result, indent=2, default=str))
    finally:
        if ns.out_dir is None:
            shutil.rmtree(out_dir, ignore_errors=True)
    return 0
