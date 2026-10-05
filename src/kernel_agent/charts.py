"""Progress charts of a run (matplotlib, optional extra ``viz``).

* ``targets/<id>/progress.png``: module speedup per evaluation. Kept candidates
  (green, annotated with their hypothesis), discarded ones (grey), failures (red ×
  on the floor), the running best and the 1.0× reference.
* ``progress.png``: end-to-end latency over wall-clock time. The projection from
  the best kernels (baseline − Σ est. saved ms, nested targets counted once:
  :mod:`kernel_agent.projection`) as a step line, measured end-to-end runs
  (transforms, integration) as diamonds, baseline lines.
* ``amdahl.png``: the baseline time split by target class share (from the
  profile), before and after the best per-target speedups, plus "other".
* ``integration.png``: waterfall of the greedy end-to-end integration.

Colours are the same everywhere: one green for kept, neutral grey for
discarded, one red for failures. Every function returns ``None`` and writes
nothing when matplotlib is not installed, so the pipeline never depends on it.
"""

from __future__ import annotations

import importlib.util
import math
import textwrap
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from kernel_agent import ledger, projection
from kernel_agent.ledger import DISCARD, E2E, FAILURES, KEEP
from kernel_agent.workspace import RunDir, read_json

DPI = 150

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f3f2ee"

KEEP_COLOR = "#1baf7a"
DISCARD_COLOR = "#b4b2ab"
FAIL_COLOR = "#d03b3b"
PROJECTED_COLOR = "#2a78d6"
COMPILE_COLOR = "#eb6834"
TOTAL_COLOR = "#52514e"
OTHER_COLOR = "#d9d8d1"
# Categorical order for targets (validated for adjacent CVD separation); green
# and red are left out so they keep meaning "kept" and "failed".
TARGET_COLORS = ("#2a78d6", "#eb6834", "#4a3aa7", "#eda100", "#e87ba4", "#008300")
# Marker of each parallel worker of a target (workers.py); the colour stays kept / discarded.
WORKER_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

_RC: dict[Any, Any] = {  # matplotlib types its rc keys as literals
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.labelsize": 10,
    "axes.labelpad": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK_2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "hatch.color": AXIS,
    "hatch.linewidth": 0.8,
}
_lock = threading.Lock()


def available() -> bool:
    """True when matplotlib can be imported (``pip install kernel-agent[viz]``)."""
    try:
        return importlib.util.find_spec("matplotlib") is not None
    except (ImportError, ValueError):
        return False


def write_charts(run: RunDir, targets: Iterable[str] | None = None) -> list[Path]:
    """(Re)write every chart (only the given targets' progress charts if ``targets``)."""
    if not available():
        return []
    rows = ledger.rows(run)
    ids = run.target_ids() if targets is None else list(targets)
    paths = [target_progress(run, t, rows) for t in ids]
    paths += [run_progress(run, rows), amdahl(run, rows), integration(run)]
    return [p for p in paths if p is not None]


# ------------------------------------------------------------------ helpers


def _render(path: Path, size: tuple[float, float], draw: Callable[[Any, Any], None]) -> Path:
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with _lock, matplotlib.rc_context(_RC):
        fig = Figure(figsize=size, dpi=DPI)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        draw(fig, ax)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.tmp.png")
        fig.savefig(tmp, dpi=DPI, bbox_inches="tight", pad_inches=0.2)
        tmp.replace(path)  # never leave a half-written PNG for the dashboard
    return path


def _header(ax: Any, title: str, subtitle: str = "", raise_pt: float = 0.0) -> None:
    """Title and subtitle above the axes (a subtitle of several lines raises the title)."""
    ax.annotate(
        title,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, (26 + 12 * subtitle.count("\n") if subtitle else 10) + raise_pt),
        textcoords="offset points",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 10 + raise_pt),
            textcoords="offset points",
            fontsize=9.5,
            color=INK_2,
            ha="left",
            va="bottom",
        )


def _legend(ax: Any, handles: list[Any], below_pt: float = 46.0, ncol: int | None = None) -> None:
    """Legend in a row under the x axis (a fixed distance in points, whatever the height)."""
    from matplotlib.transforms import offset_copy

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, 0),
        bbox_transform=offset_copy(ax.transAxes, fig=ax.figure, y=-below_pt, units="points"),
        ncol=ncol or len(handles),
        handlelength=1.8,
        columnspacing=1.8,
        borderaxespad=0,
    )


def _text_size(ax: Any, text: str, size: float) -> tuple[float, float]:
    """Width and height of ``text`` in display pixels."""
    from matplotlib.font_manager import FontProperties

    renderer = ax.figure.canvas.get_renderer()
    prop = FontProperties(family=_RC["font.family"], size=size)
    widths, height = [], 0.0
    for line in text.split("\n"):
        w, h, _ = renderer.get_text_width_height_descent(line or " ", prop, ismath=False)
        widths.append(w)
        height += h * 1.2
    return max(widths), height


def _fits(ax: Any, text: str, size: float, x0: float, x1: float, pad_px: float = 8.0) -> bool:
    """Does ``text`` fit horizontally between data x0 and x1 (with padding)?"""
    left, right = ax.transData.transform([(x0, 0), (x1, 0)])[:, 0]
    return bool(_text_size(ax, text, size)[0] + 2 * pad_px <= right - left)


def _box(
    x: float, y: float, w: float, h: float, angle: float, ha: str
) -> list[tuple[float, float]]:
    """Corners (display px) of a text box anchored at its bottom-left / bottom-right corner."""
    ux, uy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    nx, ny = -uy, ux
    if ha == "right":
        x, y = x - w * ux, y - w * uy
    return [
        (x, y),
        (x + w * ux, y + w * uy),
        (x + w * ux + h * nx, y + w * uy + h * ny),
        (x + h * nx, y + h * ny),
    ]


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]], pad: float = 2.0) -> bool:
    """Separating-axis test for two convex polygons (display px)."""
    for poly in (a, b):
        for i in range(len(poly)):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
            nx, ny = y0 - y1, x1 - x0
            norm = math.hypot(nx, ny) or 1.0
            pa = [(nx * x + ny * y) / norm for x, y in a]
            pb = [(nx * x + ny * y) / norm for x, y in b]
            if max(pa) + pad < min(pb) or max(pb) + pad < min(pa):
                return False
    return True


def _place_labels(
    ax: Any,
    items: list[tuple[float, float, float, str, str]],
    taken: list[list[tuple[float, float]]],
    size: float = 8.0,
) -> None:
    """Annotate points without overlapping labels.

    ``items`` are ``(priority, x, y, label, short_label)``. In priority order each
    point gets its full label rotated up and to the right; if that collides with
    something already placed, the short label up-left or below-right; else none.
    """
    scale = ax.figure.dpi / 72
    for _, x, y, label, short in sorted(items, key=lambda it: -it[0]):
        px, py = ax.transData.transform((x, y))
        for text, angle, dx, dy, ha in (
            (label, 28.0, 5, 7, "left"),
            (short, 0.0, -6, 5, "right"),
            (short, 0.0, 6, -16, "left"),
        ):
            w, h = _text_size(ax, text, size)
            box = _box(px + dx * scale, py + dy * scale, w, h, angle, ha)
            if any(_overlap(box, other) for other in taken):
                continue
            taken.append(box)
            ax.annotate(
                text,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                rotation=angle,
                rotation_mode="anchor",
                ha=ha,
                va="bottom",
                fontsize=size,
                color=INK_2,
                zorder=6,
                annotation_clip=False,
            )
            break


def _label_box(ax: Any, annotation: Any) -> list[tuple[float, float]]:
    bbox = annotation.get_window_extent(ax.figure.canvas.get_renderer())
    return [(bbox.x0, bbox.y0), (bbox.x1, bbox.y0), (bbox.x1, bbox.y1), (bbox.x0, bbox.y1)]


def _place_text(
    ax: Any,
    xy: tuple[float, float],
    text: str,
    options: list[tuple[float, float, str, str]],
    taken: list[list[tuple[float, float]]],
    avoid: Sequence[list[tuple[float, float]]] = (),
    **style: Any,
) -> Any:
    """Annotate ``xy`` with ``text`` at the first of ``options`` (``(dx, dy, ha, va)``,
    offsets in points) that stays inside the axes and overlaps no label in ``taken``
    nor a mark in ``avoid``; else the first that overlaps no label; else the first.
    The label's box is added to ``taken``."""
    inside = ax.get_window_extent(ax.figure.canvas.get_renderer())
    tried = []
    for dx, dy, ha, va in options:
        note = ax.annotate(
            text, xy=xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **style
        )
        box = _label_box(ax, note)
        (x0, y0), (x1, y1) = box[0], box[2]
        fits = inside.x0 <= x0 and x1 <= inside.x1 and inside.y0 <= y0 and y1 <= inside.y1
        free = not any(_overlap(box, other) for other in taken)
        tried.append((note, box, fits and free and not any(_overlap(box, o) for o in avoid), free))
        note.set_visible(False)
    note, box, _, _ = next((t for t in tried if t[2]), next((t for t in tried if t[3]), tried[0]))
    for other, _, _, _ in tried:
        if other is not note:
            other.remove()
    note.set_visible(True)
    taken.append(box)
    return note


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _ink_on(color: str) -> str:
    """White or ink text for a label set inside a filled mark."""
    r, g, b = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r**2.2 + 0.7152 * g**2.2 + 0.0722 * b**2.2
    return INK if lum > 0.36 else "#ffffff"


def _num(value: Any) -> float | None:
    return ledger._num(value)


def _thousands() -> Any:
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _: f"{v:,.0f}")


# ------------------------------------------------------------------ per target


def target_progress(
    run: RunDir, target_id: str, rows: list[dict[str, Any]] | None = None
) -> Path | None:
    """``targets/<id>/progress.png``: speedup per evaluation with the running best (benchmark
    evaluations only: no quick checks or duplicates; one marker shape per worker)."""
    rows = [r for r in (ledger.rows(run) if rows is None else rows) if r["target"] == target_id]
    rows = ledger.measured(rows)
    if not rows or not available():
        return None
    spec = read_json(run.target(target_id) / "spec.json", {}) or {}
    return _render(
        run.target(target_id) / "progress.png",
        (10.0, 5.6),
        lambda fig, ax: _draw_target(ax, target_id, spec, rows),
    )


def _draw_target(ax: Any, target_id: str, spec: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    n = len(rows)
    points = list(enumerate(rows, 1))
    kept = [(i, r) for i, r in points if r["status"] == KEEP and r["speedup"] is not None]
    discarded = [(i, r) for i, r in points if r["status"] == DISCARD and r["speedup"] is not None]
    failed = [(i, r) for i, r in points if r["status"] in FAILURES]
    best = kept[-1][1]["speedup"] if kept else None

    values = [r["speedup"] for _, r in kept + discarded] + [1.0]
    lo, hi = min(values), max(values)
    span = max(hi - lo, 0.05 * hi)
    bottom = lo - 0.22 * span
    bottom = 0.0 if bottom < 0.3 * hi else bottom
    top = bottom + (hi - bottom) / 0.62  # room for the rotated annotations
    ax.set_ylim(bottom, top)
    ax.set_xlim(-0.6, n + 0.9)

    ax.axhline(1.0, color=INK_2, lw=1.0, ls=(0, (5, 4)), zorder=1)
    reference = ax.annotate(
        "reference 1.0×",
        xy=(1, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )

    xs = [0] + [i for i, _ in kept] + [n]
    ys = [1.0] + [r["speedup"] for _, r in kept] + [best or 1.0]
    ax.step(xs, ys, where="post", color=KEEP_COLOR, lw=2.0, zorder=3)
    taken = [_label_box(ax, reference)]
    if best is not None:
        best_label = ax.annotate(
            f"best {best:.2f}×",
            xy=(n, best),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )
        taken.append(_label_box(ax, best_label))

    team = sorted({str(r["worker"]) for r in rows if r.get("worker")}, key=int)

    def marker(row: dict[str, Any]) -> str:
        worker = str(row.get("worker") or "")
        return WORKER_MARKERS[(int(worker) - 1) % len(WORKER_MARKERS)] if worker else "o"

    def scatter(points: list[tuple[int, dict[str, Any]]], **style: Any) -> None:
        for shape in dict.fromkeys(marker(r) for _, r in points):
            mine = [(i, r) for i, r in points if marker(r) == shape]
            ax.scatter([i for i, _ in mine], [r["speedup"] for _, r in mine], marker=shape, **style)

    if discarded:
        scatter(discarded, s=34, color=DISCARD_COLOR, edgecolors=SURFACE, linewidths=1.0, zorder=4)
    if kept:
        scatter(kept, s=62, color=KEEP_COLOR, edgecolors=INK, linewidths=0.8, zorder=5)
    if failed:
        ax.scatter(
            [i for i, _ in failed],
            [0.035] * len(failed),
            transform=ax.get_xaxis_transform(),
            marker="x",
            s=38,
            color=FAIL_COLOR,
            linewidths=1.8,
            zorder=5,
            clip_on=False,
        )

    # Kept points carry their hypothesis; the biggest steps win when labels collide.
    items = []
    previous = 1.0
    for i, r in kept:
        text = _short(r["hypothesis"] or r["snapshot"], 46)
        speed = f"{r['speedup']:.2f}×"
        items.append((r["speedup"] - previous, float(i), r["speedup"], f"{speed}  {text}", speed))
        previous = r["speedup"]
    # Labels must not cover points or the running-best line either.
    for i, r in kept + discarded:
        px, py = ax.transData.transform((i, r["speedup"]))
        taken.append(_box(px - 7, py - 7, 14, 14, 0.0, "left"))
    corners = ax.transData.transform(
        [(x, y) for k in range(len(xs) - 1) for x, y in ((xs[k], ys[k]), (xs[k + 1], ys[k]))]
        + [(xs[k], ys[k - 1]) for k in range(1, len(xs) - 1)]
        + [(xs[k], ys[k]) for k in range(1, len(xs) - 1)]
    )
    flat = len(xs) - 1
    for k in range(flat):  # horizontal segments
        (x0, y0), (x1, _) = corners[2 * k], corners[2 * k + 1]
        taken.append(_box(x0, y0 - 1.5, x1 - x0, 3, 0.0, "left"))
    risers = len(xs) - 2
    for k in range(risers):  # vertical segments up to each kept point
        (x0, y0), (_, y1) = corners[2 * flat + k], corners[2 * flat + risers + k]
        taken.append(_box(x0 - 1.5, y0 + 8, 3, max(y1 - y0 - 16, 0.0), 0.0, "left"))
    _place_labels(ax, items, taken)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlabel("evaluation #")
    ax.set_ylabel("module speedup vs. reference (×, higher is better)")

    kinds: dict[str, int] = {}
    for _, r in failed:
        kinds[r["status"]] = kinds.get(r["status"], 0) + 1
    parts = [f"`{spec.get('module_class')}`" if spec.get("module_class") else ""]
    if spec.get("backends"):
        parts.append("backends: " + ", ".join(spec["backends"]))
    if team:
        parts.append(f"{len(team)} workers")
    if kinds:
        parts.append(
            "failures: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in kinds.items())
        )
    best_text = f"{best:.2f}×" if best is not None else "none"
    _header(
        ax,
        f"{target_id}: {n} evaluations, {len(kept)} kept, best {best_text}",
        "  ·  ".join(p for p in parts if p).replace("`", ""),
    )
    _legend(
        ax,
        [
            Line2D([], [], ls="", marker="o", ms=8, mfc=KEEP_COLOR, mec=INK, mew=0.8, label="kept"),
            Line2D(
                [], [], ls="", marker="o", ms=6, mfc=DISCARD_COLOR, mec=SURFACE, label="discarded"
            ),
            Line2D([], [], ls="", marker="x", ms=7, mec=FAIL_COLOR, mew=1.8, label="failed"),
            Line2D([], [], color=KEEP_COLOR, lw=2, label="running best"),
            Line2D([], [], color=INK_2, lw=1, ls=(0, (5, 4)), label="reference module"),
            *(
                Line2D(
                    [],
                    [],
                    ls="",
                    marker=marker({"worker": w}),
                    ms=6,
                    mfc="none",
                    mec=INK_2,
                    label=f"worker {w}",
                )
                for w in team
            ),
        ],
    )


# ------------------------------------------------------------------ run level


def run_progress(run: RunDir, rows: list[dict[str, Any]] | None = None) -> Path | None:
    """``progress.png``: projected and measured end-to-end latency over wall-clock time."""
    baseline = read_json(run.baseline_json, {}) or {}
    base_ms = _num(baseline.get("median_ms"))
    rows = ledger.rows(run) if rows is None else rows
    if base_ms is None or not available():
        return None
    start = ledger.start_time(run, rows)
    if start is None:
        return None
    return _render(
        run.root / "progress.png",
        (10.0, 5.6),
        lambda fig, ax: _draw_run(ax, run, baseline, base_ms, rows, start),
    )


def _compiled_ms(baseline: dict[str, Any]) -> float | None:
    for key in ("compiled_ms", "compile_ms", "torch_compile_ms"):
        value = baseline.get(key)
        if isinstance(value, dict):
            value = value.get("median_ms")
        if _num(value) is not None:
            return _num(value)
    return None


def _draw_run(
    ax: Any,
    run: RunDir,
    baseline: dict[str, Any],
    base_ms: float,
    rows: list[dict[str, Any]],
    start: float,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    def minutes(ts: float | None) -> float:
        return max((ts or start) - start, 0.0) / 60

    timed = [(minutes(ledger.epoch(r["time"])), r) for r in rows]
    spans = [
        (p, minutes(a), None if b is None else minutes(b)) for p, a, b in ledger.phase_spans(run)
    ]
    end = max(
        [t for t, _ in timed]
        + [b for _, _, b in spans if b is not None]
        + [a for _, a, _ in spans],
        default=1.0,
    )
    end = max(end, 1.0)

    # projected: baseline − Σ est. saved of each target's latest kept candidate, nested
    # targets counted once (projection.py)
    tree = projection.tree(run)
    last = projection.project(tree, {}, base_ms)
    px, py = [0.0], [base_ms]
    for r, last in projection.series(tree, base_ms, rows):
        px.append(minutes(ledger.epoch(r["time"])))
        py.append(last.projected_ms)
    e2e = [(t, r) for t, r in timed if r["target"] == E2E]
    measured = [(t, r) for t, r in e2e if r["status"] in (KEEP, DISCARD) and r["new_ms"]]
    failed = [(t, r) for t, r in e2e if r["status"] in FAILURES]
    compiled = _compiled_ms(baseline)

    values = [base_ms, *py, *(r["new_ms"] for _, r in measured)]
    if compiled:
        values.append(compiled)
    lo, hi = min(values), max(values)
    span = max(hi - lo, 0.05 * hi)
    ax.set_ylim(max(lo - 0.18 * span, 0.0), hi + 0.24 * span)
    ax.set_xlim(0, end * 1.03)

    shaded = False
    for phase, a, b in spans:  # shade every other phase wide enough to be labelled
        b = end if b is None else b
        if _fits(ax, phase, 8.5, a, b, pad_px=2):
            if shaded:
                ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
            shaded = not shaded
            ax.annotate(
                phase,
                xy=((a + b) / 2, 1),
                xycoords=("data", "axes fraction"),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=MUTED,
            )

    ax.axhline(base_ms, color=INK_2, lw=1.0, ls=(0, (5, 4)), zorder=1)
    reference = ax.annotate(
        f"baseline {base_ms:,.1f} ms",
        xy=(0, base_ms),
        xycoords=("axes fraction", "data"),
        xytext=(4, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK_2,
    )
    taken = [_label_box(ax, reference)]  # labels placed so far (display px)
    if compiled:
        ax.axhline(compiled, color=COMPILE_COLOR, lw=1.2, ls=(0, (1.5, 2.5)), zorder=1)
        compiled_label = ax.annotate(
            f"torch.compile baseline {compiled:,.1f} ms",
            xy=(0, compiled),
            xycoords=("axes fraction", "data"),
            xytext=(4, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=INK_2,
        )
        taken.append(_label_box(ax, compiled_label))

    px.append(end)
    py.append(py[-1])
    ax.step(px, py, where="post", color=PROJECTED_COLOR, lw=2.0, zorder=3)
    if len(px) > 2:
        ax.scatter(px[1:-1], py[1:-1], s=16, color=PROJECTED_COLOR, zorder=3, lw=0)

    for status, color, size in ((DISCARD, DISCARD_COLOR, 40), (KEEP, KEEP_COLOR, 70)):
        pts = [(t, r) for t, r in measured if r["status"] == status]
        if pts:
            ax.scatter(
                [t for t, _ in pts],
                [r["new_ms"] for _, r in pts],
                marker="D",
                s=size,
                color=color,
                edgecolors=INK if status == KEEP else SURFACE,
                linewidths=0.8,
                zorder=5,
            )
    if failed:
        ax.scatter(
            [t for t, _ in failed],
            [0.965] * len(failed),
            transform=ax.get_xaxis_transform(),
            marker="x",
            s=38,
            color=FAIL_COLOR,
            linewidths=1.8,
            zorder=5,
            clip_on=False,
        )
    # Highlight the integrated result, else the running best measurement.
    final = (read_json(run.root / "integration.json", {}) or {}).get("final") or {}
    kept = [(t, r) for t, r in measured if r["status"] == KEEP]
    best = kept[-1] if kept else None
    if final.get("passed") and final.get("median_ms"):
        best = next(
            ((t, r) for t, r in reversed(measured) if r["new_ms"] == final["median_ms"]), best
        )
    # The two end labels (the projection's and the highlighted measurement) must not
    # overlap each other, the baseline labels, the projected line or the markers.
    (x0, y0), (x1, _) = ax.transData.transform([(px[-2], py[-1]), (px[-1], py[-1])])
    avoid = [_box(x0, y0 - 2, x1 - x0, 4, 0.0, "left")]  # the last step of the projection
    half = 4.5 * ax.figure.dpi / 72  # of a diamond
    for t, r in measured:
        cx, cy = ax.transData.transform((t, r["new_ms"]))
        avoid.append(_box(cx - half, cy - half, 2 * half, 2 * half, 0.0, "left"))
    _place_text(
        ax,
        (px[-1], py[-1]),
        f"projected {py[-1]:,.1f} ms",
        [(-2, 5, "right", "bottom"), (-2, -5, "right", "top")],
        taken,
        avoid,
        fontsize=9,
        color=INK,
    )
    if best is not None:
        t, r = best
        _place_text(
            ax,
            (t, r["new_ms"]),
            f"measured {r['new_ms']:,.1f} ms ({base_ms / r['new_ms']:.2f}×)\n"
            f"{_short(r['snapshot'] or r['hypothesis'], 44)}",
            [
                (-10, -10, "right", "top"),
                (-10, 10, "right", "bottom"),
                (10, -10, "left", "top"),
                (10, 10, "left", "bottom"),
            ],
            taken,
            avoid,
            fontsize=8.5,
            color=INK,
            zorder=6,
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("wall-clock time since the run started (min)")
    ax.set_ylabel("end-to-end latency per run (ms, lower is better)")
    data = run.load() if run.run_json.exists() else {}
    repo = (data.get("card") or {}).get("repo_id", run.root.name)
    final_ms = best[1]["new_ms"] if best else None
    headline = (
        f"{base_ms:,.1f} → {final_ms:,.1f} ms measured ({base_ms / final_ms:.2f}×)"
        if final_ms
        else f"baseline {base_ms:,.1f} ms"
    )
    n_kernel = sum(r["target"] != E2E for r in rows)
    subtitle = (
        f"{n_kernel} kernel evaluations, {len(e2e)} end-to-end runs  ·  "
        f"projected from the best kernels: {py[-1]:,.1f} ms ({base_ms / max(py[-1], 1e-9):.2f}×)"
    )
    if last.used:  # which targets the projection counts
        subtitle += "\n" + _short(f"projected from {last.describe()}", 120)
    _header(ax, f"{repo}: {headline}", subtitle, raise_pt=14 if spans else 0)
    handles = [
        Line2D([], [], color=PROJECTED_COLOR, lw=2, label="projected (kernels)"),
        Line2D(
            [],
            [],
            ls="",
            marker="D",
            ms=7,
            mfc=KEEP_COLOR,
            mec=INK,
            mew=0.8,
            label="measured, new best",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="D",
            ms=6,
            mfc=DISCARD_COLOR,
            mec=SURFACE,
            label="measured, no gain",
        ),
        Line2D([], [], ls="", marker="x", ms=7, mec=FAIL_COLOR, mew=1.8, label="failed"),
        Line2D([], [], color=INK_2, lw=1, ls=(0, (5, 4)), label="baseline"),
    ]
    if compiled:
        handles.append(
            Line2D([], [], color=COMPILE_COLOR, lw=1.2, ls=(0, (1.5, 2.5)), label="torch.compile")
        )
    _legend(ax, handles)


# ------------------------------------------------------------------ Amdahl


def target_shares(run: RunDir) -> list[tuple[str, str, float]]:
    """``(target_id, module_class, share of the profiled time)`` per target.

    Shares come from the module profile (inclusive time of the class over the
    roots' total). Nested targets can overlap; then they are scaled to sum to 1.
    """
    profile = read_json(run.profile_dir / "profile.json", {}) or {}
    classes = profile.get("classes") or []
    roots: dict[str, float] = {}
    for c in classes:
        roots[c.get("root", "")] = max(
            roots.get(c.get("root", ""), 0.0), c.get("inclusive_ms", 0.0)
        )
    total = sum(roots.values())
    out = []
    for target_id in run.target_ids():
        cls = (read_json(run.target(target_id) / "spec.json", {}) or {}).get("module_class")
        ms = sum(c.get("inclusive_ms", 0.0) for c in classes if c.get("cls") == cls)
        if total > 0 and ms > 0:
            out.append((target_id, str(cls), ms / total))
    scale = sum(s for _, _, s in out)
    if scale > 1.0:
        out = [(t, c, s / scale) for t, c, s in out]
    return out


def amdahl(run: RunDir, rows: list[dict[str, Any]] | None = None) -> Path | None:
    """``amdahl.png``: baseline time by target, before vs. after the best kernels."""
    base_ms = _num((read_json(run.baseline_json, {}) or {}).get("median_ms"))
    shares = target_shares(run)
    if base_ms is None or not shares or not available():
        return None
    rows = ledger.rows(run) if rows is None else rows
    best = {t: ledger.best_kept([r for r in rows if r["target"] == t]) for t, _, _ in shares}
    final = (read_json(run.root / "integration.json", {}) or {}).get("final") or {}
    measured = _num(final.get("median_ms")) if final.get("passed", True) else None
    height = 3.9 if measured else 3.3
    return _render(
        run.root / "amdahl.png",
        (10.0, height),
        lambda fig, ax: _draw_amdahl(ax, base_ms, shares, best, measured),
    )


def _draw_amdahl(
    ax: Any,
    base_ms: float,
    shares: list[tuple[str, str, float]],
    best: dict[str, float],
    measured: float | None,
) -> None:
    from matplotlib.patches import Patch

    colors = {t: TARGET_COLORS[i % len(TARGET_COLORS)] for i, (t, _, _) in enumerate(shares)}
    other = base_ms * max(1.0 - sum(s for _, _, s in shares), 0.0)
    before = [(t, base_ms * s) for t, _, s in shares]
    after = [(t, base_ms * s / max(best[t], 1e-9)) for t, _, s in shares]
    after_total = sum(ms for _, ms in after) + other
    bars = [("baseline", before, base_ms), ("best kernels\n(Amdahl estimate)", after, after_total)]
    if measured:
        bars.append(("measured\n(integrated)", [], measured))
    h = 0.5
    labels = []
    ax.set_xlim(0, base_ms * 1.2)  # limits first: label fitting measures in data units
    ax.set_ylim(-0.6, len(bars) - 0.4)
    ax.xaxis.set_major_formatter(_thousands())
    for row, (name, segments, total) in enumerate(bars):
        y = len(bars) - 1 - row
        labels.append((y, name))
        left = 0.0
        if name == "baseline" or segments:
            for t, ms in [*segments, ("other", other)]:
                color = colors.get(t, OTHER_COLOR)
                ax.barh(y, ms, left=left, height=h, color=color, edgecolor=SURFACE, lw=2, zorder=3)
                options = [f"{t}\n{ms:,.0f} ms", f"{ms:,.0f}"]
                if t != "other" and name != "baseline":
                    options.insert(0, f"{t} ÷{best[t]:.2f}\n{ms:,.0f} ms")
                text = next((o for o in options if _fits(ax, o, 8, left, left + ms)), None)
                if text:
                    ax.text(
                        left + ms / 2,
                        y,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=_ink_on(color),
                        zorder=4,
                    )
                left += ms
        else:
            ax.barh(y, total, height=h, color=TOTAL_COLOR, edgecolor=SURFACE, lw=2, zorder=3)
            left = total
        gain = base_ms - total
        if gain > 0:
            ax.barh(
                y,
                gain,
                left=total,
                height=h,
                color=KEEP_COLOR,
                alpha=0.16,
                edgecolor=SURFACE,
                lw=2,
                zorder=2,
            )
            if _fits(ax, f"−{gain:,.0f} ms", 8.5, total, base_ms):
                ax.text(
                    total + gain / 2,
                    y,
                    f"−{gain:,.0f} ms",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=INK_2,
                    zorder=4,
                )
        suffix = "" if name == "baseline" else f"  ({base_ms / total:.2f}×)"
        ax.text(
            max(base_ms, total) + base_ms * 0.012,
            y,
            f"{total:,.1f} ms{suffix}",
            ha="left",
            va="center",
            fontsize=9,
            color=INK,
            fontweight="bold" if name != "baseline" else "normal",
        )

    ax.set_yticks([y for y, _ in labels], [n for _, n in labels])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time per run (ms)")
    covered = sum(s for _, _, s in shares)
    _header(
        ax,
        f"Where the time goes: {base_ms:,.1f} → {after_total:,.1f} ms with the best kernels "
        f"({base_ms / after_total:.2f}×)",
        f"targets cover {covered:.0%} of the profiled time; each target's share ÷ its best "
        "module speedup, the rest unchanged (Amdahl's law)",
    )
    handles = [Patch(color=colors[t], label=f"{t} ({cls})") for t, cls, _ in shares]
    handles += [
        Patch(color=OTHER_COLOR, label="other"),
        Patch(color=KEEP_COLOR, alpha=0.25, label="saved"),
    ]
    _legend(ax, handles, ncol=min(len(handles), 4))


# ------------------------------------------------------------------ integration


def integration_steps(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Waterfall steps of the greedy integration in ``integration.json``.

    The best single item seeds the combination, or the systems agent's best
    measured combination (``composite``) when it was faster; every later history
    entry adds one item to the accepted set and is either accepted (new level),
    rejected for no gain, or failed (quality check / error). A step judged by a
    paired A/B (``ab``) starts at A's median of that session, so its bar is the
    paired difference; its ``rule`` is the acceptance rule.
    """
    base = _num(data.get("baseline_ms"))
    history = data.get("history") or []
    accepted = [a.get("item") for a in data.get("accepted") or []]
    composite = data.get("composite") or {}
    together = composite.get("items")
    if base is None:
        return []
    steps: list[dict[str, Any]] = [{"kind": "total", "label": "baseline", "ms": base}]
    level = base
    seeded = bool(composite.get("seeded"))
    first = together if seeded else accepted[:1]
    seed = next((h for h in history if h.get("items") == first), None) if accepted else None
    if seed is not None and _num(seed.get("median_ms")) is not None:
        step = _step("keep", accepted[0], level, float(seed["median_ms"]))
        steps.append(_together(step, composite) if seeded else step)
        level = float(seed["median_ms"])
    for h in history:
        items = h.get("items") or []
        if len(items) < 2 or (items == together and h is seed):
            continue
        item, ms = items[-1], _num(h.get("median_ms"))
        ab = h.get("ab") or {}
        start = _num(ab.get("a_median_ms"))
        start = level if start is None else start
        if items == together:  # measured, but slower than the best single item
            kind = "nogain" if h.get("passed") and ms is not None else "fail"
            reason = None if kind == "nogain" else h.get("reason") or h.get("status")
            step = _together(_step(kind, item, level, ms, reason), composite)
        elif item in accepted and h.get("passed") and ms is not None:
            step = _step("keep", item, start, ms)
            level = ms
        elif h.get("passed") and ms is not None:
            step = _step("nogain", item, start, ms)
        else:
            step = _step("fail", item, level, ms, h.get("reason") or h.get("status"))
        steps.append({**step, "rule": ab.get("rule")} if ab.get("rule") else step)
    steps.append({"kind": "total", "label": "final", "ms": level})
    return steps


def _step(
    kind: str, item: str, level: float, ms: float | None, reason: Any = None
) -> dict[str, Any]:
    target, sep, _ = item.partition("=")
    source = "kernel" if sep and "/" not in target else "transform"
    return {
        "kind": kind,
        "label": ledger.item_label(item),
        "source": source,
        "from": level,
        "ms": ms,
        "reason": str(reason or ""),
    }


def _together(step: dict[str, Any], composite: dict[str, Any]) -> dict[str, Any]:
    """``step`` relabelled as the measured combination (``integration.json`` ``composite``)."""
    n = len(composite.get("items") or [])
    return {**step, "label": f"exp {composite.get('exp')} combination", "source": f"{n} items"}


def integration(run: RunDir) -> Path | None:
    """``integration.png``: waterfall baseline → + item → … → final."""
    data = read_json(run.root / "integration.json", {}) or {}
    steps = integration_steps(data)
    if len(steps) < 3 or not available():
        return None
    singles = [h for h in data.get("history") or [] if len(h.get("items") or []) == 1]
    return _render(
        run.root / "integration.png",
        (max(7.0, 1.3 * len(steps) + 2.0), 5.4),
        lambda fig, ax: _draw_integration(ax, steps, singles),
    )


def _draw_integration(ax: Any, steps: list[dict[str, Any]], singles: list[dict[str, Any]]) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    base = steps[0]["ms"]
    final = steps[-1]["ms"]
    width = 0.56
    top = base * 1.13
    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, len(steps) - 0.4)
    level = base
    for x, s in enumerate(steps):
        if s["kind"] == "total":
            ax.bar(x, s["ms"], width=width, color=TOTAL_COLOR, zorder=3)
            ax.text(
                x,
                s["ms"] + base * 0.012,
                f"{s['ms']:,.1f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=INK,
            )
        elif s["kind"] == "keep":
            ax.bar(
                x,
                s["from"] - s["ms"],
                bottom=s["ms"],
                width=width,
                color=KEEP_COLOR,
                edgecolor=INK,
                lw=0.8,
                zorder=3,
            )
            ax.text(
                x,
                s["from"] + base * 0.012,
                f"−{s['from'] - s['ms']:,.1f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
            level = s["ms"]
        elif s["kind"] == "nogain":
            lo, hi = sorted((s["from"], s["ms"]))
            ax.bar(
                x,
                max(hi - lo, base * 0.004),
                bottom=lo,
                width=width,
                facecolor="none",
                edgecolor=DISCARD_COLOR,
                hatch="////",
                lw=1.0,
                zorder=3,
            )
            delta = f"{s['ms'] - s['from']:+,.1f}".replace("-", "−")
            ax.text(
                x,
                hi + base * 0.012,
                f"no gain\n{delta} ms",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=MUTED,
            )
        else:
            ax.scatter([x], [level], marker="x", s=60, color=FAIL_COLOR, linewidths=2.2, zorder=4)
            reason = "\n".join(textwrap.wrap(s["reason"], 18)[:3]) or s["reason"]
            ax.text(
                x,
                level + base * 0.03,
                f"failed\n{reason}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=MUTED,
            )
        if x < len(steps) - 1:
            ax.plot(
                [x + width / 2, x + 1 - width / 2],
                [level, level],
                color=MUTED,
                lw=0.9,
                ls=(0, (2, 2)),
                zorder=2,
            )

    names = []
    for s in steps:
        if s["kind"] == "total":
            names.append(s["label"])
        else:
            name = "\n".join(textwrap.wrap(f"+ {s['label']}".replace("_", "_ "), 15))
            names.append(name.replace("_ ", "_") + f"\n({s['source']})")
    ax.set_xticks(range(len(steps)), names)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("end-to-end latency per run (ms)")
    kept = sum(s["kind"] == "keep" for s in steps)
    dropped = [ledger.item_label(h["items"][0]) for h in singles if not h.get("passed")]
    rule = next((s["rule"] for s in steps if s.get("rule")), None)
    test = "gain > 1 %"
    if rule:  # abtest.py
        test = (
            f"win a paired A/B: ≥ {rule['min_win_rate']:.0%} of the rounds, 95 % CI of the "
            f"gain > {rule['min_gain']:.0%}"
        )
    subtitle = (
        f"{len(singles)} items measured alone, {kept} combined greedily (each must pass the "
        f"quality check and {test})"
    )
    if dropped:
        subtitle += f"  ·  failed alone: {_short(', '.join(dropped), 60)}"
    _header(ax, f"Integration: {base:,.1f} → {final:,.1f} ms ({base / final:.2f}×)", subtitle)
    _legend(
        ax,
        [
            Patch(color=TOTAL_COLOR, label="end-to-end total"),
            Patch(facecolor=KEEP_COLOR, edgecolor=INK, lw=0.8, label="accepted (saved ms)"),
            Patch(
                facecolor="none", edgecolor=DISCARD_COLOR, hatch="////", label="rejected, no gain"
            ),
            Line2D([], [], ls="", marker="x", ms=7, mec=FAIL_COLOR, mew=1.8, label="failed"),
        ],
        below_pt=24 + 11 * max(n.count("\n") + 1 for n in names),
    )
