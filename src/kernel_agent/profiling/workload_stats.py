"""Workload statistics of a captured module: how the model really calls it.

A capture keeps a few cases. This module summarises *every* call of the
target's instances during the capture run, in the spirit of auto-gpu-kernel's
workload inspector, which found its biggest wins in properties of the real
inputs:

* the call mix per entrypoint and primary-input signature, with its phase
  (``prefill`` / ``decode``, :mod:`kernel_agent.phases`) and share of calls;
* mask arguments: ``None``, all ones, causal, prefix (a decode step that may
  attend to the first *L* slots only) or other;
* layouts (contiguity, strides) and dtypes of tensor arguments, zero biases;
* integer arguments (positions, ``cache_position``): min / max / distinct;
* KV-cache arguments (``kv_cache``, ``past_key_value``): the number of cache
  slots and the valid length per call (``max(position) + 1``, or the mask's
  prefix length), e.g. VoxCPM2's decode attention reads 8192 slots of which
  at most ~130 hold data.

Statistics that need tensor values are computed on the device without a sync
(a clone of a tiny position tensor, a mask or bias reduction: a few small
kernels per call) and read back once in :meth:`WorkloadStats.finalize`.
"""

from __future__ import annotations

import collections
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent.kernels.compare import flatten
from kernel_agent.profiling.profiler import call_signature, signature_of

MASK_NAME = re.compile(r"mask", re.I)
BIAS_NAME = re.compile(r"bias", re.I)
POSITION_NAME = re.compile(r"pos", re.I)
CACHE_NAME = re.compile(r"cache|past_key|key_value|kv", re.I)

#: Device-side samples kept per argument (positions, mask kinds, ...).
MAX_SAMPLES = 100_000
#: Integer tensors up to this many elements are sampled for their value range.
MAX_INT_NUMEL = 4096
#: Masks larger than this are not classified (temporaries of the same size).
MAX_MASK_NUMEL = 1 << 26
#: Tensors inspected per argument (a cache object can hold one pair per layer).
MAX_TENSORS = 16
#: Distinct Python scalar values tracked per argument.
MAX_VALUES = 12

MASK_KINDS = ("all-ones", "causal", "prefix", "other")


def _mask_kind(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(index into MASK_KINDS, allowed slots of the last row)`` as 0-dim device tensors.

    Boolean masks keep ``True``, integer masks non-zero, float masks are
    additive (``0`` keeps, ``<= -1e4`` drops; any other value is a bias, so
    "other"). A mask is a *prefix* when every row allows exactly its first
    ``n`` slots, *causal* when it is a prefix whose ``n`` grows by one per row
    (with an offset when there are more slots than rows)."""
    if m.dtype == torch.bool:
        allowed, bias = m, None
    elif m.is_floating_point():
        allowed, bias = m == 0, ((m != 0) & (m > -1e4)).any()
    else:
        allowed, bias = m != 0, None
    if allowed.dim() < 2:
        allowed = allowed.reshape(1, -1)
    slots, rows = allowed.shape[-1], allowed.shape[-2]
    allowed = allowed.reshape(-1, rows, slots)
    n = allowed.sum(-1)
    prefix = (allowed == (torch.arange(slots, device=m.device) < n.unsqueeze(-1))).all()
    if rows > 1:
        causal = prefix & ((n[:, 1:] - n[:, :-1]) == 1).all()
    else:
        causal = torch.zeros((), dtype=torch.bool, device=m.device)
    kind = torch.where((n == slots).all(), 0, torch.where(causal, 1, torch.where(prefix, 2, 3)))
    if bias is not None:
        kind = torch.where(bias, 3, kind)
    return kind, n[:, -1].max()


@dataclass
class _Arg:
    """Statistics of one argument of one entrypoint over all calls."""

    calls: int = 0
    none: int = 0
    tensors: int = 0
    contiguous: int = 0
    layout: str | None = None  # first non-contiguous layout seen
    dtypes: collections.Counter[str] = field(default_factory=collections.Counter)
    values: collections.Counter[str] = field(default_factory=collections.Counter)
    int_seen: set[int] = field(default_factory=set)  # Python int values
    int_samples: list[torch.Tensor] = field(default_factory=list)
    masks: list[torch.Tensor] = field(default_factory=list)
    zeros: list[torch.Tensor] = field(default_factory=list)
    slots: collections.Counter[int] = field(default_factory=collections.Counter)
    valid: list[Any] = field(default_factory=list)  # ints or 0-dim device tensors
    position: str | None = None  # argument the valid length came from


class WorkloadStats:
    """Collects statistics of every call it is shown (:meth:`observe`)."""

    def __init__(self) -> None:
        self.total = 0
        self.groups: dict[tuple[str, str], dict[str, Any]] = {}
        self.args: dict[tuple[str, str], _Arg] = {}
        self._names: dict[tuple[type, str], list[str]] = {}
        self._cuda = torch.cuda.is_available()

    # ------------------------------------------------------------ per call

    def _param_names(self, module: nn.Module, method: str) -> list[str]:
        key = (type(module), method)
        if key not in self._names:
            names: list[str] = []
            try:
                params = list(inspect.signature(getattr(type(module), method)).parameters.values())
                names = [
                    p.name
                    for p in params[1:]  # self
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
            except (AttributeError, TypeError, ValueError):
                pass
            self._names[key] = names
        return self._names[key]

    def observe(
        self,
        module: nn.Module,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        phase: str,
    ) -> None:
        """Record one call (host-side metadata now, device statistics without a sync)."""
        self.total += 1
        key = (method, call_signature(method, args, kwargs, limit=1))
        group = self.groups.setdefault(
            key, {"phase": phase, "calls": 0, "instances": set(), "shapes": set()}
        )
        group["calls"] += 1
        group["instances"].add(id(module))
        if len(group["shapes"]) < 32:
            group["shapes"].add(signature_of(args, kwargs, limit=8))
        device_ok = not (self._cuda and torch.cuda.is_current_stream_capturing())
        names = self._param_names(module, method)
        named = [(names[i] if i < len(names) else f"a{i}", v) for i, v in enumerate(args)]
        position: tuple[str, Any] | None = None
        mask_valid: torch.Tensor | None = None
        cache: _Arg | None = None
        for name, value in [*named, *kwargs.items()]:
            arg = self.args.setdefault((method, name), _Arg())
            arg.calls += 1
            if value is None:
                arg.none += 1
                continue
            if isinstance(value, bool | int | float | str):
                self._scalar(arg, value)
                if POSITION_NAME.search(name) and type(value) is int and position is None:
                    position = (name, value)
                continue
            tensors = [value] if isinstance(value, torch.Tensor) else []
            if not tensors:
                tensors = [t for t in flatten(value, name).values() if isinstance(t, torch.Tensor)]
            if not tensors:
                self._count(arg.values, type(value).__name__)
                continue
            for t in tensors[:MAX_TENSORS]:
                arg.tensors += 1
                arg.dtypes[str(t.dtype).removeprefix("torch.")] += 1
                if t.is_contiguous():
                    arg.contiguous += 1
                elif arg.layout is None:
                    arg.layout = f"{list(t.shape)} strides {tuple(t.stride())}"
            first = tensors[0]
            if CACHE_NAME.search(name):
                slots = max((int(t.shape[-2]) for t in tensors if t.dim() >= 3), default=0)
                if slots:
                    arg.slots[slots] += 1
                    cache = arg
                    continue
            if not device_ok or first.device.type == "meta" or len(tensors) > 1:
                continue
            integer = first.dtype != torch.bool and not (
                first.is_floating_point() or first.is_complex()
            )
            if MASK_NAME.search(name):
                if first.numel() <= MAX_MASK_NUMEL and len(arg.masks) < MAX_SAMPLES:
                    kind, mask_valid = _mask_kind(first.detach())
                    arg.masks.append(kind)
            elif BIAS_NAME.search(name) and first.is_floating_point():
                if len(arg.zeros) < MAX_SAMPLES:
                    arg.zeros.append((first.detach() == 0).all())
            elif integer and first.numel() <= MAX_INT_NUMEL and len(arg.int_samples) < MAX_SAMPLES:
                sample = first.detach().reshape(-1).clone()
                arg.int_samples.append(sample)
                if POSITION_NAME.search(name) and position is None and sample.numel():
                    position = (name, sample.max())
        if cache is not None and len(cache.valid) < MAX_SAMPLES:
            if position is not None:
                cache.valid.append(position[1] + 1)
                cache.position = cache.position or position[0]
            elif mask_valid is not None:
                cache.valid.append(mask_valid)
                cache.position = cache.position or "mask"

    @staticmethod
    def _count(counter: collections.Counter[str], value: str) -> None:
        if value in counter or len(counter) < MAX_VALUES:
            counter[value] += 1
        else:
            counter["(other)"] += 1

    def _scalar(self, arg: _Arg, value: bool | int | float | str) -> None:
        self._count(arg.values, repr(value))
        if type(value) is int:
            arg.int_seen.add(value)

    # ------------------------------------------------------------ summary

    def finalize(self, module: nn.Module | None = None, **meta: Any) -> dict[str, Any]:
        """JSON-able summary (reads the device-side statistics back: a few syncs, once)."""
        total = max(self.total, 1)
        groups = sorted(self.groups.items(), key=lambda kv: -kv[1]["calls"])
        methods: dict[str, dict[str, Any]] = {}
        called: set[int] = set()
        for (method, _), g in groups:
            called |= g["instances"]
            m = methods.setdefault(
                method, {"calls": 0, "instances": set(), "phases": collections.Counter()}
            )
            m["calls"] += g["calls"]
            m["instances"] |= g["instances"]
            m["phases"][g["phase"]] += g["calls"]
        args = []
        for (method, name), a in self.args.items():
            args.append({"method": method, "name": name, **_arg_summary(a)})
        profile: dict[str, Any] = {
            **meta,
            "calls": self.total,
            "instances_called": len(called),
            "methods": {
                method: {
                    "calls": m["calls"],
                    "share": round(m["calls"] / total, 4),
                    "instances": len(m["instances"]),
                    "phases": dict(m["phases"].most_common()),
                }
                for method, m in methods.items()
            },
            "groups": [
                {
                    "method": method,
                    "signature": sig.removeprefix(f"{method}: "),
                    "phase": g["phase"],
                    "calls": g["calls"],
                    "share": round(g["calls"] / total, 4),
                    "instances": len(g["instances"]),
                    "shapes": len(g["shapes"]),
                    "examples": sorted(g["shapes"])[:3],
                }
                for (method, sig), g in groups
            ],
            "args": args,
            "params": _param_summary(module) if module is not None else {},
        }
        profile["facts"] = facts(profile)
        return profile


def _stack(values: list[Any]) -> torch.Tensor:
    """Python ints and 0-dim device tensors -> one CPU int64 tensor (one copy per device)."""
    by_device: dict[torch.device, list[torch.Tensor]] = collections.defaultdict(list)
    for v in values:
        if isinstance(v, torch.Tensor):
            by_device[v.device].append(v.reshape(1).to(torch.int64))
    host = torch.tensor([int(v) for v in values if not isinstance(v, torch.Tensor)])
    return torch.cat([host.to(torch.int64), *(torch.cat(t).cpu() for t in by_device.values())])


def _arg_summary(a: _Arg) -> dict[str, Any]:
    out: dict[str, Any] = {"calls": a.calls, "none": a.none}
    if a.tensors:
        out["dtypes"] = dict(a.dtypes.most_common())
        out["contiguous"] = round(a.contiguous / a.tensors, 4)
        if a.layout:
            out["layout"] = a.layout
    if a.values:
        out["values"] = dict(a.values.most_common())
    seen = set(a.int_seen)
    if a.int_samples:
        device = a.int_samples[0].device
        seen |= set(torch.cat([s.to(device) for s in a.int_samples]).unique().tolist())
    if seen:
        out["ints"] = {"min": min(seen), "max": max(seen), "distinct": len(seen)}
    if a.masks:
        kinds = collections.Counter(_stack(a.masks).tolist())
        out["masks"] = {MASK_KINDS[k]: n for k, n in sorted(kinds.items())}
    if a.zeros:
        zero = _stack(a.zeros)
        out["zero_frac"] = round(float(zero.float().mean()), 4)
    if a.slots:
        cache: dict[str, Any] = {"slots": sorted(a.slots)}
        if a.valid:
            valid = _stack(a.valid)
            cache.update(
                valid_min=int(valid.min()),
                valid_max=int(valid.max()),
                valid_median=int(valid.median()),
                valid_from=a.position,
            )
        out["cache"] = cache
    return out


def _param_summary(module: nn.Module) -> dict[str, Any]:
    biases = [(n, p) for n, p in module.named_parameters() if n.split(".")[-1] == "bias"]
    zero = [n for n, p in biases if p.numel() and not bool(p.detach().any())]
    return {"biases": len(biases), "zero_biases": zero}


# ---------------------------------------------------------------- facts + markdown


def _pct(x: float) -> str:
    if x == 0 or x >= 0.1:
        return f"{100 * x:.0f} %"
    return f"{100 * x:.1f} %" if x >= 0.001 else "<0.1 %"


def _sig(sig: str) -> str:
    return re.sub(r":(bfloat16|float16|float32|float64|int64|int32|bool)", "", sig)


def facts(p: dict[str, Any]) -> list[str]:
    """Short, actionable statements, most important first (the engineer prompt shows the top)."""
    out: list[str] = []
    total = max(int(p.get("calls", 0)), 1)
    parts = []
    for method, m in sorted(p.get("methods", {}).items(), key=lambda kv: -kv[1]["calls"]):
        groups = [g for g in p.get("groups", []) if g["method"] == method]
        phases = "/".join(m["phases"])
        shapes = ", ".join(f"`{_sig(g['signature'])}` ×{g['calls']}" for g in groups[:2])
        more = f" (+{len(groups) - 2} more)" if len(groups) > 2 else ""
        parts.append(
            f"`{method}` {m['calls']} calls ({_pct(m['calls'] / total)}, {phases}, "
            f"{m['instances']} instances): {shapes}{more}"
        )
    if parts:
        out.append(f"Call mix over {total} calls: " + "; ".join(parts))
    args = p.get("args", [])
    ranges = {(a["method"], a["name"]): a.get("ints") for a in args}
    covered: set[tuple[str, str]] = set()
    for a in args:
        cache = a.get("cache")
        if not cache:
            continue
        slots = cache["slots"]
        where = f"`{a['method']}({a['name']})`"
        if "valid_max" not in cache:
            out.append(f"{where}: KV cache of {'/'.join(map(str, slots))} slots")
            continue
        src = cache.get("valid_from")
        pos = ranges.get((a["method"], src)) if src else None
        if src and src != "mask":
            covered.add((a["method"], src))
        via = f" (`{src}` {pos['min']}..{pos['max']})" if pos else ""
        lo, hi = cache["valid_min"], cache["valid_max"]
        if len(slots) == 1 and slots[0] > hi:
            out.append(
                f"{where}: static cache of {slots[0]} slots, valid length {lo}..{hi}{via}: "
                f"≤ {_pct(hi / slots[0])} of the slots hold data → attend over the valid "
                "length only, not the whole cache"
            )
        else:
            out.append(
                f"{where}: cache of {slots[0]}..{slots[-1]} slots, valid length "
                f"{lo}..{hi}{via} (the cache grows with the sequence)"
            )
    for a in args:
        masks = a.get("masks")
        if not masks and not (a["none"] and MASK_NAME.search(a["name"])):
            continue
        kinds = {"None": a["none"], **(masks or {})}
        if not masks and a["none"] == a["calls"]:
            out.append(f"`{a['method']}({a['name']})`: always None (no mask to apply)")
            continue
        n = max(a["calls"], 1)
        text = ", ".join(
            f"{k} {_pct(c / n)}" for k, c in sorted(kinds.items(), key=lambda kv: -kv[1]) if c
        )
        hint = ""
        if kinds.get("causal"):
            hint = " → a causal kernel need not read the mask (check it at run time)"
        elif kinds.get("all-ones") or kinds.get("None"):
            hint = " → an unmasked fast path pays off (check the mask at run time)"
        elif kinds.get("prefix"):
            hint = " → only the first L slots per row are attended: skip the rest"
        out.append(f"`{a['method']}({a['name']})` masks: {text}{hint}")
    for a in args:
        values = a.get("values")
        flags = all(v in ("True", "False") or v.startswith(("'", '"')) for v in values or ())
        if not values or a.get("ints") or not flags:
            continue
        text = ", ".join(f"{v} ×{c}" for v, c in values.items())
        out.append(f"`{a['method']}({a['name']})`: {text}")
    for a in args:
        ints = a.get("ints")
        if not ints or (a["method"], a["name"]) in covered or not POSITION_NAME.search(a["name"]):
            continue
        out.append(
            f"`{a['method']}({a['name']})`: {ints['min']}..{ints['max']} "
            f"({ints['distinct']} distinct)"
        )
    for a in args:
        if a.get("contiguous", 1.0) < 1.0:
            out.append(
                f"`{a['method']}({a['name']})`: non-contiguous in "
                f"{_pct(1 - a['contiguous'])} of its tensors ({a.get('layout')}): "
                "index with strides or the kernel reads the wrong elements"
            )
        if a.get("zero_frac"):
            out.append(
                f"`{a['method']}({a['name']})`: all zeros in {_pct(a['zero_frac'])} of the calls"
            )
    params = p.get("params") or {}
    if params.get("zero_biases"):
        names = ", ".join(f"`{n}`" for n in params["zero_biases"][:6])
        out.append(f"parameters: {names} are all zeros (the bias add can be dropped)")
    elif params and not params.get("biases"):
        out.append("parameters: no biases")
    for method in p.get("methods", {}):
        floats = collections.defaultdict(list)
        for a in args:
            if a["method"] == method:
                for dtype in a.get("dtypes", {}):
                    if dtype.startswith(("float", "bfloat")):
                        floats[dtype].append(a["name"])
        if len(floats) > 1:
            mix = "; ".join(f"{d}: {', '.join(names[:4])}" for d, names in floats.items())
            out.append(f"`{method}` mixes float dtypes ({mix}): keep the reference's casts")
    return out


def render(p: dict[str, Any]) -> str:
    """``workload_profile.md``."""
    scope = f"`{p.get('class')}`"
    if p.get("qualname_regex"):
        scope += f" matching `{p['qualname_regex']}`"
    captured = f"`{p.get('qualname')}`"
    if p.get("phase"):
        captured += f", {p['phase']} calls only"
    lines = [
        f"# Workload profile: `{p.get('class')}`",
        "",
        f"Every call of the {p.get('instances', '?')} instance(s) of {scope} during one "
        f"workload run: {p['calls']} calls from {p['instances_called']} instance(s). The "
        f"capture keeps a few of them as cases (instance {captured}). `prefill` = "
        "multi-position calls, `decode` = `*step*` entrypoints and `[batch, 1, ...]` inputs.",
        "",
        "## Summary",
        "",
        *[f"{i + 1}. {f}" for i, f in enumerate(p.get("facts", []))],
        "",
        "## Calls by entrypoint and primary input",
        "",
        "| entrypoint | phase | primary input | calls | share | instances | distinct shapes |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in p.get("groups", []):
        lines.append(
            f"| `{g['method']}` | {g['phase']} | `{g['signature']}` | {g['calls']} | "
            f"{_pct(g['share'])} | {g['instances']} | {g['shapes']} |"
        )
    lines += [
        "",
        "## Arguments",
        "",
        "| entrypoint | argument | calls | None | dtypes | contiguous | statistics |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in p.get("args", []):
        stats = []
        if a.get("values"):
            stats.append("values " + ", ".join(f"{v} ×{c}" for v, c in a["values"].items()))
        if a.get("ints"):
            i = a["ints"]
            stats.append(f"range {i['min']}..{i['max']} ({i['distinct']} distinct)")
        if a.get("masks"):
            stats.append("masks " + ", ".join(f"{k} ×{c}" for k, c in a["masks"].items()))
        if "zero_frac" in a:
            stats.append(f"all zeros in {_pct(a['zero_frac'])}")
        if a.get("cache"):
            c = a["cache"]
            text = f"cache slots {'/'.join(map(str, c['slots']))}"
            if "valid_max" in c:
                text += (
                    f", valid length {c['valid_min']}..{c['valid_max']} "
                    f"(median {c['valid_median']}, from `{c['valid_from']}`)"
                )
            stats.append(text)
        if a.get("layout"):
            stats.append(f"first non-contiguous: {a['layout']}")
        dtypes = ", ".join(a.get("dtypes", {}))
        contiguous = _pct(a["contiguous"]) if "contiguous" in a else ""
        lines.append(
            f"| `{a['method']}` | `{a['name']}` | {a['calls']} | {a['none'] or ''} | {dtypes} | "
            f"{contiguous} | {'; '.join(stats)} |"
        )
    params = p.get("params") or {}
    if params:
        zero = ", ".join(f"`{n}`" for n in params.get("zero_biases", [])) or "none"
        lines += ["", f"Bias parameters: {params.get('biases', 0)}; all zeros: {zero}."]
    return "\n".join(lines) + "\n"


def write_profile(profile: dict[str, Any], directory: Path) -> Path:
    """Write ``workload_profile.md`` + ``.json`` into ``directory``; returns the .md path."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "workload_profile.json").write_text(json.dumps(profile, indent=2, default=str))
    path = directory / "workload_profile.md"
    path.write_text(render(profile))
    return path
