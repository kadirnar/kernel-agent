"""Op families from what a reference actually calls (issue #227).

The library scout decides which library kernels apply to a target from the torch functions
its reference runs on one captured case, never from module or class names: the same RMSNorm
is ``F.rms_norm`` in one model and ``x.pow(2).mean(-1, keepdim=True)`` + ``rsqrt`` in another.

:func:`trace` runs a call under a ``TorchFunctionMode`` and records every torch function and
tensor method it reaches (outermost calls only: a mode is off inside its own handler), with
the shapes and dtypes of the tensors and which earlier call produced each input (the data
flow, by tensor identity; every output is kept alive for the trace so identities stay
unique; an in-place ``add_`` / ``mul_`` counts as its operation, its output being its
input). TorchScript runs in C++, out of the mode's sight: the TorchScript functions the
module's code calls run as their Python during the trace (:func:`eager_torchscript`).
:func:`families` classifies the record:

* ``sdpa``: ``F.scaled_dot_product_attention`` (q / k / v shapes, GQA, mask, causal);
* ``rms_norm``: ``F.rms_norm``, or the manual pattern ``rsqrt(mean(x²) + eps)`` times ``x``
  (``pow(2)`` / ``x * x`` / ``square``; casts allowed), with its weight multiply;
* ``layer_norm``, ``softmax``;
* ``linear`` (``F.linear``: rows M, N, K, bias) and ``matmul`` (``matmul`` / ``bmm`` / ``@``
  outside attention);
* ``sampling``: ``multinomial``, or ``topk`` / ``sort`` + ``cumsum`` (top-k / top-p);
* ``rotary``: ``cat(-x2, x1)`` (rotate-half) multiplied and added back (RoPE), ``paired``
  when two tensors (q and k) are rotated by one cos and sin;
* ``gated_mlp``: ``act(gate) * up`` where gate and up are linears of one input (or halves
  of one merged linear), followed by a down projection.

Everything here runs on CPU tensors too (the tests' toy modules); the scout's probe
(:mod:`kernel_agent.libscout.probe`) runs it on the GPU, on the capture's dominant case.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch.overrides import TorchFunctionMode

#: Canonical names of the functions the classifiers look at, by ``__name__`` (tensor dunder
#: methods map to their operation).
_ALIASES = {
    "__mul__": "mul",
    "__rmul__": "mul",
    "__imul__": "mul",
    "__add__": "add",
    "__radd__": "add",
    "__iadd__": "add",
    "__sub__": "sub",
    "__rsub__": "sub",
    "__neg__": "neg",
    "negative": "neg",
    "__matmul__": "matmul",
    "__rmatmul__": "matmul",
    "__pow__": "pow",
    "__getitem__": "getitem",
    # in-place arithmetic: ``variance += eps`` reaches the mode as ``Tensor.add_`` (its
    # output is its input: the data flow follows it), ``x *= w`` as ``mul_``
    "add_": "add",
    "sub_": "sub",
    "mul_": "mul",
    "neg_": "neg",
    "pow_": "pow",
    "rsqrt_": "rsqrt",
    "__isub__": "sub",
    "_softmax": "softmax",
    "special_softmax": "softmax",
    "concat": "cat",
    "concatenate": "cat",
    "type_as": "to",
    "float": "to",
    "half": "to",
    "bfloat16": "to",
}
#: Calls that change a tensor's dtype or view, not its values: the pattern matchers look
#: through them.
_PASS = frozenset({"to", "contiguous", "view", "reshape", "clone"})
_ACTIVATIONS = frozenset({"silu", "gelu", "relu", "sigmoid", "tanh", "mish", "gelu_tanh"})
_MATMULS = frozenset({"matmul", "bmm", "mm", "einsum", "baddbmm", "addmm"})
#: Scalar keyword arguments worth keeping in a record (what templates and reasons need).
_KWARGS = ("is_causal", "enable_gqa", "dropout_p", "scale", "dim", "keepdim", "approximate", "eps")
FAMILIES = (
    "sdpa",
    "rms_norm",
    "layer_norm",
    "softmax",
    "linear",
    "matmul",
    "sampling",
    "rotary",
    "gated_mlp",
)


@dataclass
class Call:
    """One recorded torch function / tensor method call."""

    index: int
    name: str  # canonical (``_ALIASES``)
    qualname: str  # where it lives (``torch.nn.functional.linear``, ``Tensor.pow``)
    shapes: list[list[int]]  # of the tensor arguments, in order
    dtypes: list[str]
    inputs: list[int]  # the call that produced each tensor argument (-1: from outside)
    params: list[str | None]  # the parameter name of each tensor argument (None: not one)
    scalars: list[Any]  # positional plain values (exponents, eps, dims)
    kwargs: dict[str, Any]
    out_shapes: list[list[int]]
    out_dtypes: list[str]
    #: kwargs that are tensors or None (``attn_mask``): whether given
    tensor_kwargs: dict[str, bool] = field(default_factory=dict)


def _name(func: Callable[..., Any]) -> tuple[str, str]:
    raw = getattr(func, "__name__", None) or str(func)
    if raw == "__get__":  # a property of a tensor (shape, dtype, T)
        owner = getattr(func, "__self__", None)
        raw = f"get_{getattr(owner, '__name__', 'attr')}"
    module = getattr(func, "__module__", None) or ""
    qual = getattr(func, "__qualname__", raw)
    where = f"{module}.{qual}" if module and not qual.startswith(module) else qual
    return _ALIASES.get(raw, raw), where


def _tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _tensors(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _tensors(v)


def _plain(value: Any) -> bool:
    return value is None or isinstance(value, bool | int | float | str)


class _Recorder(TorchFunctionMode):
    """Records every call it sees (see :func:`trace`)."""

    def __init__(self, params: Mapping[int, str], invocations: list[Any] | None = None) -> None:
        super().__init__()
        self.calls: list[Call] = []
        self.invocations = invocations  # (func, args, kwargs) per call, when asked for
        self.producer: dict[int, int] = {}  # id(tensor) -> index of the call that made it
        self.keep: list[Any] = []  # every output stays alive: ids stay unique
        self.params = params

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name, where = _name(func)
        if name.startswith("get_") or name in ("__setitem__", "size", "dim", "numel", "stride"):
            return out  # metadata, not math
        ins = list(_tensors(args)) + list(_tensors(kwargs))
        outs = list(_tensors(out))
        if not ins and not outs:
            return out
        index = len(self.calls)
        self.calls.append(
            Call(
                index=index,
                name=name,
                qualname=where,
                shapes=[list(t.shape) for t in ins],
                dtypes=[str(t.dtype).removeprefix("torch.") for t in ins],
                inputs=[self.producer.get(id(t), -1) for t in ins],
                params=[self.params.get(id(t)) for t in ins],
                scalars=[a for a in args if _plain(a)][:4],
                kwargs={k: kwargs[k] for k in _KWARGS if k in kwargs and _plain(kwargs[k])},
                out_shapes=[list(t.shape) for t in outs],
                out_dtypes=[str(t.dtype).removeprefix("torch.") for t in outs],
                tensor_kwargs={
                    k: v is not None
                    for k, v in kwargs.items()
                    if v is None or isinstance(v, torch.Tensor)
                },
            )
        )
        for t in outs:
            self.producer[id(t)] = index
        self.keep.append(out)
        if self.invocations is not None:
            self.invocations.append((func, args, kwargs))
        return out


def trace(
    fn: Callable[..., Any],
    args: Any = (),
    kwargs: Any = None,
    *,
    module: Any = None,
    invocations: list[Any] | None = None,
    inlined: list[str] | None = None,
) -> list[Call]:
    """The torch calls of ``fn(*args, **kwargs)`` (see the module docstring); ``module``: the
    ``nn.Module`` whose parameters are named in :attr:`Call.params`, and whose TorchScript
    functions run as Python during the trace (:func:`eager_torchscript`: TorchScript runs
    in C++, where the mode sees nothing); ``invocations``: a list that receives ``(func,
    args, kwargs)`` of every recorded call (the scout's op bars run them again);
    ``inlined``: a list that receives the names of the TorchScript functions run as
    Python."""
    params = {id(p): n for n, p in module.named_parameters()} if module is not None else {}
    recorder = _Recorder(params, invocations)
    with torch.inference_mode(), eager_torchscript(module) as names, recorder:
        fn(*args, **(kwargs or {}))
    if inlined is not None:
        inlined.extend(names)
    return recorder.calls


@contextlib.contextmanager
def eager_torchscript(module: Any) -> Iterator[list[str]]:
    """While it is active, every TorchScript function ``module``'s code calls (a global its
    submodules' methods read, or an attribute: ``fx_rewrites.torchscript_functions``) is its
    Python equivalent (``fx_rewrites.torchscript_python``: the function it was made
    from, or its TorchScript code as Python, which falls back to TorchScript on an error),
    so a trace sees the ops inside it (an RMSNorm written in a ``@torch.jit.script``
    function). Yields the names of those it replaced; everything is put back after."""
    from kernel_agent.libscout.fx_rewrites import torchscript_functions, torchscript_python

    found = torchscript_functions(module) if module is not None else []
    patched: list[tuple[dict[str, Any], str, Any]] = []
    names: list[str] = []
    try:
        for space, name, fn in found:
            python = torchscript_python(fn)
            if python is None:
                continue
            if python is not getattr(fn, "_torchdynamo_inline", None):  # translated code
                python = _or_script(python, fn)
            patched.append((space, name, fn))
            space[name] = python
            names.append(name)
        yield sorted(set(names))
    finally:
        for space, name, fn in reversed(patched):
            space[name] = fn


def _or_script(python: Callable[..., Any], script: Any) -> Callable[..., Any]:
    """``python``, or the TorchScript function where its translation fails."""

    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return python(*args, **kwargs)
        except Exception:
            return script(*args, **kwargs)

    return call


# ------------------------------------------------------------------ classification


@dataclass
class Family:
    """What the scout found of one family: the indices of its calls and a summary each
    (``sites``), and the dtypes involved."""

    name: str
    calls: list[int] = field(default_factory=list)
    sites: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _through(calls: list[Call], index: int) -> int:
    """The call that produced ``index``'s value, looking through casts and views."""
    seen = 0
    while 0 <= index < len(calls) and calls[index].name in _PASS and seen < 8:
        index = calls[index].inputs[0] if calls[index].inputs else -1
        seen += 1
    return index


def _users(calls: list[Call]) -> dict[int, list[int]]:
    users: dict[int, list[int]] = {}
    for c in calls:
        for i in c.inputs:
            if i >= 0:
                users.setdefault(i, []).append(c.index)
    return users


def _sdpa(c: Call) -> dict[str, Any]:
    q, k = ([*c.shapes, [], []])[:2]
    site: dict[str, Any] = {"q": q, "k": k, "dtype": (c.dtypes or ["?"])[0]}
    if len(q) == 4 and len(k) == 4:
        site.update(batch=q[0], heads=q[1], kv_heads=k[1], seq_q=q[2], seq_kv=k[2], head_dim=q[3])
    site["mask"] = bool(c.tensor_kwargs.get("attn_mask")) or len(c.shapes) > 3
    # positional plain arguments after q, k, v: attn_mask (None), dropout_p, is_causal, scale
    site["causal"] = bool(c.kwargs.get("is_causal")) or any(s is True for s in c.scalars)
    site["gqa"] = bool(c.kwargs.get("enable_gqa")) or (len(q) == 4 and len(k) == 4 and q[1] != k[1])
    return site


def _linear(c: Call) -> dict[str, Any]:
    x, w = ([*c.shapes, [], []])[:2]
    k = x[-1] if x else 0
    rows = 1
    for d in x[:-1]:
        rows *= d
    return {
        "m": rows,
        "n": w[0] if w else 0,
        "k": k,
        "bias": len(c.shapes) > 2,
        "dtype": (c.dtypes or ["?"])[0],
        "weight": ([*c.params, None, None])[1],
    }


def _square_of(calls: list[Call], index: int) -> int | None:
    """``x`` when call ``index`` is ``x ** 2`` / ``x * x`` / ``square(x)`` (its producer;
    -1 for an outside tensor), else None."""
    c = calls[index]
    if c.name == "pow" and any(s == 2 for s in c.scalars if not isinstance(s, bool)):
        return c.inputs[0] if c.inputs else None
    if c.name == "square" and c.inputs:
        return c.inputs[0]
    if c.name == "mul" and len(c.inputs) == 2 and c.inputs[0] == c.inputs[1]:
        return c.inputs[0]
    return None


@dataclass
class RmsMatch:
    """One manual RMSNorm in a trace: ``square`` (the call that squares x), ``mean``,
    ``scaled`` (x times the rsqrt), ``final`` (the call whose output is the norm's: the weight
    multiply, or the cast back / ``scaled`` without a weight), ``weight`` (the parameter's
    name), ``eps``."""

    square: int
    mean: int
    scaled: int
    final: int
    weight: str | None
    eps: float | None


def _rms_matches(calls: list[Call], users: dict[int, list[int]]) -> list[RmsMatch]:
    """The manual RMSNorm pattern: ``rsqrt(mean(x², -1) + eps)`` times ``x``, with its
    weight multiply."""
    found = []
    for c in calls:
        if c.name != "rsqrt" or not c.inputs:
            continue
        add = _through(calls, c.inputs[0])
        if add < 0 or calls[add].name != "add":
            continue
        eps = next((s for s in calls[add].scalars if isinstance(s, float)), None)
        mean = next((i for i in calls[add].inputs if i >= 0 and calls[i].name == "mean"), None)
        if mean is None or not calls[mean].inputs:
            continue
        sq = _through(calls, calls[mean].inputs[0])
        x = _square_of(calls, sq) if sq >= 0 else None
        if x is None:
            continue
        scaled = next(
            (u for u in users.get(c.index, []) if calls[u].name == "mul"),
            None,
        )
        if scaled is None:
            continue
        weight, final = None, scaled
        for u in users.get(scaled, []):
            for v in [u, *users.get(u, [])]:
                if calls[v].name == "mul" and any(calls[v].params):
                    weight, final = next(p for p in calls[v].params if p), v
                    break
            if weight:
                break
        if final == scaled:  # no weight: the cast back, if any, is the output
            final = next((u for u in users.get(scaled, []) if calls[u].name == "to"), scaled)
        found.append(RmsMatch(sq, mean, scaled, final, weight, eps))
    return found


def _manual_rms(calls: list[Call], users: dict[int, list[int]]) -> list[dict[str, Any]]:
    """Sites of the manual RMSNorm pattern (:func:`_rms_matches`)."""
    sites = []
    for m in _rms_matches(calls, users):
        shape = calls[m.mean].shapes[0] if calls[m.mean].shapes else []
        sites.append(
            {
                "form": "manual",
                "hidden": shape[-1] if shape else None,
                "eps": m.eps,
                "weight": m.weight,
                "dtype": (calls[m.final].out_dtypes or ["?"])[0],
                "rows": _rows(shape),
            }
        )
    return sites


@dataclass
class RmsSpan:
    """The calls of one written-out RMSNorm (:func:`rms_spans`), what the scout's op bars
    replay against a library's fused kernel: ``calls`` (indices, in order), ``entry`` (the
    first call: its first tensor argument is x, before any cast), ``weight`` (the
    parameter's name or None), ``eps``, ``dtype`` (the norm's output dtype) and ``mid`` (the
    dtype the normalised value has before the weight multiply, where the reference rounds:
    ``fx_rewrites.fold_rms_norm``'s; None without a weight)."""

    calls: list[int]
    entry: int
    weight: str | None
    eps: float | None
    dtype: str
    mid: str | None


def rms_spans(calls: list[Call]) -> list[RmsSpan]:
    """Every manual RMSNorm of a :func:`trace` as the calls that compute it: the ancestors
    of its output back to x (its casts included; x itself, a residual add before it, is not).
    A pattern with an in-place call is left out (replaying it again would change its input)."""
    users = _users(calls)
    spans = []
    for m in _rms_matches(calls, users):
        entry = m.square
        source = calls[m.square].inputs[0] if calls[m.square].inputs else -1
        while source >= 0 and calls[source].name == "to":  # x.to(float32).pow(2): from x
            entry = source
            source = calls[source].inputs[0] if calls[source].inputs else -1
        span: set[int] = set()
        todo = [m.final]
        while todo:
            i = todo.pop()
            if i <= source or i in span:
                continue
            span.add(i)
            todo += [j for j in calls[i].inputs if j >= 0]
        if any(_in_place(calls[i].qualname) for i in span):
            continue
        final = calls[m.final]
        mid = None
        if m.weight is not None:  # the dtype of what the weight multiplies
            mid = next((d for d, p in zip(final.dtypes, final.params, strict=False) if not p), None)
        spans.append(
            RmsSpan(
                sorted(span),
                entry,
                m.weight,
                m.eps,
                (final.out_dtypes or ["?"])[0],
                mid,
            )
        )
    return spans


def _in_place(qualname: str) -> bool:
    """``Tensor.mul_`` / ``Tensor.__iadd__``: a call that writes into its input."""
    name = qualname.rpartition(".")[2]
    return name.startswith("__i") or (name.endswith("_") and not name.endswith("__"))


def _rows(shape: list[int]) -> int:
    rows = 1
    for d in shape[:-1]:
        rows *= d
    return rows


def _operand(
    calls: list[Call], call: int, other: int, shape: list[int] | None = None
) -> tuple[int, list[int]]:
    """``(producer, shape)`` of the operand of a binary ``call`` besides ``other``'s output
    (of ``shape``, when given: two outside tensors differ by it); ``(-2, [])``: none."""
    c = calls[call]
    pairs = list(zip(c.inputs, c.shapes, strict=False))
    for k, (j, s) in enumerate(pairs):
        if j == other and (shape is None or s == shape):
            rest = pairs[:k] + pairs[k + 1 :]
            return rest[0] if len(rest) == 1 else (-2, [])
    return -2, []


def _rotary(calls: list[Call], users: dict[int, list[int]]) -> list[dict[str, Any]]:
    """``cat((-x2, x1), -1)`` multiplied by sin and added to ``x * cos`` (rotate-half RoPE):
    one site per application with x's shape and dtype; ``paired`` when another application
    multiplies by the same cos and sin (q and k: what a fused RoPE kernel takes together;
    cos / sin compared by the call that produced them and their shapes)."""
    apps = []
    for c in calls:
        if c.name != "cat" or len(c.inputs) != 2:
            continue
        neg = next((i for i in c.inputs if i >= 0 and calls[i].name == "neg"), None)
        if neg is None:
            continue
        half = calls[neg].inputs[0] if calls[neg].inputs else -1
        x = calls[half].inputs[0] if half >= 0 and calls[half].inputs else -1
        for m in (u for u in users.get(c.index, []) if calls[u].name == "mul"):
            add = next((a for a in users.get(m, []) if calls[a].name == "add"), None)
            if add is None:
                continue
            shape = c.out_shapes[0] if c.out_shapes else []
            sin = _operand(calls, m, c.index)
            scaled, _ = _operand(calls, add, m)
            cos = _operand(calls, scaled, x, shape) if scaled >= 0 else (-2, [])
            key = (cos[0], str(cos[1]), sin[0], str(sin[1]))
            apps.append((key, shape, (c.out_dtypes or ["?"])[0]))
            break
    shared: dict[tuple[Any, ...], int] = {}
    for key, _, _ in apps:
        shared[key] = shared.get(key, 0) + 1
    return [
        {
            "shape": shape,
            "head_dim": shape[-1] if shape else None,
            "dtype": dtype,
            "paired": shared[key] > 1,
        }
        for key, shape, dtype in apps
    ]


def _gated(calls: list[Call], users: dict[int, list[int]]) -> list[dict[str, Any]]:
    """``act(gate) * up`` with gate and up linears of one input (or chunks of one linear),
    whose product feeds a linear (the down projection)."""
    sites = []
    for c in calls:
        if c.name != "mul" or len(c.inputs) != 2:
            continue
        a, b = c.inputs
        for act, other in ((a, b), (b, a)):
            if act < 0 or calls[act].name not in _ACTIVATIONS or not calls[act].inputs:
                continue
            gate = _through(calls, calls[act].inputs[0])
            up = _through(calls, other)
            if gate < 0 or up < 0:
                continue
            linear_pair = (
                calls[gate].name == "linear"
                and calls[up].name == "linear"
                and calls[gate].inputs[:1] == calls[up].inputs[:1]
            )
            merged = (
                calls[gate].inputs[:1] == calls[up].inputs[:1]
                and calls[gate].name in ("getitem", "chunk", "split", "unbind")
                and calls[up].name in ("getitem", "chunk", "split", "unbind")
            )
            downs = [u for u in users.get(c.index, []) if calls[u].name == "linear"]
            if (linear_pair or merged) and downs:
                down = _linear(calls[downs[0]])
                sites.append(
                    {
                        "act": calls[act].name
                        + (
                            f"({calls[act].kwargs['approximate']})"
                            if calls[act].kwargs.get("approximate") not in (None, "none")
                            else ""
                        ),
                        "merged": merged,
                        "hidden": down["n"],
                        "intermediate": down["k"],
                        "m": down["m"],
                        "dtype": down["dtype"],
                    }
                )
                break
    return sites


def families(calls: list[Call]) -> dict[str, Family]:
    """The op families in a :func:`trace` record (only those found)."""
    users = _users(calls)
    found: dict[str, Family] = {}

    def add(name: str, index: int, site: dict[str, Any]) -> None:
        fam = found.setdefault(name, Family(name))
        fam.calls.append(index)
        fam.sites.append(site)

    in_attention: set[int] = set()
    for c in calls:
        if c.name == "scaled_dot_product_attention":
            add("sdpa", c.index, _sdpa(c))
        elif c.name == "linear":
            add("linear", c.index, _linear(c))
        elif c.name == "rms_norm":
            shape = c.shapes[0] if c.shapes else []
            eps = c.kwargs.get("eps", next((s for s in c.scalars if isinstance(s, float)), None))
            add(
                "rms_norm",
                c.index,
                {
                    "form": "functional",
                    "hidden": shape[-1] if shape else None,
                    "eps": eps,
                    "weight": next((p for p in c.params if p), None),
                    "dtype": (c.dtypes or ["?"])[0],
                    "rows": _rows(shape),
                },
            )
        elif c.name in ("layer_norm", "group_norm"):
            shape = c.shapes[0] if c.shapes else []
            add("layer_norm", c.index, {"hidden": shape[-1] if shape else None, "kind": c.name})
        elif c.name in ("softmax", "log_softmax"):
            shape = c.shapes[0] if c.shapes else []
            add("softmax", c.index, {"shape": shape, "dtype": (c.dtypes or ["?"])[0]})
            # a softmax over matmul scores is hand-written attention, not a GEMM family
            for i in c.inputs:
                j = _through(calls, i)
                while j >= 0 and calls[j].name in ("mul", "add", "sub", "masked_fill", "div"):
                    j = _through(calls, calls[j].inputs[0]) if calls[j].inputs else -1
                if j >= 0 and calls[j].name in _MATMULS:
                    in_attention.add(j)
                    for u in users.get(c.index, []):
                        if calls[u].name in _MATMULS:
                            in_attention.add(u)
        elif c.name in ("multinomial", "topk", "sort", "argsort"):
            add("sampling", c.index, {"op": c.name, "shape": c.shapes[0] if c.shapes else []})
    for c in calls:
        if c.name in _MATMULS and c.index not in in_attention:
            add("matmul", c.index, {"shapes": c.shapes, "dtype": (c.dtypes or ["?"])[0]})
    sampling = found.get("sampling")
    if sampling is not None:  # topk alone (a router, a beam) is not sampling
        ops = {s["op"] for s in sampling.sites}
        cumsum = any(c.name == "cumsum" for c in calls)
        if "multinomial" not in ops and not (ops & {"sort", "argsort"} and cumsum):
            del found["sampling"]
    for site in _manual_rms(calls, users):
        add("rms_norm", -1, site)
    for site in _rotary(calls, users):
        add("rotary", -1, site)
    for site in _gated(calls, users):
        add("gated_mlp", -1, site)
    return {name: found[name] for name in FAMILIES if name in found}


def summary(found: Mapping[str, Family]) -> dict[str, Any]:
    """JSON-ready form of :func:`families` (sites capped at 16 per family)."""
    return {name: {"count": len(fam.sites), "sites": fam.sites[:16]} for name, fam in found.items()}


def describe(found: Mapping[str, Any]) -> str:
    """``sdpa x1, linear x4, rms_norm x2 (manual)`` of :func:`summary` / :func:`families`."""
    parts = []
    for name, fam in found.items():
        sites = fam.sites if isinstance(fam, Family) else fam.get("sites", [])
        count = len(fam.sites) if isinstance(fam, Family) else fam.get("count", len(sites))
        forms = sorted({str(s["form"]) for s in sites if s.get("form")})
        parts.append(f"{name} x{count}" + (f" ({', '.join(forms)})" if forms else ""))
    return ", ".join(parts) or "none"
