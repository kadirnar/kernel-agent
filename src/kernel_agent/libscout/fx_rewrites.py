"""Graph rewrites of the library scout's candidates (issue #227).

A scout candidate (:mod:`kernel_agent.libscout.template`) lets TorchDynamo trace the
reference's own entrypoint into FX graphs and runs them eagerly with some nodes pointed at
a library: nothing else of the reference changes, and the reference's Python code is not
what runs (Dynamo runs its graph), so the evaluator's fallback check judges the kernels the
graph launches (``custom_kernel_share``).

Every function here is self-contained (only ``torch``, ``operator``, the standard library
and its arguments): the template copies their source into the candidate, so the candidate
stays one readable file that runs without kernel-agent. The rewrites only match what
Dynamo's graphs contain: ``call_function`` nodes of ``torch._C._nn.*`` / ``torch.*`` /
``operator.*`` and ``call_method`` nodes of tensor methods, and they read shapes and dtypes
from the nodes' ``example_value`` (a fake tensor) where a match depends on them.

Besides the rewrites: :func:`inline_torchscript` (TorchScript functions the reference calls
made visible to Dynamo, :func:`torchscript_python`; the scout's trace runs them as Python
too, ``detect.trace``) and the guard-free variant's :func:`call_key` and
:func:`export_graph` (a graph traced once per call signature and called directly: no
Dynamo guard on the call).
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Any

import torch


def swap_calls(graph: Any, table: Mapping[Any, Callable[..., Any]]) -> int:
    """Point every ``call_function`` node whose target is a key of ``table`` (a function) and
    every ``call_method`` node whose method name is one (a string) at the table's function;
    returns how many nodes changed."""
    changed = 0
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in table:
            node.target = table[node.target]
            changed += 1
        elif node.op == "call_method" and isinstance(node.target, str) and node.target in table:
            node.op, node.target = "call_function", table[node.target]
            changed += 1
    return changed


def fold_rms_norm(graph: Any, rms_norm: Callable[..., Any]) -> int:
    """Replace the manual RMSNorm pattern (``x * rsqrt(mean(x ** 2, -1, keepdim) + eps)``,
    with the casts around it and an optional weight multiply after) by one node
    ``rms_norm(x, weight, eps, dtype, mid)`` (``weight`` None without one; ``dtype``: the
    pattern's output dtype; ``mid``: the dtype the normalised value has before the weight
    multiply, where the reference rounds it; None without a weight); ``F.rms_norm`` nodes
    become that node too (``mid`` None: the weight is inside). Returns how many it folded."""

    def value(node: Any) -> Any:
        return node.meta.get("example_value") if hasattr(node, "meta") else None

    def is_op(node: Any, names: tuple[str, ...], funcs: tuple[Any, ...]) -> bool:
        if not hasattr(node, "op"):
            return False
        if node.op == "call_method":
            return node.target in names
        return node.op == "call_function" and node.target in funcs

    def cast_input(node: Any) -> Any:
        """The input of a ``.to(...)`` / ``.float()`` / ``.type_as(...)`` chain."""
        while is_op(node, ("to", "float", "type_as", "contiguous"), ()):
            node = node.args[0]
        return node

    def squared(node: Any) -> Any:
        """``x`` of ``x ** 2`` / ``x * x`` / ``square(x)`` (None otherwise)."""
        if is_op(node, ("pow",), (torch.pow, operator.pow)) and tuple(node.args[1:2]) in (
            (2,),
            (2.0,),
        ):
            return node.args[0]
        mul = is_op(node, ("mul",), (torch.mul, operator.mul)) and len(node.args) == 2
        if mul and node.args[0] is node.args[1]:
            return node.args[0]
        if is_op(node, ("square",), (torch.square,)):
            return node.args[0]
        return None

    folded = 0
    rsqrts = [n for n in graph.nodes if is_op(n, ("rsqrt",), (torch.rsqrt,))]
    # ``variance += eps`` is ``operator.iadd`` in Dynamo's graphs (``add_`` as a method)
    adds = (torch.add, operator.add, operator.iadd)
    for rs in rsqrts:
        add = rs.args[0]
        if not is_op(add, ("add", "add_"), adds) or len(add.args) != 2:
            continue
        if add.kwargs.get("alpha") not in (None, 1):
            continue
        mean, eps = add.args
        if not hasattr(mean, "op"):
            mean, eps = eps, mean
        if not isinstance(eps, float) or not is_op(mean, ("mean",), (torch.mean,)):
            continue
        dim = mean.args[1] if len(mean.args) > 1 else mean.kwargs.get("dim")
        keep = mean.args[2] if len(mean.args) > 2 else mean.kwargs.get("keepdim")
        if dim not in (-1, [-1], (-1,)) or keep is not True:
            continue
        x = squared(mean.args[0])
        if x is None:
            continue
        # x times the rsqrt: the squared x itself, or the tensor it was cast from (an fp32 x
        # times rsqrt, or the input's dtype promoted by the fp32 rsqrt: the same values)
        same = {id(x), id(cast_input(x))}
        scaled = next(
            (
                u
                for u in rs.users
                if is_op(u, ("mul",), (torch.mul, operator.mul))
                and len(u.args) == 2
                and any(a is rs for a in u.args)
                and all(a is rs or id(a) in same or id(cast_input(a)) in same for a in u.args)
            ),
            None,
        )
        if scaled is None:
            continue
        out, weight = scaled, None
        cast = next(iter(out.users), None) if len(out.users) == 1 else None
        if cast is not None and is_op(cast, ("to", "type_as"), ()):
            out = cast
        normed = value(out)  # before the weight: its dtype is where the reference rounds
        mul = next(iter(out.users), None) if len(out.users) == 1 else None
        if mul is not None and is_op(mul, ("mul",), (torch.mul, operator.mul)):
            other = mul.args[1] if mul.args[0] is out else mul.args[0]
            w, h = value(other), value(x)
            if (
                isinstance(w, torch.Tensor)
                and isinstance(h, torch.Tensor)
                and w.dim() == 1
                and w.shape[0] == h.shape[-1]
            ):
                out, weight = mul, other
        result = value(out)
        if not isinstance(result, torch.Tensor) or not isinstance(normed, torch.Tensor):
            continue
        source = cast_input(x)
        mid = normed.dtype if weight is not None else None
        with graph.inserting_before(out):
            node = graph.call_function(rms_norm, (source, weight, eps, result.dtype, mid))
        node.meta.update(out.meta)
        out.replace_all_uses_with(node)
        erased: set[int] = set()
        dead: Any
        for dead in (out, mul, cast, scaled, rs, add, mean, mean.args[0], x):  # output first
            if not hasattr(dead, "op") or dead is node or id(dead) in erased:
                continue
            if dead.op != "placeholder" and not dead.users:
                graph.erase_node(dead)
                erased.add(id(dead))
        folded += 1
    norms = (torch.rms_norm, torch.nn.functional.rms_norm)
    for node in list(graph.nodes):  # F.rms_norm(x, shape, weight, eps): the same node
        if node.op != "call_function" or node.target not in norms or not node.args:
            continue
        x = node.args[0]
        shape = node.args[1] if len(node.args) > 1 else node.kwargs.get("normalized_shape")
        weight = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
        eps = node.args[3] if len(node.args) > 3 else node.kwargs.get("eps")
        result, given = value(node), value(x)
        if not isinstance(result, torch.Tensor) or not isinstance(given, torch.Tensor):
            continue
        if len(list(shape or [])) != 1:
            continue
        eps = float(torch.finfo(given.dtype).eps) if eps is None else float(eps)  # torch's
        node.target, node.args, node.kwargs = rms_norm, (x, weight, eps, result.dtype, None), {}
        folded += 1
    return folded


def fold_linear(
    graph: Any, linear: Callable[..., Any], *, residual: bool = True, gelu: bool = True
) -> int:
    """Point ``F.linear`` nodes at ``linear(x, weight, bias, residual, act)``, folding into
    one node what follows a linear whose only user it is: a residual ``+`` of a tensor of the
    output's shape and dtype (with ``residual`` True), or a tanh-approximated GELU (``act``
    "gelu"; with ``gelu`` True). Returns how many linear nodes changed.

    A folded residual rounds once where the reference rounds twice (the GEMM's output, then
    the sum): where the sum cancels a large product the difference exceeds the tolerance
    (a decoder layer's output, measured), so it is a choice the evaluator judges."""

    def value(node: Any) -> Any:
        return node.meta.get("example_value") if hasattr(node, "meta") else None

    targets = (torch._C._nn.linear, torch.nn.functional.linear)
    changed = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in targets:
            continue
        x = node.args[0]
        weight = node.args[1] if len(node.args) > 1 else node.kwargs.get("weight")
        bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
        out, added, act = node, None, None
        user = next(iter(node.users), None) if len(node.users) == 1 else None
        mine = value(node)
        if user is not None and user.op == "call_function" and isinstance(mine, torch.Tensor):
            sums = (operator.add, torch.add)
            if residual and user.target in sums and len(user.args) == 2:
                other = user.args[1] if user.args[0] is node else user.args[0]
                theirs = value(other)
                same = (
                    isinstance(theirs, torch.Tensor)
                    and tuple(theirs.shape) == tuple(mine.shape)
                    and theirs.dtype == mine.dtype
                    and user.kwargs.get("alpha") in (None, 1)
                    and other is not node
                )
                if same:
                    out, added = user, other
            elif (
                gelu
                and user.target is torch.nn.functional.gelu
                and user.kwargs.get("approximate") == "tanh"
            ):
                out, act = user, "gelu"
        with graph.inserting_before(out):
            new = graph.call_function(linear, (x, weight, bias, added, act))
        new.meta.update(out.meta)
        out.replace_all_uses_with(new)
        if out is not node:
            graph.erase_node(out)
        graph.erase_node(node)
        changed += 1
    return changed


def fold_silu_mul(graph: Any, silu_mul: Callable[..., Any]) -> int:
    """Replace ``silu(gate) * up`` (the SwiGLU of a gated MLP; the SiLU's only user is the
    product) by ``silu_mul(gate, up)``. Returns how many it folded."""
    silus = (torch.nn.functional.silu,)
    changed = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in silus:
            continue
        if node.kwargs.get("inplace") or len(node.users) != 1:
            continue
        mul = next(iter(node.users))
        if mul.op != "call_function" or mul.target not in (operator.mul, torch.mul):
            continue
        if len(mul.args) != 2 or mul.args[0] is mul.args[1]:
            continue
        up = mul.args[1] if mul.args[0] is node else mul.args[0]
        if not hasattr(up, "op"):
            continue
        with graph.inserting_before(mul):
            new = graph.call_function(silu_mul, (node.args[0], up))
        new.meta.update(mul.meta)
        mul.replace_all_uses_with(new)
        graph.erase_node(mul)
        graph.erase_node(node)
        changed += 1
    return changed


def fold_rope(graph: Any, rope: Callable[..., Any]) -> int:
    """Replace two rotate-half rotary embeddings with the same cos and sin (q and k:
    ``x * cos + cat(-x2, x1) * sin`` each, ``x1`` / ``x2`` the halves of x's last dimension,
    sliced or chunked) by one node ``rope(q, k, cos, sin, q_free, k_free)`` and a getitem
    per output; ``q_free`` / ``k_free``: nothing else reads the tensor (or the one whose
    storage it views), so the library may write its result into it. Pairs are taken in
    graph order among the applications sharing one cos and sin (a whole model's layers
    share them): 4-D tensors of one batch, sequence length, head size and dtype, each
    output of its input's shape and dtype. Returns how many pairs it folded."""

    def value(node: Any) -> Any:
        return node.meta.get("example_value") if hasattr(node, "meta") else None

    def is_op(node: Any, names: tuple[str, ...], funcs: tuple[Any, ...]) -> bool:
        if not hasattr(node, "op"):
            return False
        if node.op == "call_method":
            return node.target in names
        return node.op == "call_function" and node.target in funcs

    def binary(node: Any, names: tuple[str, ...], funcs: tuple[Any, ...]) -> bool:
        return is_op(node, names, funcs) and len(node.args) == 2 and not node.kwargs

    def half(node: Any) -> tuple[Any, int] | None:
        """``(x, 0)`` for x's first half on the last dimension, ``(x, 1)`` for its second."""
        if not is_op(node, (), (operator.getitem,)) or len(node.args) != 2:
            return None
        src, index = node.args
        if is_op(src, ("chunk",), (torch.chunk,)) and isinstance(index, int):
            chunks = src.args[1] if len(src.args) > 1 else src.kwargs.get("chunks")
            dim = src.args[2] if len(src.args) > 2 else src.kwargs.get("dim", 0)
            x = src.args[0]
            t = value(x)
            ok = isinstance(t, torch.Tensor) and chunks == 2 and dim in (-1, t.dim() - 1)
            return (x, index) if ok and index in (0, 1) else None
        t = value(src)
        if not isinstance(t, torch.Tensor) or not isinstance(index, tuple) or not index:
            return None
        *lead, last = index
        full = slice(None, None, None)
        if not isinstance(last, slice) or last.step not in (None, 1):
            return None
        if Ellipsis not in lead and len(index) != t.dim():
            return None
        if any(i is not Ellipsis and i != full for i in lead) or lead.count(Ellipsis) > 1:
            return None
        d = t.shape[-1]
        if d % 2:
            return None
        if last.start in (None, 0) and last.stop == d // 2:
            return src, 0
        if last.start == d // 2 and last.stop in (None, d):
            return src, 1
        return None

    def other(node: Any, mine: Any) -> Any:
        a, b = node.args
        return b if a is mine else a if b is mine else None

    views: tuple[str, ...] = ("view", "reshape", "transpose", "permute", "unflatten")
    views += ("flatten", "unsqueeze", "squeeze", "contiguous", "expand", "narrow", "swapaxes")
    views += ("movedim", "t")
    view_funcs = tuple(getattr(torch, n) for n in views if hasattr(torch, n))

    def free(x: Any, nodes: list[Any]) -> bool:
        """Nothing but the pattern's ``nodes`` reads x (nor x's halves the pattern takes),
        nor the tensors it views back to a fresh one."""
        own = {id(n) for n in nodes}
        inner = [x, *nodes[1:]]  # nodes[0]: the pattern's output, read by what follows
        if any(id(u) not in own for n in inner for u in n.users):
            return False
        node = x
        while is_op(node, views, view_funcs) and hasattr(node.args[0], "op"):
            node = node.args[0]
            if len(node.users) != 1:
                return False
        return node.op not in ("placeholder", "get_attr")

    mul_ops, add_ops = (operator.mul, torch.mul), (operator.add, torch.add)
    apps = []
    for cat in list(graph.nodes):
        if not is_op(cat, ("cat",), (torch.cat, torch.concat)) or len(cat.users) != 1:
            continue
        parts = cat.args[0] if cat.args else cat.kwargs.get("tensors")
        dim = cat.args[1] if len(cat.args) > 1 else cat.kwargs.get("dim", 0)
        if not isinstance(parts, list | tuple) or len(parts) != 2:
            continue
        neg, first = parts
        if not is_op(neg, ("neg",), (operator.neg, torch.neg)) or len(neg.users) != 1:
            continue
        a, b = half(first), half(neg.args[0])
        if a is None or b is None or a[0] is not b[0] or (a[1], b[1]) != (0, 1):
            continue
        x = a[0]
        t = value(x)
        if not isinstance(t, torch.Tensor) or t.dim() != 4 or dim not in (-1, 3):
            continue
        mul_sin = next(iter(cat.users))
        if not binary(mul_sin, ("mul",), mul_ops) or len(mul_sin.users) != 1:
            continue
        sin, add = other(mul_sin, cat), next(iter(mul_sin.users))
        if not binary(add, ("add",), add_ops):
            continue
        mul_cos = other(add, mul_sin)
        if not binary(mul_cos, ("mul",), mul_ops) or len(mul_cos.users) != 1:
            continue
        cos = other(mul_cos, x)
        out = value(add)
        if cos is None or sin is None or not isinstance(out, torch.Tensor):
            continue
        if out.dtype != t.dtype or tuple(out.shape) != tuple(t.shape):
            continue
        nodes = [add, mul_cos, mul_sin, cat, neg, neg.args[0], first]
        nodes += [n.args[0] for n in (first, neg.args[0]) if n.args[0] is not x]  # a chunk
        nodes = list(dict.fromkeys(nodes))  # output first: erased in this order
        apps.append({"x": x, "cos": cos, "sin": sin, "out": add, "nodes": nodes, "t": t})
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for app in apps:
        groups.setdefault((id(app["cos"]), id(app["sin"])), []).append(app)
    folded = 0
    for group in groups.values():
        for qa, ka in zip(group[0::2], group[1::2], strict=False):
            q, k = qa["t"], ka["t"]
            same = (q.shape[0], q.shape[2], q.shape[3], q.dtype)
            if same != (k.shape[0], k.shape[2], k.shape[3], k.dtype):
                continue
            pos = {id(n): i for i, n in enumerate(graph.nodes)}
            first_out, last_out = sorted((qa["out"], ka["out"]), key=lambda n: pos[id(n)])
            inputs = [n for n in (qa["x"], ka["x"], qa["cos"], qa["sin"]) if hasattr(n, "op")]
            if all(pos[id(n)] < pos[id(first_out)] for n in inputs):
                anchor = first_out
            elif all(pos[id(u)] > pos[id(last_out)] for u in first_out.users):
                anchor = last_out
            else:
                continue
            flags = (free(qa["x"], qa["nodes"]), free(ka["x"], ka["nodes"]))
            with graph.inserting_before(anchor):
                pair = graph.call_function(rope, (qa["x"], ka["x"], qa["cos"], qa["sin"], *flags))
                q_new = graph.call_function(operator.getitem, (pair, 0))
                k_new = graph.call_function(operator.getitem, (pair, 1))
            pair.meta["example_value"] = (value(qa["out"]), value(ka["out"]))
            q_new.meta.update(qa["out"].meta)
            k_new.meta.update(ka["out"].meta)
            qa["out"].replace_all_uses_with(q_new)
            ka["out"].replace_all_uses_with(k_new)
            for dead in [*qa["nodes"], *ka["nodes"]]:  # outputs first
                if dead.op not in ("placeholder", "get_attr") and not dead.users:
                    graph.erase_node(dead)
            folded += 1
    return folded


def torchscript_python(fn: Any) -> Callable[..., Any] | None:
    """The eager Python equivalent of a TorchScript function (``torch.jit.ScriptFunction``,
    scripted or traced): the Python function it was made from (``_torchdynamo_inline``,
    which TorchDynamo inlines), else its TorchScript code (``fn.code``) run as Python with
    TorchScript's spellings mapped (``torch.to(x, 6)``: dtype codes, ``ops.prim.dtype(x)``,
    ``torch.<tensor method>(x, ...)``); None when neither works (a TorchScript archive's
    function without code)."""
    import re
    import types
    import typing

    original = getattr(fn, "_torchdynamo_inline", None)
    if callable(original):
        return original
    code, name = getattr(fn, "code", None), getattr(fn, "name", None)
    if not isinstance(code, str) or not isinstance(name, str):
        return None
    # c10::ScalarType codes TorchScript writes for dtypes
    codes = dict(enumerate([torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    codes.update({5: torch.float16, 6: torch.float32, 7: torch.float64, 11: torch.bool})
    codes[15] = torch.bfloat16

    def dtype(v: Any) -> Any:
        return codes.get(v, v) if isinstance(v, int) and not isinstance(v, bool) else v

    def to(t: Any, *args: Any) -> Any:
        args_ = [dtype(a) for a in args]
        while args_ and (args_[-1] is None or args_[-1] is False):  # TorchScript's defaults
            args_.pop()
        return t.to(*args_)

    def method(attr: str) -> Callable[..., Any]:
        def call(t: Any, *args: Any, **kwargs: Any) -> Any:
            return getattr(t, attr)(*args, **kwargs)

        return call

    special = {
        "to": to,
        "__is__": lambda a, b: a is b,
        "__isnot__": lambda a, b: a is not b,
        "len": len,
        "list": list,
        "dim": lambda t: t.dim(),
        "size": lambda t, d=None: t.size() if d is None else t.size(d),
    }
    used = set(re.findall(r"\btorch\.(\w+)", code))
    shim = types.SimpleNamespace(
        **{
            n: special.get(n) or getattr(torch, n, None) or method(n)
            for n in used
            if n in special or hasattr(torch, n) or hasattr(torch.Tensor, n)
        }
    )
    prim = types.SimpleNamespace(dtype=lambda t: t.dtype, device=lambda t: t.device)
    namespace: dict[str, Any] = {
        "torch": shim,
        "ops": types.SimpleNamespace(prim=prim),
        "Tensor": torch.Tensor,
        "annotate": lambda kind, v: v,
        **{n: getattr(typing, n) for n in ("Any", "Dict", "List", "Optional", "Tuple")},
    }
    try:
        exec(compile(code, f"<torchscript {name}>", "exec"), namespace)
    except Exception:
        return None
    found = namespace.get(name)
    return found if callable(found) else None


def torchscript_functions(module: Any) -> list[tuple[dict[str, Any], str, Any]]:
    """The TorchScript functions (``torch.jit.ScriptFunction``: scripted or traced) that
    ``module``'s code calls: ``(namespace, name, function)`` for each global name the methods
    of its submodules' classes (defined outside torch) read, and the Python functions they
    call read in turn, that holds one, and for each one held by a submodule's attribute. A
    scripted *module* is not one (it has no Python original)."""
    import types

    script = getattr(torch.jit, "ScriptFunction", None)
    if script is None or not hasattr(module, "modules"):
        return []
    found: dict[tuple[int, str], tuple[dict[str, Any], str, Any]] = {}
    todo: list[Any] = []  # functions whose code is read
    for sub in module.modules():
        for name, value in vars(sub).items():
            if isinstance(value, script):
                found[(id(vars(sub)), name)] = (vars(sub), name, value)
        for cls in type(sub).__mro__:
            if cls.__module__.split(".")[0] not in ("torch", "builtins"):
                todo += [getattr(a, "__func__", a) for a in vars(cls).values()]
    seen: set[int] = set()
    while todo:
        fn = todo.pop()
        if not isinstance(fn, types.FunctionType) or id(fn) in seen or len(seen) > 4096:
            continue
        seen.add(id(fn))
        if fn.__module__ and fn.__module__.split(".")[0] == "torch":
            continue
        space, codes = fn.__globals__, [fn.__code__]
        while codes:  # the function's own code and the code of what it defines inside
            code = codes.pop()
            codes += [c for c in code.co_consts if isinstance(c, types.CodeType)]
            for name in code.co_names:
                value = space.get(name)
                if isinstance(value, script):
                    found[(id(space), name)] = (space, name, value)
                elif isinstance(value, types.FunctionType):
                    todo.append(value)
    return list(found.values())


def inline_torchscript(module: Any) -> list[str]:
    """Let TorchDynamo trace into the TorchScript functions ``module``'s code calls
    (:func:`torchscript_functions`): Dynamo inlines one that has ``_torchdynamo_inline``
    (its Python original: ``torch.jit.script`` / ``torch.jit.trace`` of a Python function
    set it) and breaks the graph at one without (a function of a TorchScript archive),
    which gets its :func:`torchscript_python`. Eager callers (the reference) still run
    TorchScript. Returns the functions' names."""
    names = set()
    for _space, name, fn in torchscript_functions(module):
        names.add(name)
        if not callable(getattr(fn, "_torchdynamo_inline", None)):
            python = torchscript_python(fn)
            if python is not None:
                fn._torchdynamo_inline = python
    return sorted(names)


def call_key(name: str, args: Any, kwargs: Any) -> tuple[Any, ...] | None:
    """What a graph traced once for a call is specialised on (:func:`export_graph`, a scout
    candidate's guard-free variant): the entrypoint, grad and inference mode, the default
    dtype and, for every argument (nested in lists, tuples and dicts), a tensor's shape,
    strides, dtype, device and ``requires_grad``, a plain value itself. None when an
    argument is anything else (a cache object): only Dynamo's guards follow what a call
    changes in it. Not whether a tensor was made in inference mode: the graph computes the
    same on both (Dynamo's guards tell them apart for its compiled code's sake, and
    recompile on an evaluator's copy of an input made in inference mode, even inside a
    CUDA graph capture)."""
    parts: list[Any] = [name, torch.is_grad_enabled(), torch.is_inference_mode_enabled()]
    parts.append(torch.get_default_dtype())
    plain = (bool, int, float, str, torch.dtype, torch.device)
    todo = [args, kwargs]
    while todo:
        v = todo.pop()
        if isinstance(v, torch.Tensor):
            parts.append((v.shape, v.stride(), v.dtype, v.device, v.requires_grad))
        elif v is None or isinstance(v, plain):
            parts.append((type(v), v))
        elif isinstance(v, list | tuple):
            parts.append((type(v), len(v)))
            todo.extend(v)
        elif isinstance(v, dict):
            parts.append((dict, tuple(v)))
            todo.extend(v.values())
        else:
            return None
    return tuple(parts)


def export_graph(
    fn: Callable[..., Any], args: Any, kwargs: Any, rewrite: Callable[[Any], int]
) -> tuple[Any, int]:
    """``fn`` traced once by TorchDynamo for these arguments (``torch._dynamo.export``: one
    graph, static shapes, the module's parameters and buffers as attributes of the graph
    module, the same tensors; traced on fake tensors, so nothing runs) with ``rewrite(graph)``
    applied: ``(graph module, called like fn; nodes rewritten)``. Raises when Dynamo cannot
    trace the call as one graph (a graph break: data-dependent control flow, an opaque
    call)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch._dynamo.export(fn, aten_graph=False, assume_static_by_default=True)(
            *args, **kwargs
        )
    gm = traced.graph_module
    changed = rewrite(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return gm, changed


#: The helpers a template may copy, by name.
HELPERS = {
    f.__name__: f
    for f in (
        swap_calls,
        fold_rms_norm,
        fold_linear,
        fold_silu_mul,
        fold_rope,
        torchscript_python,
        torchscript_functions,
        inline_torchscript,
        call_key,
        export_graph,
    )
}
