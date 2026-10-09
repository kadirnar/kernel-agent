"""Graph rewrites of the library scout's candidates (issue #227).

A scout candidate (:mod:`kernel_agent.libscout.template`) lets TorchDynamo trace the
reference's own entrypoint into FX graphs and runs them eagerly with some nodes pointed at
a library: nothing else of the reference changes, and the reference's Python code is not
what runs (Dynamo runs its graph), so the evaluator's fallback check judges the kernels the
graph launches (``custom_kernel_share``).

Every function here is self-contained (only ``torch``, ``operator`` and its arguments): the
template copies their source into the candidate, so the candidate stays one readable file
that runs without kernel-agent. They only match what Dynamo's graphs contain:
``call_function`` nodes of ``torch._C._nn.*`` / ``torch.*`` / ``operator.*`` and
``call_method`` nodes of tensor methods, and they read shapes and dtypes from the nodes'
``example_value`` (a fake tensor) where a match depends on them.
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
    for rs in rsqrts:
        add = rs.args[0]
        if not is_op(add, ("add",), (torch.add, operator.add)) or len(add.args) != 2:
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


#: The helpers a template may copy, by name.
HELPERS = {f.__name__: f for f in (swap_calls, fold_rms_norm, fold_linear, fold_silu_mul)}
