"""The library scout's candidate files (issue #227).

:func:`render` writes one self-contained candidate per adapter and target:

* a docstring that says what it is (generated, not an agent's), which families the reference
  calls, which configs the scout sweeps and whether the template ran on a GPU yet;
* ``KA_LIBRARY = "<package>@<version>"`` (the ledger's backend ``library:<package>@<version>``,
  :func:`kernel_agent.ledger.detect_backend`; the export's ``requirements.txt``),
  ``KA_LICENCE``, and ``ARCHS`` / ``ARCHS_WHY`` like the bundled examples;
* the graph rewrite helpers it needs, copied from :mod:`kernel_agent.libscout.fx_rewrites`;
* the adapter's op code (:meth:`Adapter.code`): ``ops(reference, **config)`` and
  ``rewrite(graph, reference, **config)``;
* ``LibraryScout``: the reference's captured entrypoints compiled by TorchDynamo with a
  backend that runs ``rewrite`` on each FX graph and then runs the graph eagerly (no
  Inductor), and ``build(reference, <config>=<default>, ..., GUARDS=1)``.

The reference's Python code does not run (Dynamo runs its graph), so the evaluator's fallback
check judges the kernels: a candidate that launches only the reference's kernels is a
``fallback``, one with library kernels has ``custom_kernel_share`` > 0.

**Guards** (follow-up 5 of #227). Dynamo's compiled callable checks its guards on every
call: host time a launch-bound target pays at module level, on top of the Python around
the library call (the A10's SDPA core: flash 0.37x, cuDNN 0.35x eager though cuDNN's
kernel is 1.8x faster; guard-free 0.41x, 0.40x). ``GUARDS=0`` is the guard-free variant:
per call signature (:func:`fx_rewrites.call_key`: the shapes, strides, dtypes and devices
of the tensors, the plain values, grad and inference mode) the
entrypoint is traced once by ``torch._dynamo.export`` (:func:`fx_rewrites.export_graph`:
one graph, the reference's own parameters and buffers, ``rewrite`` applied) and that graph
module is called directly. What the guards would follow beyond the key is not followed:
the scout sweeps the variant only where nothing else changes between calls
(:mod:`kernel_agent.libscout.probe`: not for a module whose capture tracks state); a call
whose arguments hold other objects (a cache) or that Dynamo cannot trace as one graph runs
the guarded graphs, and ``guarded`` says why.
"""

from __future__ import annotations

import inspect
import keyword
import re
from collections.abc import Iterable, Mapping
from typing import Any

from kernel_agent.libscout import fx_rewrites
from kernel_agent.libscout.registry import Adapter

#: Dynamo's recompile limits the candidate raises (one compile per module instance and
#: input shape: an integration builds one candidate per layer).
RECOMPILE_LIMIT = 256
ACCUMULATED_LIMIT = 4096

_SCAFFOLD = '''

class LibraryScout(nn.Module):
    """The reference's entrypoints compiled by TorchDynamo with ``rewrite`` applied to each
    FX graph, which then runs eagerly (no Inductor): only the rewritten nodes change.

    ``guards`` False (``GUARDS=0``, the guard-free variant): each call signature's graph is
    traced once (``export_graph``) and called directly, without Dynamo's guard check on
    every call; a call it cannot take that way (an argument that is not a tensor or a plain
    value, a graph break) runs the guarded graphs, and ``guarded`` says why."""

    def __init__(self, reference, config, guards=True):
        super().__init__()
        self.reference = reference  # its weights and state, shared
        self.rewritten = 0  # graph nodes ``rewrite`` changed (0: the library runs nowhere)
        self.guards = guards
        self.graphs = {{}}  # call signature -> its rewritten graph module (GUARDS=0)
        self.guarded = {{}}  # call signature -> why it runs the guarded graphs (GUARDS=0)
        self.inlined = {inline}  # TorchScript functions Dynamo traces into
        self._config = config

        def compile_graph(gm, example_inputs):
            self.rewritten += rewrite(gm.graph, reference, **config)
            gm.graph.lint()
            gm.recompile()
            return gm.forward

        self._compiled = {{
            name: torch.compile(getattr(reference, name), backend=compile_graph)
            for name in METHODS
        }}

    def _rewrite(self, graph):
        return rewrite(graph, self.reference, **self._config)

    def _call(self, name, args, kwargs):
        if self.guards:
            return self._compiled[name](*args, **kwargs)
        key = call_key(name, args, kwargs)
        graph = self.graphs.get(key)
        if graph is not None:
            return graph.forward(*args, **kwargs)  # no nn.Module call machinery either
        if key is None:
            self.guarded[key] = NOT_PLAIN
        elif key not in self.guarded:
            graph = self._trace(name, key, args, kwargs)
        if graph is None:
            return self._compiled[name](*args, **kwargs)
        return graph.forward(*args, **kwargs)

    def _trace(self, name, key, args, kwargs):
        """The rewritten graph of a new call signature, traced once (None: it runs the
        guarded graphs, and ``guarded[key]`` says why)."""
        if len(self.graphs) >= MAX_GRAPHS:
            self.guarded[key] = f"more than {{MAX_GRAPHS}} call signatures"
            return None
        method = getattr(self.reference, name)
        try:
            graph, changed = export_graph(method, args, kwargs, self._rewrite)
        except Exception as exc:
            why = f"{{type(exc).__name__}}: {{exc}}".strip().splitlines()[0][:200]
            self.guarded[key] = f"Dynamo does not trace the call as one graph ({{why}})"
            return None
        self.rewritten += changed
        self.graphs[key] = graph
        return graph
{methods}

def build(reference{params}, GUARDS={guards}):
    """The scout's candidate: {title}. ``GUARDS=0``: its guard-free variant."""
    limits = torch._dynamo.config
    limits.recompile_limit = max(limits.recompile_limit, {limit})
    limits.accumulated_recompile_limit = max(limits.accumulated_recompile_limit, {total})
    prepare(reference)
    return LibraryScout(reference, {config}, guards=bool(GUARDS))
'''

_METHOD = """
    def {name}(self, *args, **kwargs):
        return self._call("{name}", args, kwargs)
"""

#: The scaffold's own keyword argument: 1 the guarded graphs (TorchDynamo's compiled
#: callable), 0 the guard-free variant (one graph per call signature, called directly)
GUARDS = "GUARDS"
#: Graphs a guard-free candidate traces (one per call signature); calls beyond run guarded
MAX_GRAPHS = 64
#: Why a guard-free candidate's call runs the guarded graphs: an argument it cannot key on
NOT_PLAIN = (
    "an argument is not a tensor or a plain value (a cache object): only Dynamo's guards "
    "follow what a call changes in it"
)


def _identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def config_keys(configs: Iterable[Mapping[str, Any]]) -> list[str]:
    """The keyword arguments the configs set, in first-seen order."""
    keys: dict[str, None] = {}
    for config in configs:
        for key in config:
            keys.setdefault(str(key), None)
    return list(keys)


def render(
    adapter: Adapter,
    *,
    target: str,
    version: str | None,
    configs: list[dict[str, Any]],
    families: str = "",
    methods: Iterable[str] = ("forward",),
    torchscript: bool = False,
) -> str:
    """The candidate file of ``adapter`` for ``target`` (see the module docstring); the
    first config gives ``build``'s defaults (``GUARDS``: the scaffold's, 1 when no config
    sets it); ``torchscript``: the reference calls TorchScript functions (the candidate
    makes sure Dynamo traces into them, ``fx_rewrites.inline_torchscript``)."""
    methods = [m for m in dict.fromkeys(["forward", *methods]) if _identifier(m)]
    keys = [k for k in config_keys(configs) if _identifier(k) and k != GUARDS]
    first = configs[0] if configs else {}
    params = "".join(f", {k}={first.get(k)!r}" for k in keys)
    config = "{" + ", ".join(f"{k!r}: {k}" for k in keys) + "}"
    label = f"{adapter.package}@{version or 'unknown'}"
    swept = "; ".join(", ".join(f"{k}={v!r}" for k, v in c.items()) or "default" for c in configs)
    verified = (
        "Verified: this template ran on a GPU in kernel-agent's tests."
        if adapter.verified
        else f"Not verified on a GPU yet: written from {adapter.package}'s documentation "
        "(the library was not installed where kernel-agent was developed)."
    )
    doc = (
        f'"""Library scout candidate for target `{target}`: {adapter.title}.\n\n'
        f"Generated by kernel-agent's library scout (``kernel_agent/libscout``, adapter "
        f"``{adapter.name}``), not by an agent. TorchDynamo traces the reference's own "
        "entrypoints into FX graphs; ``rewrite`` points the library's ops into them; the "
        "graphs run eagerly (no Inductor), so every other op launches the reference's own "
        "kernels and the evaluator's fallback check passes only when the library's kernels "
        "differ from the reference's (``custom_kernel_share`` > 0).\n\n"
        f"The reference calls (one captured case): {families or 'see spec.json'}.\n"
        f"Configs swept by the scout (``build`` keyword arguments): {swept}.\n"
        f"{verified}\n\n"
        "``GUARDS=1``: Dynamo's compiled callable, which checks its guards on every call (host "
        "time: tens of microseconds on a busy CPU, which a launch-bound target pays; inside a "
        "CUDA graph it is gone). ``GUARDS=0``: each call signature's graph traced once and "
        "called directly, no guard on the call (a call it cannot take runs guarded). The "
        "library call itself can go into a hand-written module instead.\n"
        '"""\n'
    )
    head = [
        doc,
        "from __future__ import annotations",
        "",
        "import operator",
        "",
        "import torch",
        "import torch._dynamo",
        "from torch import nn",
        "",
        f"KA_LIBRARY = {label!r}  # the ledger's backend: library:<package>@<version>",
        f"KA_LICENCE = {adapter.licence!r}",
        f"KA_ADAPTER = {adapter.name!r}",
    ]
    if adapter.archs:
        head.append(f"ARCHS = {adapter.archs!r}")
        head.append(f"ARCHS_WHY = {adapter.archs_why!r}")
    head.append(f"METHODS = {tuple(methods)!r}  # the captured entrypoints")
    head.append(f"MAX_GRAPHS = {MAX_GRAPHS}  # GUARDS=0: call signatures traced, then guarded")
    head.append(f"NOT_PLAIN = {NOT_PLAIN!r}")
    names = [*_helpers(adapter), "call_key", "export_graph"]
    if torchscript:
        names += ["torchscript_python", "torchscript_functions", "inline_torchscript"]
    helpers = [inspect.getsource(fx_rewrites.HELPERS[h]) for h in dict.fromkeys(names)]
    code = adapter.code().strip("\n")
    if not re.search(r"^def prepare\(", code, re.M):
        code += '\n\n\ndef prepare(reference):\n    """Nothing to build before timing."""\n'
    guards = first.get(GUARDS, 1)
    scaffold = _SCAFFOLD.format(
        methods="".join(_METHOD.format(name=m) for m in methods),
        params=params,
        guards=int(guards) if isinstance(guards, bool | int) else 1,
        title=adapter.title,
        limit=RECOMPILE_LIMIT,
        total=ACCUMULATED_LIMIT,
        config=config,
        inline="inline_torchscript(reference)" if torchscript else "[]",
    )
    body = "\n\n".join(h.strip("\n") for h in helpers)
    text = "\n".join(head) + "\n\n\n" + body + "\n\n" + code.rstrip() + "\n\n" + scaffold
    return re.sub(r"\n{4,}", "\n\n\n", text).rstrip() + "\n"


def _helpers(adapter: Adapter) -> list[str]:
    """The helper functions the adapter's code calls (``swap_calls`` always: the
    simplest rewrite, and the one most adapters use)."""
    names = ["swap_calls", *adapter.helpers]
    return [n for n in dict.fromkeys(names) if n in fx_rewrites.HELPERS]


def guard_free(configs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each config twice: Dynamo's guarded graphs (``GUARDS=1``) and the guard-free variant
    (``GUARDS=0``), what a sweep tries when the guard-free variant applies."""
    return [{**c, GUARDS: g} for c in configs for g in (1, 0)]


def library_of(source: str) -> str | None:
    """``<package>@<version>`` of a candidate's ``KA_LIBRARY`` line (None: not a scout
    candidate, or one written by hand that declares none)."""
    match = re.search(r"""^KA_LIBRARY\s*=\s*['"]([^'"\n]+)['"]""", source, re.M)
    return match.group(1) if match else None


def licence_of(source: str) -> str | None:
    match = re.search(r"""^KA_LICENCE\s*=\s*['"]([^'"\n]+)['"]""", source, re.M)
    return match.group(1) if match else None
