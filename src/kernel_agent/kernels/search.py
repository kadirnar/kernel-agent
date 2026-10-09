"""Search-based autotuning over declared parameter spaces (issue #229).

InferenceBench measured search beating every agent at an equal budget (SMAC3 11.53x, TPE
11.25x, the best agent 8.08x) because the agents "tested few distinct configurations"
(arXiv 2607.20468). A sweep of a hand-written list times at most :data:`sweep.MAX_CONFIGS
<kernel_agent.kernels.sweep.MAX_CONFIGS>` configs; a *space* declares the values every
parameter may take, and this module chooses what to time next from what was measured, until
the sweep's time bound (:mod:`kernels.sweep`: 3 x the evaluation timeout, one evaluation of
the budget). Nothing here imports torch or touches a GPU: the sweep subprocess asks for
configs, checks and times them, and tells the scores back.

* **Spaces** (:func:`parse_space`): per ``build()`` keyword argument a list of values,
  ``{"pow2": [lo, hi]}`` (powers of two), ``{"range": [lo, hi]}`` / ``{"range": [lo, hi,
  step]}``, the strings ``"pow2:16..256"`` / ``"2..5"`` / ``"0..64:16"``, or one fixed
  value. Numbers are ordered (a step moves to the next larger or smaller value); anything
  else is categorical.
* **Constraints** (:class:`Constraints`): Python expressions over the parameters and this
  GPU's facts (:meth:`Arch.names`: ``smem_per_block`` in bytes, ``sm`` (120 for sm_120),
  ``sm_count``, ``has_tma``, ``has_wgmma``, ``has_tcgen05``), e.g. ``"(BLOCK_M + BLOCK_N) *
  BLOCK_K * 2 * num_stages <= smem_per_block"``. Parsed with ``ast``: numbers, arithmetic,
  comparisons, ``and`` / ``or`` / ``not``, ``a if c else b``, tuples / lists for ``in`` and
  the functions of :data:`FUNCTIONS`; nothing else. Without a GPU the expressions that read
  its facts are not applied.
* **Pruning before compile** (:func:`arch_reason`, from :mod:`kernel_agent.gpu_arch`):
  warp specialisation (a parameter named ``warp_specialize`` or alike, truthy) only on
  Hopper and datacenter Blackwell (on sm_120 it measured slower and mostly failed to
  compile, docs/RESEARCH-TRITON.md §1.2); TMA (a ``tma`` / ``use_tma`` flag, or the
  indexing ``"tensor_descriptor"``) only where the family has it (sm_90+).
* **Pruning from feedback** (:meth:`Search.tell`): a config whose build ran out of a
  resource (Triton's ``OutOfResources``: shared memory beyond the per-block limit,
  threads, registers) prunes every config at least as large in each numeric parameter
  (the categorical ones equal): they need at least as much. A config that spilled
  registers (the compiler's stats, :func:`kernels.ncu.triton_stats`) prunes the ones with
  at least as large numeric parameters and at most as many warps: more work per thread
  spills at least as much. The config itself still counts with its measured time.
* **Strategies** (:class:`Search`, ask / tell; deterministic for a seed):

  - ``grid``: every valid config, in a seeded random order (a sweep cut short by its time
    bound has then timed a uniform sample of the space, not one corner);
  - ``pattern``: pattern search on the lattice of values (Hooke-Jeeves): an initial
    design (warm starts and seeded random configs), then the neighbours of the best
    config at its step size (one step up and down per numeric parameter, every other
    value of a categorical one), halving the steps when none of them is new, a pattern
    move along the last improvement, the runners-up's neighbours next, and random
    restarts once the neighbourhoods are measured;
  - ``tpe``: a tree-structured Parzen estimator in plain Python (Bergstra et al. 2011):
    the configs measured so far split into the best quarter and the rest, a Parzen
    density per parameter for each, and the next configs are the draws from the good
    density with the highest good / rest ratio;
  - ``auto``: ``grid`` when the space has at most :data:`AUTO_GRID` valid configs
    (what one listed sweep times), else ``pattern``;
  - ``helion`` (no space): Helion's own autotuner on the candidate's ``@helion.kernel``
    functions, over the space Helion derives from each (:mod:`kernels.helion_tune`).

* **Warm starts**: the sweep stores every measured point in the tuned-config cache
  (:meth:`kernels.tuned.TunedConfigs.add_points`: GPU, library versions, op and shape
  bucket) and a later search starts from the best points of the same GPU and versions at
  the same or a neighbouring shape bucket (:meth:`TunedConfigs.points
  <kernel_agent.kernels.tuned.TunedConfigs.points>`); they are measured again, never trusted.

The score of a config is its weighted speedup over the reference, timed in the same rounds
as the reference (so batches timed minutes apart compare), higher is better.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

STRATEGIES = ("auto", "grid", "pattern", "tpe")
#: The strategy of a Helion candidate: Helion's own autotuner (kernels/helion_tune.py)
HELION = "helion"
#: ``auto``: a space with at most this many valid configs is swept whole (one listed sweep)
AUTO_GRID = 64
#: Configs per ask: checked, then timed together against the reference
BATCH = 16
#: Values of one parameter at most (a range typo like ``"1..1000000000"`` is refused)
MAX_VALUES = 4096
#: Spaces up to this many configs are enumerated (the grid, the count of valid configs);
#: larger ones are sampled
ENUMERATE = 200_000
#: Measured configs whose neighbourhoods a pattern search polls in one ask (the best first)
CENTERS = 3
#: TPE: the share of the measured configs that is "good", configs before the model is used,
#: and random draws per wanted config
TPE_GAMMA = 0.25
TPE_STARTUP = 10
TPE_DRAWS = 24
#: Warm starts taken from the tuned-config cache
WARM_STARTS = 8
#: Names a constraint may use besides the parameters (:meth:`Arch.names`)
FACTS = ("sm", "smem_per_block", "sm_count", "has_tma", "has_wgmma", "has_tcgen05")


def _cdiv(a: Any, b: Any) -> Any:
    return -(-a // b)


#: Functions a constraint may call
FUNCTIONS: dict[str, Callable[..., Any]] = {
    "min": min,
    "max": max,
    "abs": abs,
    "int": int,
    "float": float,
    "round": round,
    "log2": math.log2,
    "cdiv": _cdiv,
}


def _literal(value: Any) -> bool:
    if value is None or isinstance(value, bool | int | str):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


# ------------------------------------------------------------------ spaces


@dataclass(frozen=True)
class Dim:
    """One parameter: its values (numbers ascending when ``ordinal``)."""

    name: str
    values: tuple[Any, ...]
    ordinal: bool


@dataclass(frozen=True)
class Space:
    """The product of its parameters' values; a config is addressed by its value indices."""

    dims: tuple[Dim, ...]

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dims]

    @property
    def size(self) -> int:
        return math.prod(len(d.values) for d in self.dims)

    def config(self, key: Sequence[int]) -> dict[str, Any]:
        return {d.name: d.values[i] for d, i in zip(self.dims, key, strict=True)}

    def key(self, config: Mapping[str, Any]) -> tuple[int, ...] | None:
        """The indices of ``config``'s values (None: another parameter set or a value outside
        the space)."""
        if set(config) != set(self.names):
            return None
        out = []
        for d in self.dims:
            value = config[d.name]
            match = [i for i, v in enumerate(d.values) if v == value and type(v) is type(value)]
            if not match:  # 64 vs 64.0 from JSON: equal numbers
                match = [i for i, v in enumerate(d.values) if _number(v) and v == value]
            if not match:
                return None
            out.append(match[0])
        return tuple(out)

    def decode(self, n: int) -> tuple[int, ...]:
        """The ``n``-th config of the product (mixed radix, the last parameter fastest)."""
        out = []
        for d in reversed(self.dims):
            n, i = divmod(n, len(d.values))
            out.append(i)
        return tuple(reversed(out))


_RANGE = re.compile(r"^\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*(?::\s*(\d+)\s*)?$")


def _pow2(lo: Any, hi: Any) -> list[int]:
    lo, hi = int(lo), int(hi)
    if lo < 1 or hi < lo:
        raise ValueError(f"pow2 needs 1 <= lo <= hi, not {lo}..{hi}")
    out, v = [], 1
    while v <= hi:
        if v >= lo:
            out.append(v)
        v *= 2
    if not out:
        raise ValueError(f"no power of two in {lo}..{hi}")
    return out


def _range(lo: Any, hi: Any, step: Any = 1) -> list[int]:
    lo, hi, step = int(lo), int(hi), int(step)
    if step < 1 or hi < lo:
        raise ValueError(f"a range needs lo <= hi and step >= 1, not {lo}..{hi}:{step}")
    if (hi - lo) // step + 1 > MAX_VALUES:
        raise ValueError(f"{lo}..{hi}:{step} has more than {MAX_VALUES} values")
    return list(range(lo, hi + 1, step))


def _values(name: str, spec: Any) -> list[Any]:
    """The values of one parameter of a space (module docstring)."""
    if isinstance(spec, list):
        values = spec
    elif isinstance(spec, dict):
        if len(spec) != 1 or next(iter(spec)) not in ("pow2", "range"):
            raise ValueError(
                f'{name}: a dict value is {{"pow2": [lo, hi]}} or {{"range": [lo, hi, step]}}'
            )
        kind, args = next(iter(spec.items()))
        if not isinstance(args, list) or not 2 <= len(args) <= (2 if kind == "pow2" else 3):
            form = "[lo, hi]" if kind == "pow2" else "[lo, hi] or [lo, hi, step]"
            raise ValueError(f"{name}: {kind} takes {form}")
        try:
            values = _pow2(*args) if kind == "pow2" else _range(*args)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: {exc}") from None
    elif isinstance(spec, str) and (spec.startswith("pow2:") or _RANGE.match(spec)):
        text = spec.removeprefix("pow2:")
        match = _RANGE.match(text)
        if match is None or (spec.startswith("pow2:") and match.group(3)):
            raise ValueError(f'{name}: {spec!r} is not "pow2:lo..hi", "lo..hi" or "lo..hi:step"')
        lo, hi, step = match.group(1), match.group(2), match.group(3) or 1
        try:
            values = _pow2(lo, hi) if spec.startswith("pow2:") else _range(lo, hi, step)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None
    else:
        values = [spec]  # one fixed value
    if not values:
        raise ValueError(f"{name}: no values")
    if len(values) > MAX_VALUES:
        raise ValueError(f"{name}: more than {MAX_VALUES} values")
    out: list[Any] = []
    for value in values:
        if not _literal(value):
            raise ValueError(f"{name}: {value!r} is not a number, string, bool or null"[:300])
        if not any(value == v and type(value) is type(v) for v in out):
            out.append(value)
    return out


def parse_space(value: Any) -> Space:
    """A :class:`Space` from a dict of parameter -> values (module docstring), possibly as
    JSON text. Raises ``ValueError``."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"space is not valid JSON: {exc}") from None
    if not isinstance(value, dict) or not value:
        raise ValueError(
            'a space is a dict of build() keyword arguments to their values, e.g. {"BLOCK": '
            '{"pow2": [64, 4096]}, "num_warps": [1, 2, 4, 8, 16], "num_stages": "1..4"}'
        )
    dims = []
    for name, spec in value.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"{name!r} is not a keyword argument name")
        if name in FACTS or name in FUNCTIONS or name.startswith("_op"):
            raise ValueError(f"{name!r} is reserved for the GPU's facts and constraint functions")
        values = _values(name, spec)
        ordinal = all(_number(v) for v in values)
        dims.append(Dim(name, tuple(sorted(values) if ordinal else values), ordinal))
    return Space(tuple(dims))


# ------------------------------------------------------------------ the GPU's facts


@dataclass(frozen=True)
class Arch:
    """What the pruning reads from the GPU at hand (unknown: nothing is pruned for it)."""

    capability: tuple[int, int] | None = None
    smem_per_block: int = 0  # bytes a block may opt in to (0: unknown)
    sm_count: int = 0

    @classmethod
    def detect(cls) -> Arch:
        """This process's GPU (:func:`kernel_agent.toolchain.gpu_info`); empty without one."""
        from kernel_agent import gpu_arch, toolchain

        gpu = toolchain.gpu_info()
        if gpu is None:
            return cls()
        capability = (int(gpu.capability[0]), int(gpu.capability[1]))
        kb = gpu.smem_per_block_kb
        if not kb:
            fam = gpu_arch.family(capability)
            kb = gpu_arch.SMEM_PER_BLOCK_KB.get(capability, fam.smem_kb if fam else 0.0)
        return cls(capability, round(kb * 1024), int(gpu.sm_count))

    @property
    def arch(self) -> str:
        from kernel_agent.gpu_arch import arch_of

        return arch_of(self.capability) or "an unknown GPU"

    def has(self, feature: str) -> bool:
        from kernel_agent.gpu_arch import has

        return has(self.capability, feature)

    def names(self) -> dict[str, Any]:
        """The facts a constraint may read (none without a GPU)."""
        if self.capability is None:
            return {}
        out: dict[str, Any] = {
            "sm": self.capability[0] * 10 + self.capability[1],
            "has_tma": self.has("tma"),
            "has_wgmma": self.has("wgmma"),
            "has_tcgen05": self.has("tcgen05"),
        }
        if self.smem_per_block:
            out["smem_per_block"] = self.smem_per_block
        if self.sm_count:
            out["sm_count"] = self.sm_count
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": list(self.capability) if self.capability else None,
            "smem_per_block": self.smem_per_block,
            "sm_count": self.sm_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Arch:
        data = data or {}
        cap = data.get("capability")
        return cls(
            (int(cap[0]), int(cap[1])) if cap else None,
            int(data.get("smem_per_block") or 0),
            int(data.get("sm_count") or 0),
        )


_WARP_SPECIALIZE = re.compile(r"warp_?speciali[sz]", re.I)
_TMA_FLAG = re.compile(r"(?:^|_)tma$", re.I)
_TMA_INDEXING = ("tensor_descriptor", "tma")


def _truthy(value: Any) -> bool:
    if isinstance(value, list | tuple):
        return any(_truthy(v) for v in value)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "off", "no", "none")
    return bool(value)


def _uses_tma(value: Any) -> bool:
    if isinstance(value, list | tuple):
        return any(_uses_tma(v) for v in value)
    return isinstance(value, str) and value.strip().lower() in _TMA_INDEXING


def arch_reason(config: Mapping[str, Any], arch: Arch) -> str | None:
    """Why ``config`` cannot run (well) on ``arch``, before anything compiles; None: it may."""
    if arch.capability is None:
        return None
    for name, value in config.items():
        if _WARP_SPECIALIZE.search(name) and _truthy(value):
            if arch.has("wgmma") or arch.has("tcgen05"):
                continue
            if arch.capability[0] == 12:
                return (
                    f"{name}={value!r}: warp specialisation on {arch.arch} measured slower and "
                    "mostly failed to compile (docs/RESEARCH-TRITON.md §1.2)"
                )
            return (
                f"{name}={value!r}: warp specialisation needs Hopper (sm_90) or datacenter "
                f"Blackwell (sm_100); this GPU is {arch.arch}"
            )
        tma = (_TMA_FLAG.search(name) and _truthy(value)) or _uses_tma(value)
        if tma and not arch.has("tma"):
            return (
                f"{name}={value!r}: TMA (tensor descriptors) needs sm_90+; this GPU is {arch.arch}"
            )
    return None


# ------------------------------------------------------------------ constraints

_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
    ast.BitAnd: lambda a, b: a & b,
    ast.BitOr: lambda a, b: a | b,
    ast.BitXor: lambda a, b: a ^ b,
}
_OP_NAMES = {op: f"_op{i}" for i, op in enumerate(_OPS)}
_ALLOWED = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.IfExp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.BinOp,
    *_OPS,
)


def _arith(op: Callable[[Any, Any], Any], pow_: bool) -> Callable[[Any, Any], Any]:
    """``op`` on numbers only (no string or list repetition), a power with a small exponent."""

    def run(a: Any, b: Any) -> Any:
        if not (isinstance(a, int | float) and isinstance(b, int | float)):
            raise TypeError(f"arithmetic on {type(a).__name__} and {type(b).__name__}")
        if pow_ and (abs(b) > 64 or abs(a) > 2**64):
            raise ValueError(f"{a} ** {b} is out of range")
        return op(a, b)

    return run


_SAFE_OPS = {name: _arith(_OPS[op], op is ast.Pow) for op, name in _OP_NAMES.items()}


class _Calls(ast.NodeTransformer):
    """Every binary operation as a call of its number-only function (:func:`_arith`)."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        func = ast.Name(id=_OP_NAMES[type(node.op)], ctx=ast.Load())
        call = ast.Call(func=func, args=[node.left, node.right], keywords=[])
        return ast.copy_location(call, node)


@dataclass
class Constraints:
    """Expressions every config must satisfy (module docstring)."""

    texts: list[str] = field(default_factory=list)
    _codes: list[Any] = field(default_factory=list, repr=False)
    _facts: list[set[str]] = field(default_factory=list, repr=False)

    @classmethod
    def parse(cls, value: Any, params: Iterable[str]) -> Constraints:
        """From an expression, several separated by ``;`` or new lines, or a list of them.
        Raises ``ValueError`` naming the expression and what it may not use."""
        if value is None or value == "":
            return cls()
        if isinstance(value, str):
            texts = [t.strip() for t in re.split(r"[;\n]", value)]
        elif isinstance(value, list) and all(isinstance(t, str) for t in value):
            texts = [t.strip() for t in value]
        else:
            raise ValueError("constraints is an expression or a list of expressions (strings)")
        params = set(params)
        out = cls()
        for text in (t for t in texts if t):
            try:
                tree = ast.parse(text, mode="eval")
            except SyntaxError as exc:
                raise ValueError(f"constraint {text!r}: {exc.msg}") from None
            used: set[str] = set()
            for node in ast.walk(tree):
                if not isinstance(node, _ALLOWED):
                    raise ValueError(
                        f"constraint {text!r}: {type(node).__name__} is not allowed (numbers, "
                        "arithmetic, comparisons, and / or / not, a if c else b, "
                        f"{', '.join(FUNCTIONS)})"
                    )
                if isinstance(node, ast.Call) and (
                    not isinstance(node.func, ast.Name)
                    or node.func.id not in FUNCTIONS
                    or node.keywords
                ):
                    raise ValueError(
                        f"constraint {text!r}: only {', '.join(FUNCTIONS)} may be called "
                        "(positional arguments)"
                    )
                if isinstance(node, ast.Constant) and not _literal(node.value):
                    raise ValueError(f"constraint {text!r}: {node.value!r} is not allowed")
                if isinstance(node, ast.Name) and node.id not in FUNCTIONS:
                    if node.id not in params and node.id not in FACTS:
                        raise ValueError(
                            f"constraint {text!r}: {node.id!r} is neither a parameter of the "
                            f"space ({', '.join(sorted(params))}) nor a GPU fact "
                            f"({', '.join(FACTS)})"
                        )
                    used.add(node.id)
            safe = ast.fix_missing_locations(_Calls().visit(tree))
            out.texts.append(text)
            out._codes.append(compile(safe, "<constraint>", "eval"))
            out._facts.append(used & set(FACTS))
        return out

    def violated(self, config: Mapping[str, Any], facts: Mapping[str, Any]) -> str | None:
        """The first expression ``config`` fails (with the facts it read), or None. An
        expression that reads a fact ``facts`` does not have (no GPU) is not applied."""
        scope = {**FUNCTIONS, **_SAFE_OPS, **facts, **config}
        for text, code, used in zip(self.texts, self._codes, self._facts, strict=True):
            if not used <= set(facts):
                continue
            try:
                ok = eval(code, {"__builtins__": {}}, scope)  # whitelisted AST (parse)
            except Exception as exc:
                return f"{text} ({type(exc).__name__}: {exc})"[:300]
            if not ok:
                where = ", ".join(f"{name} = {facts[name]}" for name in sorted(used))
                return text + (f" ({where})" if where else "")
        return None


# ------------------------------------------------------------------ feedback

_RESOURCE = re.compile(
    r"out of resource|OutOfResources|Hardware limit|too many resources requested", re.I
)
_WARPS = re.compile(r"warps|threads", re.I)


def resource_error(error: Any) -> bool:
    """Whether a config's error says it ran out of a GPU resource (shared memory, threads,
    registers) at compile or launch."""
    return bool(_RESOURCE.search(str(error or "")))


# ------------------------------------------------------------------ the search


def spec_from(
    space: Any,
    constraints: Any = None,
    strategy: Any = None,
    seed: Any = None,
) -> dict[str, Any]:
    """A validated search spec (JSON-able: it is passed to the sweep subprocess): ``space``,
    ``constraints``, ``strategy``, ``seed``. Strategy :data:`HELION` takes no space: Helion's
    autotuner searches the one it derives from each kernel (:mod:`kernels.helion_tune`).
    Raises ``ValueError``."""
    strategy = str(strategy or "auto").strip().lower()
    if strategy not in (*STRATEGIES, HELION):
        choices = ", ".join((*STRATEGIES, HELION))
        raise ValueError(f"strategy is one of {choices}, not {strategy!r}")
    try:
        seed = 0 if seed is None else int(seed)
    except (TypeError, ValueError):
        raise ValueError(f"seed is an integer, not {seed!r}") from None
    if strategy == HELION:
        if space is not None or constraints:
            raise ValueError(
                "strategy helion runs Helion's own autotuner on the candidate's @helion.kernel "
                "functions over the space Helion derives from each: no space or constraints"
            )
        return {"strategy": HELION, "seed": seed}
    if space is None:
        raise ValueError("a search needs a space: the values every parameter may take")
    parsed = parse_space(space)
    if isinstance(space, str):
        space = json.loads(space)
    texts = Constraints.parse(constraints, parsed.names).texts
    spec = {"space": space, "constraints": texts, "strategy": strategy, "seed": seed}
    found = Search.from_spec(spec)  # no GPU here: the constraints on its facts wait for it
    if found.valid == 0:
        raise ValueError(
            f"no config of the space ({found.space.size}) satisfies the constraints; the first "
            f"fails {found.examples.get('constraint')!r}"
        )
    return spec


class Search:
    """Ask / tell search over a :class:`Space` (module docstring). ``ask(n)`` returns up to
    ``n`` configs never asked before (valid, not pruned); ``tell`` reports one's score
    (weighted speedup, higher is better) or its failure. Deterministic for a seed and the
    sequence of tells."""

    def __init__(
        self,
        space: Space,
        constraints: Constraints | None = None,
        *,
        strategy: str = "auto",
        seed: int = 0,
        arch: Arch | None = None,
        warm: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy is one of {', '.join(STRATEGIES)}, not {strategy!r}")
        self.space = space
        self.constraints = constraints or Constraints()
        self.arch = arch or Arch()
        self.facts = self.arch.names()
        self.seed = seed
        self.rng = random.Random(seed)
        self.requested = strategy
        self.scores: dict[tuple[int, ...], float | None] = {}  # told: score, None if it failed
        self.told: list[tuple[int, ...]] = []
        self.failed: set[tuple[int, ...]] = set()  # told with an error
        self.asked: set[tuple[int, ...]] = set()
        self._static: dict[tuple[int, ...], str | None] = {}
        self.limits: list[tuple[tuple[int, ...], str]] = []  # out of resources
        self.spilled: list[tuple[int, ...]] = []
        self.pruned: dict[str, int] = {"constraint": 0, "arch": 0, "feedback": 0}
        self.examples: dict[str, str] = {}
        self.restarts = 0
        self.steps: dict[tuple[int, ...], tuple[int, ...]] = {}
        self.trail: list[tuple[int, ...]] = []  # the incumbents, in order
        self._valid: list[tuple[int, ...]] | None = None
        if space.size <= ENUMERATE:
            every = itertools.product(*(range(len(d.values)) for d in space.dims))
            self._valid = [k for k in every if self._static_reason(k) is None]
        self.valid = len(self._valid) if self._valid is not None else None
        if strategy == "auto":
            small = self.valid is not None and self.valid <= AUTO_GRID
            strategy = "grid" if small else "pattern"
        self.strategy = strategy
        self._grid: list[tuple[int, ...]] | None = None
        self._cursor = 0  # the grid's next position (passed configs were asked or pruned)
        if strategy == "grid" and self._valid is not None:
            self._grid = list(self._valid)
            self.rng.shuffle(self._grid)
        self.warm: list[tuple[int, ...]] = []
        for config in warm:
            key = space.key(config)
            if key is not None and key not in self.warm and self._static_reason(key) is None:
                self.warm.append(key)
        self.warm = self.warm[:WARM_STARTS]

    @classmethod
    def from_spec(
        cls,
        spec: Mapping[str, Any],
        *,
        arch: Arch | None = None,
        warm: Iterable[Mapping[str, Any]] = (),
    ) -> Search:
        space = parse_space(spec["space"])
        constraints = Constraints.parse(spec.get("constraints"), space.names)
        return cls(
            space,
            constraints,
            strategy=str(spec.get("strategy") or "auto"),
            seed=int(spec.get("seed") or 0),
            arch=arch,
            warm=warm,
        )

    # -- what may be tried

    def _static_reason(self, key: tuple[int, ...]) -> str | None:
        """Constraint or architecture (counted once per config)."""
        if key in self._static:
            return self._static[key]
        config = self.space.config(key)
        why, kind = self.constraints.violated(config, self.facts), "constraint"
        if why is None:
            why, kind = arch_reason(config, self.arch), "arch"
        if why is not None:
            self.pruned[kind] += 1
            self.examples.setdefault(kind, why)
        self._static[key] = why
        return why

    def _dominates(self, key: Sequence[int], base: Sequence[int], *, warps_down: bool) -> bool:
        """``key`` at least as large as ``base`` in every numeric parameter (with
        ``warps_down``: at most as many warps / threads), equal in the categorical ones."""
        for d, k, b in zip(self.space.dims, key, base, strict=True):
            if not d.ordinal:
                if k != b:
                    return False
            elif warps_down and _WARPS.search(d.name):
                if k > b:
                    return False
            elif k < b:
                return False
        return True

    def _feedback_reason(self, key: tuple[int, ...]) -> str | None:
        for base, error in self.limits:
            if self._dominates(key, base, warps_down=False):
                label = _label(self.space.config(base))
                return f"at least as large as {label}, which ran out of resources ({error})"
        for base in self.spilled:
            if self._dominates(key, base, warps_down=True):
                return (
                    f"at least as much work per thread as {_label(self.space.config(base))}, "
                    "which spilled registers"
                )
        return None

    def reason(self, config: Mapping[str, Any]) -> str | None:
        """Why ``config`` is not tried (None: it may be)."""
        key = self.space.key(config)
        if key is None:
            return "not in the space"
        return self._static_reason(key) or self._feedback_reason(key)

    def _fresh(self, key: tuple[int, ...]) -> bool:
        if key in self.scores or key in self.asked or self._static_reason(key) is not None:
            return False
        why = self._feedback_reason(key)
        if why is not None:
            self.pruned["feedback"] += 1
            self.examples.setdefault("feedback", why)
            self.asked.add(key)  # counted once, never asked
            return False
        return True

    # -- ask / tell

    def ask(self, n: int = BATCH) -> list[dict[str, Any]]:
        """Up to ``n`` configs to measure next (empty: the space is exhausted)."""
        if self.strategy == "grid":
            keys = self._ask_grid(n)
        elif self.strategy == "tpe":
            keys = self._ask_tpe(n)
        else:
            keys = self._ask_pattern(n)
        self.asked.update(keys)
        return [self.space.config(k) for k in keys]

    def tell(
        self,
        config: Mapping[str, Any],
        score: float | None,
        *,
        error: Any = None,
        spills: Any = None,
    ) -> None:
        """``config``'s weighted speedup (None: it failed, ``error`` says why; an untimed
        pass is told with its score None and no error), and the registers it spilled."""
        key = self.space.key(config)
        if key is None:
            return
        if key not in self.scores:
            self.told.append(key)
        ok = score is not None and math.isfinite(float(score))
        self.scores[key] = float(score) if ok else None  # type: ignore[arg-type]
        if error is not None:
            self.failed.add(key)
        if score is None and resource_error(error):
            self.limits.append((key, _short(error)))
        if spills:
            self.spilled.append(key)
        best = self.best_key()
        if best is not None and (not self.trail or self.trail[-1] != best):
            self.trail.append(best)

    def _ranked(self) -> list[tuple[int, ...]]:
        """The configs with a score, the best first (ties: the first told)."""
        scored = [(-s, i, k) for i, k in enumerate(self.told) if (s := self.scores[k]) is not None]
        return [k for _, _, k in sorted(scored)]

    def best_key(self) -> tuple[int, ...] | None:
        ranked = self._ranked()
        return ranked[0] if ranked else None

    def best(self) -> tuple[dict[str, Any], float] | None:
        """The best config told so far and its score."""
        key = self.best_key()
        if key is None:
            return None
        score = self.scores[key]
        assert score is not None
        return self.space.config(key), score

    # -- strategies

    def _random(self, n: int, out: list[tuple[int, ...]]) -> None:
        """Append up to ``n`` fresh seeded random configs to ``out``: draws (rejected when
        taken or pruned), then, for an enumerated space nearly measured, the rest of it."""
        want = len(out) + n
        if n <= 0:
            return
        for _ in range(50 * n):
            if len(out) >= want:
                return
            if self._valid is not None:
                if not self._valid:
                    return
                key = self._valid[self.rng.randrange(len(self._valid))]
            else:
                key = self.space.decode(self.rng.randrange(self.space.size))
            if key not in out and self._fresh(key):
                out.append(key)
        if self._valid is not None:
            pool = [k for k in self._valid if k not in out and self._fresh(k)]
            out += self.rng.sample(pool, min(want - len(out), len(pool)))

    def _ask_grid(self, n: int) -> list[tuple[int, ...]]:
        out = [k for k in self.warm if self._fresh(k)][:n]  # a cut grid still has them
        if self._grid is None:  # too large to enumerate: random configs
            self._random(n - len(out), out)
            return out
        while len(out) < n and self._cursor < len(self._grid):
            key = self._grid[self._cursor]
            self._cursor += 1
            if key not in out and self._fresh(key):  # a warm start is in the grid too
                out.append(key)
        return out

    def _initial(self, n: int) -> list[tuple[int, ...]]:
        out = [k for k in self.warm if self._fresh(k)][:n]
        self._random(n - len(out), out)
        return out

    def _first_step(self) -> tuple[int, ...]:
        return tuple(max(1, (len(d.values) - 1) // 4) if d.ordinal else 1 for d in self.space.dims)

    def _neighbours(self, center: tuple[int, ...], step: tuple[int, ...]) -> list[tuple[int, ...]]:
        out = []
        for i, d in enumerate(self.space.dims):
            if d.ordinal:
                moves = [center[i] + step[i], center[i] - step[i]]
                targets = [min(max(m, 0), len(d.values) - 1) for m in moves]
            else:
                targets = list(range(len(d.values)))
            for j in targets:
                if j != center[i]:
                    key = (*center[:i], j, *center[i + 1 :])
                    if key not in out:
                        out.append(key)
        return out

    def _ask_pattern(self, n: int) -> list[tuple[int, ...]]:
        ranked = self._ranked()
        if not ranked:
            return self._initial(n)
        out: list[tuple[int, ...]] = []

        def add(key: tuple[int, ...]) -> None:
            if key not in out and self._fresh(key):
                out.append(key)

        incumbent = ranked[0]
        previous = self.trail[-2] if len(self.trail) >= 2 and self.trail[-1] == incumbent else None
        if previous is not None:  # the pattern move: on along the last improvement
            add(
                tuple(
                    min(max(2 * c - p, 0), len(d.values) - 1) if d.ordinal else c
                    for d, c, p in zip(self.space.dims, incumbent, previous, strict=True)
                )
            )
        for rank, center in enumerate(ranked[:CENTERS]):
            # a new incumbent keeps the step its predecessor had reached
            inherited = self.steps.get(previous) if rank == 0 and previous is not None else None
            step = self.steps.get(center) or inherited or self._first_step()
            while True:
                polls = [k for k in self._neighbours(center, step) if self._fresh(k)]
                if polls or all(s == 1 for s in step):
                    break
                step = tuple(max(1, s // 2) for s in step)  # nothing new at this step: refine
            self.steps[center] = step
            if rank == 0 and not polls:  # a local optimum of the lattice: explore elsewhere
                self.restarts += 1
            for key in polls:
                add(key)
            if len(out) >= n:
                return out[:n]
        self._random(n - len(out), out)  # the rest of the batch: seeded random configs
        return out[:n]

    def _parzen(self, d: Dim, picks: list[int]) -> list[float]:
        """A smoothed density over ``d``'s values from the value indices ``picks``."""
        size = len(d.values)
        weights = [1.0 / size] * size  # a uniform prior worth one observation
        if d.ordinal:
            width = max(1.0, size / 8)
            for p in picks:
                for j in range(size):
                    weights[j] += math.exp(-0.5 * ((j - p) / width) ** 2)
        else:
            for p in picks:
                weights[p] += 1.0
        total = sum(weights)
        return [w / total for w in weights]

    def _ask_tpe(self, n: int) -> list[tuple[int, ...]]:
        ranked = self._ranked()
        if len(ranked) < TPE_STARTUP:
            return self._initial(n)
        n_good = max(1, math.ceil(TPE_GAMMA * len(ranked)))
        good = ranked[:n_good]
        bad = ranked[n_good:] + [k for k in self.told if self.scores[k] is None]
        dims = self.space.dims
        lows = [self._parzen(d, [k[i] for k in good]) for i, d in enumerate(dims)]
        highs = [self._parzen(d, [k[i] for k in bad]) for i, d in enumerate(dims)]
        drawn: list[tuple[int, ...]] = []
        for _ in range(TPE_DRAWS * n):
            key = tuple(
                self.rng.choices(range(len(d.values)), weights=lows[i])[0]
                for i, d in enumerate(dims)
            )
            if key not in drawn and self._fresh(key):
                drawn.append(key)

        def ratio(key: tuple[int, ...]) -> float:
            return math.prod(lows[i][j] / highs[i][j] for i, j in enumerate(key))

        out = sorted(drawn, key=lambda k: -ratio(k))[:n]  # stable: ties in draw order
        self._random(n - len(out), out)
        return out

    # -- report

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "strategy": self.strategy,
            "seed": self.seed,
            "space": self.space.size,
            "valid": self.valid,
            "measured": sum(s is not None for s in self.scores.values()),
            "failed": len(self.failed),
            "pruned": {k: v for k, v in self.pruned.items() if v},
        }
        if untimed := sum(s is None for s in self.scores.values()) - len(self.failed):
            out["untimed"] = untimed  # passed their check, nothing timed them (no GPU)
        if self.requested != self.strategy:
            out["requested"] = self.requested
        if self.examples:
            out["pruned_examples"] = dict(self.examples)
        if self.warm:
            out["warm_starts"] = len(self.warm)
        if self.restarts:
            out["restarts"] = self.restarts
        if self.arch.capability is not None:
            out["arch"] = self.arch.arch
        return out


def _label(config: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in config.items()) or "default"


def _short(error: Any) -> str:
    text = str(error or "").strip()
    match = re.search(r"out of resource[^.\n]*", text, re.I)
    return (match.group(0) if match else text.splitlines()[-1] if text else "")[:160]
