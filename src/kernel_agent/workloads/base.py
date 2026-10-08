"""Workload contract: how to load, run and compare one model end to end.

A workload is the ground truth for an optimisation run.  It must be
deterministic (fixed seeds, greedy decoding) so that the output of the
optimised model can be compared against the baseline output.

Autoregressive models with continuous, sampled outputs (diffusion/flow-matching
TTS heads fed back into an LM) are *chaotic*: any numerically-correct kernel
change makes the free-running trajectory diverge.  Such workloads implement
teacher forcing: the candidate replays the baseline trajectory and only its
per-step predictions are compared, so errors cannot compound.

What a run optimises is the workload's *metric* (``-o metric=``,
:mod:`kernel_agent.objective`): the end-to-end latency by default, the time to the
first audio chunk of a streaming run for ``metric=ttfa``, the wall time per second of
generated audio for ``metric=throughput``. :func:`measure` returns its value as
``median_ms`` (lower is better for every metric).

A metric whose value is known before the run ends (``ttfa``: at the first chunk) is timed
inside :meth:`Workload.metric_window`: the timed requests stop there, and one request per
measurement streams to the end for the output the quality checks judge and the full-run
details (:func:`measure`). A request up to the first chunk costs a fraction of a streamed
run (VoxCPM2: 98 ms of 5.8 s eager), so every time-to-first-audio check is that much
cheaper with the same number of requests.

Generation loops get generic serving hooks (:mod:`.serving`): :meth:`Workload.async_flags`
(stop flags read on the host without draining the GPU), :meth:`Workload.serving` (the
``-o serving=static|continuous`` opt-in for a request queue with continuous batching and a
pipelined post stage) and :meth:`Workload.mark_ready` (a request's latency without a
device-wide synchronize).
"""

from __future__ import annotations

import contextlib
import itertools
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from kernel_agent import objective
from kernel_agent.hub import Modality

if TYPE_CHECKING:
    from kernel_agent.workloads.serving import AsyncFlags, ServingOptions

DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
#: :func:`compare_stop`: a baseline stop logit margin this close to zero is a near-tie.
STOP_NEAR_TIE = 0.5
#: Version of the built-in workloads' inputs a new run records (``WorkloadSpec.inputs_version``):
#: 2 = long non-repeating LLM prompts (#170). A ``run.json`` from before the field existed
#: loads as 1 and keeps the inputs its baseline was recorded with.
INPUTS_VERSION = 2
#: Counters a generation loop reports per run (:meth:`Workload.report_stats`), summed over
#: the run: ``steps`` (forward passes of the loop: decode steps and verifications),
#: ``verifies`` (verifications of a draft), ``drafted`` (draft tokens proposed),
#: ``accepted`` (draft tokens accepted) and ``tokens`` (tokens emitted). Others are kept too.
DECODE_COUNTERS = ("steps", "verifies", "drafted", "accepted", "tokens")


@dataclass
class WorkloadSpec:
    repo_id: str
    modality: str
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    trust_remote_code: bool = False
    harness: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    #: Model family with a dedicated built-in workload (``hub.detect_family``).
    family: str | None = None
    #: Which inputs the built-in workloads build (:data:`INPUTS_VERSION`); recorded in
    #: ``run.json``, so a run keeps the inputs of its baseline when the defaults change.
    inputs_version: int = INPUTS_VERSION

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadSpec:
        """A recorded spec; one from before ``inputs_version`` existed is version 1."""
        return cls(**{"inputs_version": 1, **data})


@dataclass
class Comparison:
    passed: bool
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    reason: str = ""


class Workload(ABC):
    """Base class for built-in and agent-written workloads."""

    modality: ClassVar[Modality] = Modality.UNKNOWN
    #: Default knobs; overridden by ``spec.options``.
    defaults: ClassVar[dict[str, Any]] = {}
    #: Extra module entrypoints besides ``forward`` that the run calls directly
    #: (class name -> method names, e.g. ``{"Attention": ["step_decode"]}``) and
    #: the name pattern in :mod:`kernel_agent.profiling.methods` misses.  The
    #: ``entrypoints=Cls.method,...`` option adds more.
    entrypoints: ClassVar[dict[str, list[str]]] = {}
    #: The free-running output diverges under any numerically-correct change, so
    #: :meth:`compare` on it is informational only (quality = teacher forcing).
    chaotic: ClassVar[bool] = False
    #: :meth:`run_teacher_forced` and :meth:`compare_teacher_forced` are implemented.
    supports_teacher_forcing: ClassVar[bool] = False
    #: Where teacher forcing hooks in (shown to the agents in the profile summary).
    teacher_forcing_note: ClassVar[str] = ""
    #: Per-output RMS tolerance of the free-running sanity check (default ±25 %),
    #: keyed like the output dict, e.g. ``{"audio": 0.6}``.
    sanity_rms_tolerance: ClassVar[dict[str, float]] = {}
    #: Metrics :func:`measure` can time for this workload (``-o metric=``,
    #: :mod:`kernel_agent.objective`). A streaming workload that calls :meth:`mark_chunk`
    #: for every output chunk can add ``"ttfa"``.
    metrics: ClassVar[tuple[str, ...]] = (objective.LATENCY,)
    #: The workload runs a request queue under ``-o serving=static|continuous``
    #: (:mod:`.serving`); the others reject the option (:meth:`check_serving`).
    supports_serving: ClassVar[bool] = False
    #: ``--quality near-lossless`` with the perceptual gate (:mod:`.perceptual`): option
    #: overrides of the end-to-end checks, a looser *sanity floor* (teacher-forcing
    #: thresholds that numerics-changing variants such as FP8 weights pass and broken
    #: kernels still fail, ``stop_tolerance``). Options the user set win.
    near_lossless_options: ClassVar[dict[str, Any]] = {}
    #: ``--quality relaxed`` (#175): overrides on top of :attr:`near_lossless_options` and the
    #: gate's ``perceptual.RELAXED_GATE``: the sanity floor loosened in proportion to the
    #: relaxed error budgets (about twice near-lossless's), still failing broken kernels.
    #: Empty: the near-lossless floor (with the relaxed gate thresholds).
    relaxed_options: ClassVar[dict[str, Any]] = {}
    #: Inside :meth:`metric_window` with a windowed metric (``ttfa``): a streaming
    #: :meth:`run` returns after its first chunk (check it after every :meth:`mark_chunk`).
    in_window: bool = False

    def __init__(self, spec: WorkloadSpec) -> None:
        self.spec = spec
        self.options = {**self.defaults, **spec.options}
        #: ``(time.perf_counter(), audio ms)`` of every chunk since the last timed run.
        self.chunk_marks: list[tuple[float, float | None]] = []
        #: Counters of the current run (:meth:`report_stats`), reset by every timed run.
        self.run_stats: dict[str, float] = {}

    @property
    def device(self) -> torch.device:
        return torch.device(self.spec.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.spec.torch_dtype

    @abstractmethod
    def load(self) -> None:
        """Download (if needed) and load the model onto the device."""

    @abstractmethod
    def roots(self) -> dict[str, nn.Module]:
        """Top-level ``nn.Module``s that are profiled and patched (name -> module)."""

    @abstractmethod
    def make_inputs(self) -> Any:
        """Deterministic inputs for :meth:`run`."""

    @abstractmethod
    def run(self, inputs: Any) -> Any:
        """One end-to-end inference.  Return CPU tensors / plain data for :meth:`compare`."""

    @abstractmethod
    def compare(self, reference: Any, candidate: Any) -> Comparison:
        """Decide whether ``candidate`` output is acceptable versus the baseline."""

    def run_teacher_forced(self, inputs: Any, reference: Any) -> Any:
        """Replay the trajectory recorded in ``reference`` (the baseline output of
        :meth:`run`): at every step feed the *reference* state back into the loop
        and record the model's own prediction for that step.

        Sampling noise must be drawn exactly as in :meth:`run` (same seed, same
        RNG calls in the same order) so that step *i* sees the same noise."""
        raise NotImplementedError(f"{type(self).__name__} does not support teacher forcing")

    def compare_teacher_forced(self, reference: Any, candidate: Any) -> Comparison:
        """Compare the per-step predictions of :meth:`run_teacher_forced` with the
        steps recorded in ``reference``."""
        raise NotImplementedError(f"{type(self).__name__} does not support teacher forcing")

    def reference_optimizations(self) -> str | None:
        """Optional hook: apply, in place and after :meth:`load`, what a competent
        user would do to speed this model up without custom kernels (the model's
        own ``torch.compile`` path, a static KV cache, a compiled denoiser, ...).
        Returns a one-line description, or ``None`` when there is nothing to apply.

        ``analyze`` measures it in a fresh process as the *compiled baseline*
        (``baseline.json`` ``compiled_ms``) and the reports show speedups vs eager
        and vs compiled; the integration also measures it on top of the accepted
        kernels.  The default does nothing (no compiled baseline unless
        ``--compile-baseline`` asks for a generic ``torch.compile`` of the roots)."""
        return None

    def holdout_options(self, variant: int = 1) -> dict[str, Any] | None:
        """Option overrides for held-out input number ``variant`` (``None``: none).

        Variant 1 is the held-out e2e input: ``analyze`` stores its baseline output
        and ``e2e`` judges every candidate on it as well, untimed
        (:mod:`kernel_agent.workloads.holdout`). It differs in content from the main
        input (another prompt, text, audio or seed) and keeps its shapes where it
        can. Variants ``>= 2`` are the memoisation probe's fresh inputs: the shapes of
        variant 1, content that differs from variant 1 (another seed, say).

        Overrides apply to ``self.options`` around :meth:`make_inputs`, :meth:`run`
        and teacher forcing (:meth:`with_options`), so seeds read in ``run`` count.
        The default changes ``seed`` when the workload has one."""
        if "seed" not in self.options:
            return None
        return {"seed": int(self.options["seed"]) + variant}

    def variants(self) -> list[dict[str, Any]]:
        """Option overrides of extra workload settings (other prompt lengths, batch
        sizes, texts) whose calls ``capture`` records as *correctness-only* cases:
        checked by the evaluator, not timed, not weighted. Keep them short (few
        decode steps); only calls with shapes the main run lacks are kept."""
        return []

    def diverse_inputs(self) -> dict[str, dict[str, Any]]:
        """The diverse input set (:mod:`kernel_agent.workloads.diverse`): option overrides
        by label, plain JSON-able values, content of different styles, languages and
        lengths (LLM: requests of 12 kinds at the end of the prompt; TTS: a few texts).
        Keep the main input's shapes where the content allows (the same ``prompt_len``,
        patches and seed): the set measures how a candidate's speedup depends on the
        *data*, not on the shapes, and new shapes would recompile or recapture.

        ``analyze`` times the baseline on every input (a warm-up and two timed runs) and
        records it in ``baseline.json`` ``diverse``; ``e2e`` times every candidate that
        passes its checks the same way and reports the per-input speedups (median, min,
        max). A candidate whose speedup varies across the set beyond the noise, or whose
        decode steps per token (:meth:`report_stats`) change with the input, is labelled
        *data-dependent* (speculative decoding, early exit): reported, never rejected.
        ``{}`` (the default): no diverse set."""
        return {}

    def report_stats(self, **counters: float) -> None:
        """Add counters of the current run, e.g. one verification of a speculative decoding
        loop: ``workload.report_stats(steps=1, verifies=1, drafted=k, accepted=a,
        tokens=a + 1)`` (:data:`DECODE_COUNTERS`; a plain decode step:
        ``report_stats(steps=1, tokens=1)``). Transforms call it from the loop they replace;
        every timed run starts from zero. The medians over the timed runs land in
        ``metric_detail.decode_stats`` with the acceptance rate, tokens per verification and
        tokens per step (:func:`decode_stats`), and the reports show them. Steps per token
        that change with the input label a candidate data-dependent
        (:mod:`kernel_agent.workloads.diverse`)."""
        stats = self.__dict__.setdefault("run_stats", {})
        for key, value in counters.items():
            stats[key] = stats.get(key, 0) + value

    def natural_length_run(self, reference: Any = None) -> dict[str, Any] | None:
        """Optional hook for autoregressive models that decide their own output length
        (a stop head, an end-of-sequence token): one untimed run in which that decision
        is live. ``None`` (the default): the workload has no stop condition.

        A fixed-length workload (VoxCPM forces ``min_len = max_len``) never lets the stop
        condition decide, so a transform that skipped or delayed it would pass every other
        check (:mod:`kernel_agent.workloads.stopping`). ``analyze`` calls this free running
        (``reference=None``) and stores the result in ``.truth/``; ``e2e`` calls it with
        ``reference`` set to that result and replays it teacher forced where the workload
        supports it, so every stop decision sees the baseline history.

        Returns a dict with ``steps`` (generated steps), ``min_steps`` / ``max_steps``
        (the limits of the run) and, optionally, ``stop_margins`` (baseline: stop minus
        continue logit of every step), ``output_length`` (e.g. audio samples) and whatever
        the teacher-forced replay needs (CPU tensors). Keep it short (tens of steps)."""
        return None

    def compare_natural_length(self, reference: Any, candidate: Any) -> Comparison:
        """Results of :meth:`natural_length_run`: the candidate must stop at the same step
        (:func:`compare_stop`; options ``stop_tolerance`` and ``stop_near_tie``)."""
        return compare_stop(
            reference,
            candidate,
            tolerance=int(self.options.get("stop_tolerance", 0)),
            near_tie=float(self.options.get("stop_near_tie", STOP_NEAR_TIE)),
        )

    @property
    def metric(self) -> str:
        """What :func:`measure` times (``-o metric=``, default ``latency``)."""
        return str(self.options.get("metric") or objective.DEFAULT).lower()

    def check_metric(self) -> None:
        """Raise ``ValueError`` when this workload cannot time ``self.metric``."""
        objective.check(self.metric, type(self).metrics, type(self).__name__)

    @property
    def windowed(self) -> bool:
        """The metric's value is known before the run ends (``ttfa``: at the first chunk),
        so :meth:`metric_window` may shorten a run."""
        return self.metric == objective.TTFA

    @contextlib.contextmanager
    def metric_window(self) -> Iterator[None]:
        """Context in which :meth:`run` may stop once the metric's value is known: with
        ``metric=ttfa`` :attr:`in_window` is set, and a streaming run returns after its
        first chunk. ``analyze`` profiles inside it, so the profile shows where the
        *metric's* time goes, and the timed requests of every measurement but one run
        inside it (:func:`measure`, the A/B rounds, the diverse set), so a time to first
        audio costs the request up to its first chunk, not a whole streamed run. Other
        metrics: the whole run."""
        if not self.windowed:
            yield
            return
        saved, self.in_window = self.in_window, True
        try:
            yield
        finally:
            self.in_window = saved

    def mark_chunk(self, audio_ms: float | None = None) -> None:
        """Streaming workloads: an output chunk is available now (call it when the chunk
        reaches the caller, e.g. on the host). GPU-synchronised. ``audio_ms``: the
        duration of the chunk's audio, for the real-time factor. ``metric=throughput``:
        one request's complete output is available now (its latency)."""
        synchronize()
        self.chunk_marks.append((time.perf_counter(), audio_ms))

    def mark_ready(self, audio_ms: float | None = None) -> None:
        """:meth:`mark_chunk` for an output the caller has already waited for on its own
        (the event of its host copy, :class:`~kernel_agent.workloads.serving.HostCopy`): no
        device-wide synchronize, so a request is marked when *its* output arrived, not when
        the work queued after it (the next requests, a post stage on a side stream)
        finished."""
        self.chunk_marks.append((time.perf_counter(), audio_ms))

    def async_flags(self, depth: int = 2) -> AsyncFlags:
        """A reader of a generation loop's per-step device flags (stop tokens, a finished
        mask; :class:`~kernel_agent.workloads.serving.AsyncFlags`): ``ticket =
        flags.send(stop)`` as soon as they exist copies them to pinned memory without
        blocking, ``flags.read(ticket)`` after the rest of the step is queued (or a step
        later) waits for that copy only. Same values as ``stop.cpu()``, without the GPU
        idling while the host catches up."""
        from kernel_agent.workloads.serving import AsyncFlags

        return AsyncFlags(depth)

    def serving(self) -> ServingOptions | None:
        """The serving opt-in of this run's options (``-o serving=static|continuous``,
        ``requests``, ``pipeline``; :func:`~kernel_agent.workloads.serving.serving_options`),
        None for the default benchmark. A workload that supports it runs its request queue
        through :func:`~kernel_agent.workloads.serving.serve`."""
        from kernel_agent.workloads.serving import serving_options

        return serving_options(self.options)

    def check_serving(self) -> None:
        """``ValueError`` for a malformed serving opt-in, or one this workload does not
        implement (``supports_serving``): never run the default benchmark in its place."""
        if self.serving() is not None and not type(self).supports_serving:
            raise ValueError(
                f"{type(self).__name__} does not implement -o serving=static|continuous "
                "(a request queue over its batch slots, kernel_agent.workloads.serving); "
                "a harness opts in with `supports_serving = True`"
            )

    def metric_value(self, start: float, end: float) -> tuple[float, dict[str, Any]]:
        """``(value in ms, per-run details)`` of the metric for one run of :meth:`run`
        that started at ``start`` and ended at ``end`` (``time.perf_counter()``, both
        GPU-synchronised), with :attr:`chunk_marks` as marked during the run.

        ``ttfa``: the first mark minus ``start``; the details hold the median latency of
        the next ``steady_chunks`` chunks (``chunk_ms``), its real-time factor (``rtf``)
        and the full run (``run_ms``), none for a run inside :meth:`metric_window` (it
        stopped at its first chunk: the full request of the measurement reports them).

        ``throughput``: the run's wall time per second of generated audio
        (:meth:`output_seconds`), i.e. ``1000 / throughput`` ms, so that lower is better
        and a speedup is the throughput ratio; the details hold the ``throughput`` (audio
        seconds per wall second), ``audio_s``, ``run_ms`` and the median latency of the
        requests (``request_ms``: each marked when its output was ready; also
        ``request_ms_mean`` and ``request_ms_max``). The extension point for new metrics."""
        total = (end - start) * 1000
        metric = self.metric
        if metric == objective.LATENCY:
            return total, {}
        if metric == objective.THROUGHPUT:
            seconds = self.output_seconds()
            if not seconds or seconds <= 0:
                raise RuntimeError(
                    "metric=throughput: the run generated no audio (Workload.output_seconds)"
                )
            done = [(t - start) * 1000 for t, _ in self.chunk_marks] or [total]
            return total / seconds, {
                "throughput": seconds * 1000 / total,
                "audio_s": seconds,
                "run_ms": total,
                "requests": len(self.chunk_marks),
                "request_ms": statistics.median(done),
                "request_ms_mean": statistics.fmean(done),
                "request_ms_max": max(done),
            }
        if metric != objective.TTFA:
            raise NotImplementedError(f"metric={metric} is not implemented")
        marks = self.chunk_marks
        if not marks:
            raise RuntimeError(
                "metric=ttfa: the run produced no audio chunk (Workload.mark_chunk was never "
                "called): the streaming path was bypassed"
            )
        if self.in_window:  # stopped at its first chunk: no steady state, no full run
            return (marks[0][0] - start) * 1000, {}
        k = int(self.options.get("steady_chunks", objective.STEADY_CHUNKS))
        steady = marks[: k + 1]
        gaps = [(b[0] - a[0]) * 1000 for a, b in itertools.pairwise(steady)]
        chunk_ms = statistics.median(gaps) if gaps else None
        audio = [a for _, a in steady[1:] if a]
        rtf = chunk_ms / statistics.median(audio) if chunk_ms is not None and audio else None
        return (marks[0][0] - start) * 1000, {
            "run_ms": total,
            "chunks": len(marks),
            "steady_chunks": len(gaps),
            "chunk_ms": chunk_ms,
            "rtf": rtf,
        }

    def output_seconds(self) -> float:
        """``metric=throughput``: seconds of audio the last run of :meth:`run` generated; by
        default what it marked (``mark_chunk(audio_ms=...)``). A workload whose output
        length is fixed by its options returns that length, so the objective cannot be
        inflated by what a candidate reports."""
        return sum(a for _, a in self.chunk_marks if a) / 1000

    def self_check(self, inputs: Any, reference: Any) -> dict[str, Any] | None:
        """Optional ``analyze`` check that the workload's own run is faithful to the model
        (a batched loop against the model's batch-1 inference, say), given the baseline
        output ``reference`` of ``inputs``: ``{"passed", "reason", "check", ...}``, stored
        in ``baseline.json`` ``self_check``. ``None`` (the default): nothing to check."""
        return None

    def perceptual_samples(self) -> list[dict[str, Any]]:
        """Held-out samples of the perceptual gate (``--quality near-lossless``,
        :mod:`kernel_agent.workloads.perceptual`): option overrides of :meth:`run`, one per
        sample, plain JSON-able values (TTS: ``text`` + ``language`` + ``seed`` at natural
        length; LLM: held-out prompts). ``[]`` (the default): the workload has no
        perceptual gate, and a near-lossless run keeps the exact checks.

        ``analyze`` runs them free running on the baseline, scores them
        (:meth:`perceptual_quality`) and stores outputs and scores in ``.truth/``;
        ``e2e`` runs the same samples on every candidate, untimed, and compares the scores
        paired (:meth:`compare_perceptual`). Both go through :meth:`run`, the code path the
        run's metric times (``metric=ttfa``: the whole streamed output). Keep them few and
        short."""
        return []

    def perceptual_quality(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Scores of generated samples (``{"options", "output"}`` each: the overrides of
        :meth:`perceptual_samples` and the output of :meth:`run` under them; in ``e2e`` also
        ``reference``, the baseline's scores of the same sample), one dict per sample, plain
        data (TTS: transcript + error rate against the text, speaker embedding, MOS; LLM:
        teacher forced on the baseline's continuation, KL and top-1 agreement, and the
        log-likelihood of its own continuation). Load scoring models here, lazily, and free
        them before returning."""
        raise NotImplementedError(f"{type(self).__name__} has no perceptual gate")

    def compare_perceptual(
        self, reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
    ) -> Comparison:
        """Paired comparison of the candidate's :meth:`perceptual_quality` scores with the
        baseline's (sample *i* with sample *i*): pass when the perceptual quality stays
        within the noise of the baseline (calibrated thresholds)."""
        raise NotImplementedError(f"{type(self).__name__} has no perceptual gate")

    @contextlib.contextmanager
    def with_options(self, overrides: dict[str, Any] | None) -> Iterator[None]:
        """Apply option ``overrides`` (in place, so references to ``self.options``
        see them) and restore the previous options afterwards."""
        saved = dict(self.options)
        self.options.update(overrides or {})
        try:
            yield
        finally:
            self.options.clear()
            self.options.update(saved)

    def describe(self) -> str:
        opts = ", ".join(f"{k}={v}" for k, v in sorted(self.options.items()))
        return f"{type(self).__name__}({self.spec.repo_id}, {self.spec.dtype}; {opts})"


# ---------------------------------------------------------------- helpers


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_run(
    workload: Workload, inputs: Any, *, window: bool = False
) -> tuple[Any, float, dict[str, Any]]:
    """One GPU-synchronised run of ``workload.run``: ``(output, value of the workload's
    metric in ms, its per-run details)`` (:meth:`Workload.metric_value`; with the counters
    the run reported, :meth:`Workload.report_stats`, as ``decode_stats``). The clock starts
    before ``workload.run`` is called, so whatever a transform does around it counts.
    ``window``: inside :meth:`Workload.metric_window` (``ttfa``: the request up to its first
    chunk; the output is the window's, the full-run details are missing)."""
    workload.chunk_marks.clear()
    workload.run_stats = {}
    with workload.metric_window() if window else contextlib.nullcontext():
        with torch.inference_mode():
            synchronize()
            start = time.perf_counter()
            output = workload.run(inputs)
            synchronize()
            end = time.perf_counter()
        ms, detail = workload.metric_value(start, end)
        short = workload.in_window
    if workload.run_stats and not short:  # a request stopped at the window: partial counts
        detail = {**detail, "decode_stats": dict(workload.run_stats)}
    return output, ms, detail


def decode_stats(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The counters of a few runs (:meth:`Workload.report_stats`; one dict per run, ``{}``
    where a run reported none) as their medians, plus ``acceptance_rate`` (accepted /
    drafted), ``tokens_per_verify`` (the accepted drafts plus the model's own token of each
    verification, per verification) and ``tokens_per_step``. None: no run reported any."""
    runs = [r for r in runs if r]
    if not runs:
        return None
    out: dict[str, Any] = objective.aggregate(runs)
    num = {k: float(v) for k, v in out.items() if isinstance(v, int | float)}
    if num.get("drafted"):
        out["acceptance_rate"] = round(num.get("accepted", 0.0) / num["drafted"], 4)
    if num.get("verifies"):
        verifies = num["verifies"]
        out["tokens_per_verify"] = round((num.get("accepted", 0.0) + verifies) / verifies, 3)
    if num.get("steps") and num.get("tokens"):
        out["tokens_per_step"] = round(num["tokens"] / num["steps"], 3)
    return out


def measure(workload: Workload, inputs: Any, *, warmup: int = 1, iters: int = 3) -> dict[str, Any]:
    """The workload's metric over ``iters`` GPU-synchronised runs after ``warmup`` untimed
    ones, in milliseconds: the wall-clock latency of ``workload.run`` by default, the time
    to first audio for ``metric=ttfa`` (:mod:`kernel_agent.objective`). ``median_ms`` is
    the optimiser's objective; other metrics add ``metric_detail`` (medians over the runs),
    and so do counters the runs reported (``metric_detail.decode_stats``, :func:`decode_stats`).

    A windowed metric (``ttfa``, :attr:`Workload.windowed`): the first warm-up and the last
    timed run are whole requests (every step warmed up; the ``output`` the quality checks
    judge and the full-run details), the other runs stop at the end of the
    :meth:`~Workload.metric_window`. Still ``iters`` samples of the metric, for one
    streamed request and ``iters - 1`` short ones. Other metrics: every run is whole."""
    output = None
    if torch.cuda.is_available():
        # Same clock warm-up for baseline and optimised runs (fair comparison).
        from kernel_agent.kernels.bench import warm_gpu

        warm_gpu(500.0)
    with torch.inference_mode():
        for i in range(warmup):
            # the first warm-up runs every step (lazy init, graph capture, compilation);
            # the others warm up what the windowed runs time
            short = i > 0 and iters > 0
            with workload.metric_window() if short else contextlib.nullcontext():
                output = workload.run(inputs)
        synchronize()
    times, details, counters = [], [], []
    for i in range(iters):
        output, ms, detail = timed_run(workload, inputs, window=i < iters - 1)
        times.append(ms)
        counters.append(detail.pop("decode_stats", {}))
        details.append(detail)
    result: dict[str, Any] = {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "times_ms": times,
        "metric": workload.metric,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0,
        "output": output,
    }
    if workload.metric != objective.LATENCY:
        result["metric_detail"] = objective.aggregate(details)
    if (stats := decode_stats(counters)) is not None:  # Workload.report_stats
        result["metric_detail"] = {**result.get("metric_detail", {}), "decode_stats": stats}
    return result


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    n = min(a.numel(), b.numel())
    if n == 0:
        return 1.0
    a, b = a[:n], b[:n]
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not bool(finite.all()):
        # Masked logits (-inf) must match exactly; compare the rest numerically.
        if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
            return 0.0
        a, b = a[finite], b[finite]
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b) / denom)


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float | None = None) -> float:
    a = a.detach().float()
    b = b.detach().float()
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float("inf")
    rng = data_range if data_range is not None else float(a.max() - a.min()) or 1.0
    return 10.0 * torch.log10(torch.tensor(rng**2 / mse)).item()


def compare_tokens(
    ref_tokens: torch.Tensor,
    new_tokens: torch.Tensor,
    ref_logits: torch.Tensor | None,
    new_logits: torch.Tensor | None,
    *,
    min_prefix: int,
    min_cosine: float,
    ref_margins: torch.Tensor | None = None,
    near_tie: float = 0.0,
) -> Comparison:
    """Greedy-decoding comparison: first-step logits must agree and the first
    ``min_prefix`` generated tokens must be identical.  Later divergence is
    tolerated because low-precision kernels legitimately flip near-ties.

    ``ref_margins`` (the baseline's top-1 minus top-2 logit of every step): an earlier
    divergence passes too when it happens at a near-tie of the baseline (a margin at most
    ``near_tie``), a choice that rounding flips; after it nothing more can be compared."""
    ref = ref_tokens.flatten().tolist()
    new = new_tokens.flatten().tolist()
    prefix = 0
    for x, y in zip(ref, new, strict=False):
        if x != y:
            break
        prefix += 1
    total = max(len(ref), 1)
    metrics: dict[str, float | int | str] = {
        "token_prefix_match": prefix,
        "token_match_ratio": round(prefix / total, 4),
        "generated_tokens": len(new),
    }
    passed = prefix >= min(min_prefix, len(ref))
    reason = "" if passed else f"tokens diverge at position {prefix}"
    margins = ref_margins.flatten() if ref_margins is not None else None
    if not passed and margins is not None and prefix < min(len(ref), margins.numel()):
        margin = float(margins[prefix])
        metrics["divergence_margin"] = round(margin, 4)
        reason += f" (baseline top-2 logit margin {margin:.3f} there)"
        if margin <= near_tie and len(new) >= len(ref):
            passed, reason = True, ""
            metrics["tolerated"] = f"a near-tie of the baseline (margin <= {near_tie:g})"
    if ref_logits is not None and new_logits is not None:
        cos = cosine(ref_logits, new_logits)
        metrics["first_logits_cosine"] = round(cos, 6)
        same_top1 = bool(torch.equal(ref_logits.float().argmax(-1), new_logits.float().argmax(-1)))
        metrics["first_top1_match"] = int(same_top1)
        if cos < min_cosine:
            passed = False
            reason = f"first-step logits cosine {cos:.5f} < {min_cosine}"
    return Comparison(passed, metrics, reason)


def compare_audio(
    ref: torch.Tensor, new: torch.Tensor, *, min_spec_cosine: float, max_len_ratio: float = 0.1
) -> Comparison:
    """Spectral comparison (robust to tiny phase/sample differences)."""
    ref = ref.detach().float().flatten()
    new = new.detach().float().flatten()
    len_diff = abs(ref.numel() - new.numel()) / max(ref.numel(), 1)
    n = min(ref.numel(), new.numel())
    if n < 1024:
        return Comparison(False, {"samples": n}, "audio too short to compare")

    def spec(x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(1024)
        mag = torch.stft(x[:n], 1024, 256, window=window, return_complex=True).abs()
        return torch.log1p(mag)

    cos = cosine(spec(ref), spec(new))
    metrics: dict[str, float | int | str] = {
        "spectral_cosine": round(cos, 5),
        "length_diff_ratio": round(len_diff, 4),
        "waveform_cosine": round(cosine(ref[:n], new[:n]), 5),
    }
    if len_diff > max_len_ratio:
        return Comparison(False, metrics, f"length differs by {len_diff:.1%}")
    if cos < min_spec_cosine:
        return Comparison(False, metrics, f"spectral cosine {cos:.4f} < {min_spec_cosine}")
    return Comparison(True, metrics)


def compare_steps(
    ref_steps: torch.Tensor,
    new_steps: torch.Tensor,
    *,
    min_step_cosine: float,
    min_mean_step_cosine: float,
    max_rms_change: float = 0.1,
) -> Comparison:
    """Teacher-forced comparison of per-step predictions (dim 0 = step).

    ``new_steps[i]`` was predicted from the *reference* history, so a kernel's
    error shows up once per step instead of compounding along the trajectory.
    The mean cosine catches small systematic errors, the minimum a single bad
    step, and the RMS ratio magnitude errors that cosines cannot see."""
    if ref_steps.shape[0] == 0:
        return Comparison(False, {"steps": 0}, "the reference recorded no steps")
    if ref_steps.shape[0] != new_steps.shape[0]:
        return Comparison(
            False,
            {"steps": int(new_steps.shape[0]), "reference_steps": int(ref_steps.shape[0])},
            f"{new_steps.shape[0]} teacher-forced steps != {ref_steps.shape[0]} reference steps",
        )
    if ref_steps.shape[1:] != new_steps.shape[1:]:
        return Comparison(
            False,
            {"steps": int(new_steps.shape[0])},
            f"step shape {tuple(new_steps.shape[1:])} != {tuple(ref_steps.shape[1:])}",
        )
    ref = ref_steps.detach().float().flatten(1)
    new = new_steps.detach().float().flatten(1)
    cosines = [cosine(r, n) for r, n in zip(ref, new, strict=True)]
    rel_err = (new - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-12)
    worst = min(range(len(cosines)), key=cosines.__getitem__)
    mean_cos = sum(cosines) / len(cosines)
    ref_rms = float(ref.pow(2).mean().sqrt())
    rms_ratio = float(new.pow(2).mean().sqrt()) / ref_rms if ref_rms > 0 else 1.0
    metrics: dict[str, float | int | str] = {
        "steps": len(cosines),
        "min_step_cosine": round(cosines[worst], 6),
        "mean_step_cosine": round(mean_cos, 6),
        "worst_step": worst,
        "mean_rel_error": round(float(rel_err.mean()), 6),
        "max_rel_error": round(float(rel_err.max()), 6),
        "rms_ratio": round(rms_ratio, 6),
    }
    if not bool(torch.isfinite(new).all()):
        return Comparison(False, metrics, "non-finite teacher-forced predictions")
    if mean_cos < min_mean_step_cosine:
        return Comparison(
            False, metrics, f"mean step cosine {mean_cos:.5f} < {min_mean_step_cosine}"
        )
    if cosines[worst] < min_step_cosine:
        return Comparison(
            False, metrics, f"step {worst} cosine {cosines[worst]:.5f} < {min_step_cosine}"
        )
    if abs(rms_ratio - 1.0) > max_rms_change:
        return Comparison(False, metrics, f"RMS x{rms_ratio:.4f} (allowed ±{max_rms_change:.0%})")
    return Comparison(True, metrics)


def compare_stop(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tolerance: int = 0,
    near_tie: float = STOP_NEAR_TIE,
) -> Comparison:
    """Natural-length comparison (:meth:`Workload.natural_length_run`): the candidate
    must stop after as many steps as the baseline.

    Replayed teacher forced, every stop decision is an argmax over a hidden state on
    the baseline trajectory, so the check is exact by default. ``tolerance`` > 0 accepts
    a candidate that stops up to that many steps away when the baseline's stop margin
    (stop minus continue logit, ``reference["stop_margins"]``) at the first step where
    the two decisions differ is within ``near_tie`` of zero, a near-tie that rounding can
    flip. Without recorded margins the check stays exact. The margins are reported."""
    ref_steps, new_steps = int(reference["steps"]), int(candidate["steps"])
    margins = reference.get("stop_margins")
    if margins is not None and len(margins) != ref_steps:
        margins = None
    metrics: dict[str, float | int | str] = {"steps": new_steps, "reference_steps": ref_steps}
    if margins:
        metrics["reference_stop_margin"] = round(float(margins[-1]), 4)
    if new_steps == ref_steps:
        ref_len, new_len = reference.get("output_length"), candidate.get("output_length")
        if ref_len is None or new_len is None or int(ref_len) == int(new_len):
            return Comparison(True, metrics)
        metrics.update(output_length=int(new_len), reference_output_length=int(ref_len))
        return Comparison(
            False, metrics, f"output length {new_len} != {ref_len} after the same {new_steps} steps"
        )
    max_steps = reference.get("max_steps")
    if new_steps > ref_steps and max_steps is not None and new_steps >= int(max_steps):
        what = f"never fires (ran to max_steps={max_steps})"
    elif new_steps > ref_steps:
        what = f"fires {new_steps - ref_steps} step(s) late"
    else:
        what = f"fires {ref_steps - new_steps} step(s) early"
    reason = f"the stop condition {what}: {new_steps} steps, the baseline {ref_steps}"
    if not margins:
        return Comparison(False, metrics, reason)
    # The first step whose decisions differ: the baseline stops at its last step while
    # the candidate goes on, or the candidate stops at an earlier one.
    step = min(ref_steps, new_steps) - 1
    margin = float(margins[step])
    metrics.update(differing_step=step, differing_step_margin=round(margin, 4))
    reason += f" (baseline stop margin {margin:+.3f} at step {step})"
    consulted = step > int(reference.get("min_steps", 0))  # before that the stop is ignored
    if abs(new_steps - ref_steps) <= tolerance and abs(margin) <= near_tie and consulted:
        metrics["tolerated"] = f"{reason}: a near-tie (|margin| <= {near_tie:g})"
        return Comparison(True, metrics)
    return Comparison(False, metrics, reason)
