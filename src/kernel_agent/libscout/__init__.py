"""The library scout: drop-in library kernels as automatic first candidates and the bar to
beat (issue #227).

Agents found library choices late and by hand (cuDNN over flash attention at 11 tokens,
cuBLASLt's timed algorithms for an FP8 layer), and a plain configuration search beat agents
on InferenceBench. The scout tries what installed libraries give on every target before an
engineer starts, deterministically and with no Claude session:

* :mod:`.detect`: the op families of what the reference calls (a ``TorchFunctionMode``
  trace of one captured case; never module names);
* :mod:`.registry` / :mod:`.adapters`: one adapter per library and family, with its
  availability probe, architectures, dtypes and candidate template;
* :mod:`.fx_rewrites` / :mod:`.template`: the candidate: the reference's own FX graphs
  (TorchDynamo, no Inductor) with the library's ops pointed in;
* :mod:`.probe`: the GPU step (detection, decisions, candidates, op bars);
* :mod:`.scout`: the step in a run (sweeps, ledger rows ``library:<package>@<version>``,
  the remembered scout keyed by the installed library versions and the GPU, the library
  bar of the digest, the planner, ``ceilings.md`` and ``report.md``, export requirements,
  ``doctor``).
"""

from __future__ import annotations
