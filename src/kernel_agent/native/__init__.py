"""Native engines: multi-file CUDA / C++ projects and the systems-native agent (issue #134).

* :mod:`kernel_agent.native.project`: project candidates (a directory with headers, several
  ``.cu`` files and a build manifest) and their bundles, digests and build cache, which make
  them evaluable, snapshotable, memcheckable and integrable like single-file candidates.
* :mod:`kernel_agent.native.engine`: native engine targets (one stage, a group of stages or
  the whole generation loop) derived from the profile's stage graph, the staged plan, and
  when the improve loop opens the native arm (module arms plateaued; opt-in).

Design: ``docs/NATIVE.md``; agent-facing contract: ``agent/knowledge/native.md``.
"""
