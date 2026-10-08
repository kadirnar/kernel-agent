# Native projects: layout, build, timing

Part of the `native-engines` skill.

## Projects

```
<stage id>/
  kernel_project.toml     [project] name, entry, kind; [build] backend, sources, flags
  candidate.py            build(reference) (kind = "kernel") / apply(workload) ("transform")
  include/*.cuh           device helpers shared by the .cu files
  csrc/*.cu csrc/binding.cpp
```

* `project.load(__file__)` (`from kernel_agent.native import project`) compiles the
  project once per content digest and toolchain (`~/.cache/kernel-agent/native/`) and
  returns it; its attributes are the functions `binding.cpp` binds.
* kernel-agent's toolkit headers are on every build's include path:
  `#include "ka_launch.cuh"` for PDL and cooperative launches (`ka_launch`,
  `ka_pdl_wait`, `ka_pdl_launch_dependents`, `ka_coresident_blocks`; see the `cuda-graphs-streams-pdl` skill).
* `python -m kernel_agent.native.project check <dir>` validates the manifest and files;
  `... build <dir>` compiles on the CPU (no GPU, no evaluation used): fix compiler errors
  there. The tools also compile a project before its evaluation, outside the GPU lock.
* `backend = "command"` runs your own build (`command = ["bash", "{src}/build.sh"]`,
  CMake, make) and loads `outputs` (`load = "torch_ops"` for `TORCH_LIBRARY` libraries,
  `"python"` for extension modules, `"none"` for ctypes).
* Text files only (sources, headers, scripts), at most 200 files / 4 MB; `build/` and
  hidden directories are not part of the project.
* Evaluations snapshot the project's **bundle**: one `.py` file holding every file and the
  project's sha256, which the evaluator, the sweep (`build(reference, **config)` keyword
  arguments), memcheck, the integration and the export use like any candidate file.

## Timing

Stage targets are timed per captured case against the reference stage (CUDA events, full
clocks, interleaved rounds), end-to-end runs as the median of the workload's runs after a
warm-up (compile and capture time excluded). Inside CUDA graphs host overhead does not
count; a persistent engine is still judged by the whole stage's GPU time.
