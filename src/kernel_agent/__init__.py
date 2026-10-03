"""kernel-agent: Claude-powered kernel-level optimisation of Hugging Face models.

Quick start::

    kernel-agent doctor --smoke
    kernel-agent optimize https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct

Python API::

    import asyncio
    from kernel_agent import OptimizeConfig, optimize
    run = asyncio.run(optimize(OptimizeConfig(model_ref="org/model")))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kernel_agent.config import OptimizeConfig

__version__ = "0.1.0"
__all__ = ["OptimizeConfig", "__version__", "optimize"]

if TYPE_CHECKING:
    from kernel_agent.workspace import RunDir


async def optimize(cfg: OptimizeConfig, until: str | None = None) -> RunDir:
    """Run the full pipeline (see :mod:`kernel_agent.orchestrator`)."""
    from kernel_agent.orchestrator import optimize as _optimize

    return await _optimize(cfg, until=until)
