"""
UMRS Step interface — abstract base class for user-defined processing steps.

Users implement UMRSStep subclasses in their own Python scripts, then
reference them from a workflow.json file.  UMRS discovers, loads, and
executes these steps without any internal code changes.

Typical usage in a user script::

    from umrs.steps import UMRSStep, StepContext, StepResult

    class MyPreprocess(UMRSStep):
        name = "preprocess"

        def setup(self, config: dict) -> None:
            self.size = config.get("target_size", 1024)

        def run(self, ctx: StepContext) -> StepResult:
            out_keys = []
            for key in ctx.input_keys:
                data = ctx.download_bytes(key)
                processed = self._do_work(data)
                out_key = f"preprocessed/{key}"
                ctx.upload_bytes(out_key, processed)
                out_keys.append(out_key)
            return StepResult(output_keys=out_keys)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from umrs.steps.context import StepContext, StepResult


class UMRSStep(ABC):
    """Abstract base class every user-defined step must inherit."""

    # Subclasses MUST set a unique name (used as key in workflow.json).
    name: str = ""

    def setup(self, config: dict) -> None:
        """Optional one-time initialisation (load weights, parse config, …).

        Called once after instantiation, before any ``run()`` calls.
        Override this instead of ``__init__`` so the loader can instantiate
        the class without arguments.
        """

    @abstractmethod
    def run(self, ctx: StepContext) -> StepResult:
        """Execute the step on one batch of S3 objects.

        Args:
            ctx: A :class:`StepContext` that provides S3 helpers,
                 ``input_keys``, ``output_bucket``, and a shared
                 ``metadata`` dict for passing state between steps.

        Returns:
            A :class:`StepResult` describing what was produced.
        """

    def teardown(self) -> None:
        """Optional cleanup (release GPU memory, close files, …).

        Called once after **all** batches have been processed for this step.
        """
