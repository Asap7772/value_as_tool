"""Adapter base for the existing, regression-tested solver protocols."""

from __future__ import annotations

from value_as_tool.harnesses.base import HarnessRuntime, HarnessSpec


class LegacyHarness:
    spec: HarnessSpec

    async def run(self, runtime: HarnessRuntime) -> None:
        condition = self.spec.condition
        if condition is None:  # pragma: no cover - invalid subclass guard
            raise RuntimeError("legacy harness requires a condition")
        await runtime.run_builtin(condition)


__all__ = ["LegacyHarness"]
