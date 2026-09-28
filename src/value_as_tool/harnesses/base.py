"""Stable interface for source-hashed proof harness implementations.

Harnesses own orchestration policy while :class:`HarnessRuntime` keeps model
access, token accounting, durable call records, and checkpointing inside the
trusted evaluator.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from value_as_tool.schemas import (
    CandidateRecord,
    ChatCompletion,
    Condition,
    Role,
    TrajectoryRequest,
    TrajectoryStatus,
    VerdictRecord,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


HarnessAccess = Literal["blind", "reference_assisted"]


@dataclass(frozen=True, slots=True)
class HarnessSpec:
    """Immutable public metadata attached to one harness implementation."""

    harness_id: str
    display_name: str
    access: HarnessAccess = "blind"
    condition: Condition | None = None
    requires_reference: bool = False
    entrypoint: str = ""
    source_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.harness_id or not self.display_name:
            raise ValueError("harness id and display name must be nonempty")
        if self.access not in ("blind", "reference_assisted"):
            raise ValueError(f"unsupported harness access class: {self.access!r}")
        if self.requires_reference != (self.access == "reference_assisted"):
            raise ValueError(
                "requires_reference must agree with the harness access class"
            )


class HarnessAbort(RuntimeError):
    """Terminate a trajectory with an explicit, reportable status."""

    def __init__(self, status: TrajectoryStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


@runtime_checkable
class ProofHarness(Protocol):
    """A proof-solving policy evaluated by the trusted runtime."""

    spec: HarnessSpec

    async def run(self, runtime: HarnessRuntime) -> None: ...


class HarnessRuntime:
    """Capability-limited facade exposed to a proof harness.

    The wrapped orchestrator is deliberately not public. Harness code can
    choose prompts, call order, and tools, but cannot bypass the shared budget
    or the artifact lifecycle.
    """

    def __init__(self, orchestrator: Any, state: Any, spec: HarnessSpec) -> None:
        self.__orchestrator = orchestrator
        self.__state = state
        request = state.result.request
        if spec.access == "blind":
            safe_metadata = {
                key: value
                for key, value in request.metadata.items()
                if not any(
                    marker in key.casefold()
                    for marker in ("answer", "grading", "reference", "rubric", "solution")
                )
            }
            request = request.__class__(
                benchmark=request.benchmark,
                problem_id=request.problem_id,
                problem=request.problem,
                condition=request.condition,
                seed=request.seed,
                harness_id=request.harness_id,
                reference_proof=None,
                solver_prompt=request.solver_prompt,
                metadata=safe_metadata,
            )
        self.__request = request

    @property
    def request(self) -> TrajectoryRequest:
        return self.__request

    @property
    def minimum_call_tokens(self) -> int:
        return int(self.__orchestrator.config.minimum_call_tokens)

    @property
    def remaining_generated_tokens(self) -> int:
        return int(self.__state.budget.remaining_generated_tokens)

    def stable_seed(self, label: str) -> int:
        return self.__orchestrator.stable_seed_for_harness(
            self.request.seed,
            self.request.problem_id,
            label,
        )

    async def run_builtin(self, condition: Condition) -> None:
        """Run an existing protocol without duplicating its mature logic."""

        if self.request.condition is not condition:
            raise HarnessAbort(
                TrajectoryStatus.PROTOCOL_ERROR,
                f"harness condition mismatch: {self.request.condition!r} != {condition.value!r}",
            )
        if condition is Condition.DIRECT:
            await self.__orchestrator._run_direct(self.__state)
        elif condition in {
            Condition.VALUE_TOOL,
            Condition.VALUE_TOOL_RATIONALE_SCORE,
        }:
            await self.__orchestrator._run_value_tool(self.__state)
        else:
            await self.__orchestrator._run_gvr(self.__state)

    async def call(
        self,
        *,
        role: Role,
        cycle: int,
        label: str,
        messages: Sequence[Mapping[str, Any]],
        cap: int,
        keep: int,
        seed: int,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        parallel_tool_calls: bool = True,
        force_no_thinking: bool = False,
        use_sampling: bool | None = True,
    ) -> tuple[ChatCompletion, int]:
        """Make or replay one uniquely labelled, budgeted model call."""

        matches = [call for call in self.__state.result.calls if call.label == label]
        if len(matches) > 1:
            raise HarnessAbort(
                TrajectoryStatus.PROTOCOL_ERROR,
                f"checkpoint contains duplicate harness call label {label!r}",
            )
        if matches:
            existing = matches[0]
            expected_messages = tuple(copy.deepcopy(dict(item)) for item in messages)
            expected_tools = tuple(copy.deepcopy(dict(item)) for item in (tools or ()))
            if (
                existing.role is not role
                or existing.cycle != cycle
                or existing.seed != seed
                or existing.messages != expected_messages
                or existing.tools != expected_tools
                or existing.response is None
                or existing.error is not None
            ):
                raise HarnessAbort(
                    TrajectoryStatus.PROTOCOL_ERROR,
                    f"checkpoint call {label!r} does not match the harness request",
                )
            return existing.response, existing.index
        try:
            return await self.__orchestrator._call(
                self.__state,
                role=role,
                cycle=cycle,
                label=label,
                messages=messages,
                cap=cap,
                keep=keep,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                force_no_thinking=force_no_thinking,
                use_sampling=use_sampling,
            )
        except Exception as exc:
            # Preserve the orchestrator's status-bearing aborts without making
            # private implementation types part of the harness API.
            status = getattr(exc, "status", None)
            if isinstance(status, TrajectoryStatus):
                raise HarnessAbort(status, str(exc)) from exc
            raise

    async def add_candidate(
        self,
        *,
        cycle: int,
        role: Role,
        content: str,
        reasoning: str | None,
        call_index: int,
    ) -> CandidateRecord:
        existing = next(
            (
                candidate
                for candidate in self.__state.result.candidates
                if candidate.cycle == cycle
            ),
            None,
        )
        if existing is not None:
            if (
                existing.role is not role
                or existing.content != content
                or existing.reasoning != reasoning
                or existing.call_index != call_index
            ):
                raise HarnessAbort(
                    TrajectoryStatus.PROTOCOL_ERROR,
                    f"checkpoint candidate {cycle} does not match the harness output",
                )
            return existing
        record = CandidateRecord(
            cycle=cycle,
            role=role,
            content=content,
            reasoning=reasoning,
            call_index=call_index,
        )
        self.__state.result.candidates.append(record)
        await self.__state.sync()
        return record

    async def add_verdict(self, record: VerdictRecord) -> VerdictRecord:
        existing = next(
            (
                verdict
                for verdict in self.__state.result.verdicts
                if verdict.cycle == record.cycle
            ),
            None,
        )
        if existing is not None:
            if existing != record:
                raise HarnessAbort(
                    TrajectoryStatus.PROTOCOL_ERROR,
                    f"checkpoint verdict {record.cycle} does not match the harness output",
                )
            return existing
        self.__state.result.verdicts.append(record)
        await self.__state.sync()
        return record

    async def transition(
        self,
        cycle: int,
        source: str,
        action: str,
        target: str,
        detail: str = "",
    ) -> None:
        await self.__orchestrator._transition_once(
            self.__state,
            cycle,
            source,
            action,
            target,
            detail=detail,
        )

    async def finish(
        self,
        output: str,
        *,
        status: TrajectoryStatus,
    ) -> None:
        if not output.strip():
            raise HarnessAbort(
                TrajectoryStatus.PROTOCOL_ERROR,
                "harness produced an empty final proof",
            )
        self.__state.result.final_output = output
        self.__state.result.status = status
        await self.__state.sync()

    def abort(self, status: TrajectoryStatus, message: str) -> None:
        raise HarnessAbort(status, message)


def module_source_path(value: type[Any]) -> Path:
    """Return the concrete Python source backing a harness class."""

    import inspect

    path = inspect.getsourcefile(value)
    if path is None:
        raise ValueError(f"cannot locate source for harness class {value!r}")
    return Path(path).resolve()


__all__ = [
    "HarnessAbort",
    "HarnessAccess",
    "HarnessRuntime",
    "HarnessSpec",
    "ProofHarness",
    "module_source_path",
]
