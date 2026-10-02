"""Stable interface for source-hashed proof harness implementations.

Harnesses own orchestration policy while :class:`HarnessRuntime` keeps model
access, token accounting, durable call records, and checkpointing inside the
trusted evaluator.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
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


HarnessAccess = Literal[
    "blind", "reference_assisted", "attempt_assisted", "attempt_and_reference_assisted"
]
ATTEMPT_CONDITIONING_MODES = ("solutions", "solution_summary", "thinking_summary")


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
    conditioning_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.harness_id or not self.display_name:
            raise ValueError("harness id and display name must be nonempty")
        if self.access not in (
            "blind", "reference_assisted", "attempt_assisted", "attempt_and_reference_assisted"
        ):
            raise ValueError(f"unsupported harness access class: {self.access!r}")
        if self.requires_reference != (
            self.access in {"reference_assisted", "attempt_and_reference_assisted"}
        ):
            raise ValueError(
                "requires_reference must agree with the harness access class"
            )
        if (
            self.conditioning_mode is not None
            and self.conditioning_mode not in ATTEMPT_CONDITIONING_MODES
        ):
            raise ValueError(f"unsupported conditioning mode: {self.conditioning_mode!r}")
        if (self.conditioning_mode is not None) != (
            self.access in {"attempt_assisted", "attempt_and_reference_assisted"}
        ):
            raise ValueError("conditioning_mode must agree with the harness access class")


class HarnessAbort(RuntimeError):
    """Terminate a trajectory with an explicit, reportable status."""

    def __init__(self, status: TrajectoryStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class HarnessReplayError(HarnessAbort):
    """A checkpoint or label conflict: never contained by a harness."""

    def __init__(self, message: str) -> None:
        super().__init__(TrajectoryStatus.PROTOCOL_ERROR, message)


class VerdictParseError(HarnessAbort):
    """No usable verdict after the in-budget recovery attempt; usage is known."""

    def __init__(self, message: str) -> None:
        super().__init__(TrajectoryStatus.PROTOCOL_ERROR, message)


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
        if not spec.requires_reference:
            safe_metadata = {
                key: value
                for key, value in request.metadata.items()
                if not any(
                    marker in key.casefold()
                    for marker in (
                        "answer", "grading", "reference", "rubric", "solution",
                        "evidence", "conditioning",
                    )
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
        # Conditioning is a capability of the trusted verifier paths, never
        # a general-purpose prompt ingredient exposed to harness generators.
        request = replace(request, verifier_evidence=None)
        self.__request = request
        # Labels of calls currently dispatched; concurrent harness tasks must
        # never issue the same label twice.
        self.__inflight_labels: set[str] = set()

    @property
    def request(self) -> TrajectoryRequest:
        return self.__request

    @property
    def minimum_call_tokens(self) -> int:
        return int(self.__orchestrator.config.minimum_call_tokens)

    @property
    def total_generated_tokens(self) -> int:
        """The trajectory's whole generated-token budget."""

        return int(self.__orchestrator.config.total_generated_tokens)

    @property
    def cch_stage_tokens(self) -> int | None:
        """Optional allowance for each stage of the plan/work/review harness."""

        return self.__orchestrator.config.cch_stage_tokens

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

        if label in self.__inflight_labels:
            raise HarnessReplayError(f"harness call label {label!r} is already in flight")
        matches = [call for call in self.__state.result.calls if call.label == label]
        if len(matches) > 1:
            raise HarnessReplayError(f"checkpoint contains duplicate harness call label {label!r}")
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
                raise HarnessReplayError(
                    f"checkpoint call {label!r} does not match the harness request"
                )
            return existing.response, existing.index
        self.__inflight_labels.add(label)
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
        finally:
            self.__inflight_labels.discard(label)

    def feedback_for(self, verdict: VerdictRecord) -> str:
        """The bounded feedback text the built-in protocol shows a solver."""

        return self.__orchestrator._feedback_for_solver(verdict)

    async def verify(
        self,
        *,
        cycle: int,
        label: str,
        candidate: str,
        cap: int,
        recovery_reserve: int = 0,
        keep: int = 0,
        rationale_score: bool = False,
        branch: int | None = None,
        parent_call_index: int | None = None,
    ) -> VerdictRecord:
        """A trusted verdict on ``candidate`` from at most two labelled calls.

        The first call samples with ``cap - recovery_reserve`` tokens. If its
        verdict cannot be parsed, one greedy call with thinking disabled,
        labelled ``label + ".recovery"``, may use whatever the first left of
        ``cap``. The orchestrator builds the prompt and parses and scrubs the
        verdict, so a harness never handles privileged verifier evidence.
        """

        orchestrator, state = self.__orchestrator, self.__state
        system, user, tools, tool_choice, recovery_suffix = orchestrator._verifier_request(
            state, candidate, rationale_score=rationale_score
        )
        privileged = orchestrator._privileged_sources(state)
        first_cap = cap - recovery_reserve
        if first_cap < self.minimum_call_tokens:
            raise ValueError("verifier cap leaves less than one minimum call before recovery")
        errors: list[str] = []
        spent = 0
        for recovery in (False, True):
            attempt_cap = cap - spent if recovery else first_cap
            if attempt_cap < self.minimum_call_tokens:
                break
            attempt_label = f"{label}.recovery" if recovery else label
            completion, call_index = await self.call(
                role=Role.VERIFIER,
                cycle=cycle,
                label=attempt_label,
                messages=[
                    {
                        "role": "system",
                        "content": f"{system}\n\n{recovery_suffix}" if recovery else system,
                    },
                    {"role": "user", "content": user},
                ],
                cap=attempt_cap,
                keep=keep,
                seed=self.stable_seed(attempt_label),
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,
                force_no_thinking=recovery,
                use_sampling=not recovery,
            )
            try:
                verdict = orchestrator._parse_verdict_completion(
                    state.result.request,
                    cycle=cycle,
                    candidate=candidate,
                    completion=completion,
                    call_index=call_index,
                    privileged_sources=privileged,
                    rationale_score=rationale_score,
                )
            except Exception as exc:
                if not isinstance(getattr(exc, "status", None), TrajectoryStatus):
                    raise
                errors.append(
                    "verifier response reached its generation limit"
                    if completion.finish_reason == "length"
                    else str(exc)
                )
                spent += completion.usage.completion_tokens
                continue
            return replace(verdict, branch=branch, parent_call_index=parent_call_index)
        raise VerdictParseError(f"{label}: " + ("; ".join(errors) or "no verdict"))

    async def add_candidate(
        self,
        *,
        cycle: int,
        role: Role,
        content: str,
        reasoning: str | None,
        call_index: int,
        branch: int | None = None,
        parent_call_index: int | None = None,
    ) -> CandidateRecord:
        """Record one candidate per ``(cycle, branch)``; replays must match."""

        record = CandidateRecord(
            cycle=cycle,
            role=role,
            content=content,
            reasoning=reasoning,
            call_index=call_index,
            branch=branch,
            parent_call_index=parent_call_index,
        )
        existing = next(
            (
                candidate
                for candidate in self.__state.result.candidates
                if candidate.cycle == cycle and candidate.branch == branch
            ),
            None,
        )
        if existing is not None:
            if existing != record:
                raise HarnessReplayError(
                    f"checkpoint candidate {cycle}/{branch} does not match the harness output"
                )
            return existing
        self.__state.result.candidates.append(record)
        await self.__state.sync()
        return record

    async def add_verdict(self, record: VerdictRecord) -> VerdictRecord:
        """Record one verdict per ``(cycle, branch)``; replays must match."""

        existing = next(
            (
                verdict
                for verdict in self.__state.result.verdicts
                if verdict.cycle == record.cycle and verdict.branch == record.branch
            ),
            None,
        )
        if existing is not None:
            if existing != record:
                raise HarnessReplayError(
                    f"checkpoint verdict {record.cycle}/{record.branch} does not match "
                    "the harness output"
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
    "ATTEMPT_CONDITIONING_MODES",
    "HarnessAbort",
    "HarnessAccess",
    "HarnessReplayError",
    "HarnessRuntime",
    "HarnessSpec",
    "ProofHarness",
    "VerdictParseError",
    "module_source_path",
]
