"""Typed, JSON-serializable records shared by the solver runtime.

The module deliberately uses only the standard library.  The on-disk artifact
layer can therefore deserialize records without importing the HTTP client or a
particular validation framework.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from typing import Any

RATIONALE_MAX_CHARS = 1_200


class Condition(StrEnum):
    """Solver condition used for a trajectory."""

    DIRECT = "direct"
    GVR = "gvr"
    GVR_SUBAGENTS = "gvr_subagents"
    GVR_REFERENCE = "gvr_reference"
    VALUE_TOOL = "value_tool"
    GVR_RATIONALE_SCORE = "gvr_rationale_score"
    VALUE_TOOL_RATIONALE_SCORE = "value_tool_rationale_score"
    GVR_REFERENCE_RATIONALE_SCORE = "gvr_reference_rationale_score"


class Role(StrEnum):
    DIRECT = "direct"
    PLANNER = "planner"
    WORKER = "worker"
    REVIEWER = "reviewer"
    GENERATOR = "generator"
    VERIFIER = "verifier"
    REVISER = "reviser"
    SUBAGENT = "subagent"
    VALUE_SOLVER = "value_solver"
    VALUE_VERIFIER = "value_verifier"


class Verdict(StrEnum):
    CORRECT = "correct"
    MINOR_FIX = "minor_fix"
    CRITICAL_FLAW = "critical_flaw"


class TrajectoryStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    ACCEPTED = "accepted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONTEXT_EXHAUSTED = "context_exhausted"
    CYCLE_LIMIT = "cycle_limit"
    INVALID_USAGE = "invalid_usage"
    PROTOCOL_ERROR = "protocol_error"
    FAILED = "failed"


# Only these terminal states contain an intentional candidate eligible for the
# condition-blind external judge. Exhaustion and protocol/runtime failures are
# scheduled zeroes and remain incomplete in aggregate reports.
ADJUDICATABLE_TRAJECTORY_STATUSES = frozenset(
    {
        TrajectoryStatus.COMPLETED.value,
        TrajectoryStatus.ACCEPTED.value,
        TrajectoryStatus.CYCLE_LIMIT.value,
    }
)


@dataclass(frozen=True)
class TokenUsage:
    """Exact token usage reported by an OpenAI-compatible server.

    ``completion_tokens`` is the generated-token currency for all budgets.  On
    Qwen reasoning servers it includes hidden reasoning tokens.  The optional
    detail fields are retained for diagnostics, not added to the total again.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.reasoning_tokens is not None and self.reasoning_tokens < 0:
            raise ValueError("reasoning_tokens must be non-negative")
        if self.cached_prompt_tokens is not None and self.cached_prompt_tokens < 0:
            raise ValueError("cached_prompt_tokens must be non-negative")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")

    @classmethod
    def from_api(cls, value: Mapping[str, Any]) -> TokenUsage:
        """Parse usage and require exact prompt and completion token counts."""

        if "prompt_tokens" not in value or "completion_tokens" not in value:
            raise ValueError("response usage lacks prompt_tokens or completion_tokens")
        prompt = _required_int(value, "prompt_tokens")
        completion = _required_int(value, "completion_tokens")
        # Some compatible servers have reported inconsistent totals. Prompt
        # and completion are the auditable components, so derive total exactly
        # as the direct baseline does and retain the provider value in ``raw``.
        total = prompt + completion

        completion_details = value.get("completion_tokens_details")
        reasoning = _optional_int(value.get("reasoning_tokens"))
        if isinstance(completion_details, Mapping):
            detailed_reasoning = _optional_int(completion_details.get("reasoning_tokens"))
            if detailed_reasoning is not None:
                reasoning = detailed_reasoning
        prompt_details = value.get("prompt_tokens_details")
        cached: int | None = None
        if isinstance(prompt_details, Mapping):
            cached = _optional_int(prompt_details.get("cached_tokens"))
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reasoning_tokens=reasoning,
            cached_prompt_tokens=cached,
            raw=dict(value),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TokenUsage:
        """Restore the normalized artifact representation."""

        prompt = _required_int(value, "prompt_tokens")
        completion = _required_int(value, "completion_tokens")
        total = _optional_int(value.get("total_tokens"))
        raw = value.get("raw", {})
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion if total is None else total,
            reasoning_tokens=_optional_int(value.get("reasoning_tokens")),
            cached_prompt_tokens=_optional_int(value.get("cached_prompt_tokens")),
            raw=dict(raw) if isinstance(raw, Mapping) else {},
        )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        reasoning = _sum_optional(self.reasoning_tokens, other.reasoning_tokens)
        cached = _sum_optional(self.cached_prompt_tokens, other.cached_prompt_tokens)
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=reasoning,
            cached_prompt_tokens=cached,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ToolCall:
    """A native Chat Completions function tool call."""

    id: str
    name: str
    arguments: str
    type: str = "function"
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_api(cls, value: Mapping[str, Any]) -> ToolCall:
        function = value.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool call lacks a function object")
        call_id = value.get("id")
        name = function.get("name")
        arguments = function.get("arguments", "")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool call lacks a nonempty id")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call lacks a nonempty function name")
        if not isinstance(arguments, str):
            # A few compatible servers return an already-decoded object. Keep
            # its information while normalizing to the OpenAI wire shape.
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return cls(
            id=call_id,
            name=name,
            arguments=arguments,
            type=str(value.get("type", "function")),
            raw=dict(value),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolCall:
        if "function" in value:
            return cls.from_api(value)
        call_id = _required_string(value, "id")
        name = _required_string(value, "name")
        arguments = _required_string(value, "arguments", allow_empty=True)
        raw = value.get("raw", {})
        return cls(
            id=call_id,
            name=name,
            arguments=arguments,
            type=str(value.get("type", "function")),
            raw=dict(raw) if isinstance(raw, Mapping) else {},
        )

    def parsed_arguments(self) -> Mapping[str, Any]:
        parsed = json.loads(self.arguments)
        if not isinstance(parsed, Mapping):
            raise ValueError(f"arguments for {self.name!r} must be a JSON object")
        return parsed

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {"name": self.name, "arguments": self.arguments},
        }

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class AssistantMessage:
    """Normalized assistant message, including hidden reasoning and tools."""

    content: str | None
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_api(cls, value: Mapping[str, Any]) -> AssistantMessage:
        content = value.get("content")
        if content is not None and not isinstance(content, str):
            # Multimodal output is not expected in this text-only harness, but
            # retaining it as JSON is preferable to silently dropping it.
            content = json.dumps(content, ensure_ascii=False)
        reasoning = value.get("reasoning_content")
        if reasoning is None:
            reasoning = value.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            reasoning = json.dumps(reasoning, ensure_ascii=False)
        raw_calls = value.get("tool_calls") or ()
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raise ValueError("assistant tool_calls must be an array")
        calls = tuple(ToolCall.from_api(call) for call in raw_calls if isinstance(call, Mapping))
        if len(calls) != len(raw_calls):
            raise ValueError("assistant tool_calls contains a non-object entry")
        return cls(content=content, reasoning=reasoning, tool_calls=calls, raw=dict(value))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AssistantMessage:
        content = value.get("content")
        reasoning = value.get("reasoning")
        if content is not None and not isinstance(content, str):
            raise ValueError("assistant content must be a string or null")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("assistant reasoning must be a string or null")
        raw_calls = _required_sequence(value.get("tool_calls", []), "tool_calls")
        calls = tuple(
            ToolCall.from_dict(_required_mapping(item, "tool call")) for item in raw_calls
        )
        raw = value.get("raw", {})
        return cls(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
            raw=dict(raw) if isinstance(raw, Mapping) else {},
        )

    def to_api_dict(self) -> dict[str, Any]:
        """Return a lossless-enough message for the next Qwen tool turn."""

        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning is not None:
            # Both SGLang's qwen3 reasoning parser and vLLM accept this field on
            # replay; inbound ``reasoning`` is normalized to it as well.
            message["reasoning_content"] = self.reasoning
        if self.tool_calls:
            message["tool_calls"] = [call.to_api_dict() for call in self.tool_calls]
        return message

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ChatCompletion:
    id: str | None
    model: str | None
    message: AssistantMessage
    finish_reason: str | None
    usage: TokenUsage
    created: int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChatCompletion:
        identifier = value.get("id")
        model = value.get("model")
        finish_reason = value.get("finish_reason")
        created = value.get("created")
        raw = value.get("raw", {})
        return cls(
            id=identifier if isinstance(identifier, str) else None,
            model=model if isinstance(model, str) else None,
            message=AssistantMessage.from_dict(
                _required_mapping(value.get("message"), "completion.message")
            ),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=TokenUsage.from_dict(_required_mapping(value.get("usage"), "completion.usage")),
            created=created if isinstance(created, int) and not isinstance(created, bool) else None,
            raw=dict(raw) if isinstance(raw, Mapping) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class TrajectoryRequest:
    benchmark: str
    problem_id: str
    problem: str
    condition: Condition | None
    seed: int
    harness_id: str | None = None
    reference_proof: str | None = None
    solver_prompt: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    verifier_evidence: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrajectoryRequest:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("request.metadata must be an object")
        reference = value.get("reference_proof")
        solver_prompt = value.get("solver_prompt")
        verifier_evidence = value.get("verifier_evidence")
        if reference is not None and not isinstance(reference, str):
            raise ValueError("request.reference_proof must be a string or null")
        if solver_prompt is not None and not isinstance(solver_prompt, str):
            raise ValueError("request.solver_prompt must be a string or null")
        if verifier_evidence is not None and not isinstance(verifier_evidence, Mapping):
            raise ValueError("request.verifier_evidence must be an object or null")
        return cls(
            benchmark=_required_string(value, "benchmark"),
            problem_id=_required_string(value, "problem_id"),
            problem=_required_string(value, "problem", allow_empty=True),
            condition=(
                Condition(_required_string(value, "condition"))
                if value.get("condition") is not None
                else None
            ),
            seed=_required_int(value, "seed"),
            harness_id=(
                _required_string(value, "harness_id")
                if value.get("harness_id") is not None
                else None
            ),
            reference_proof=reference,
            solver_prompt=solver_prompt,
            metadata=dict(metadata),
            verifier_evidence=(
                dict(verifier_evidence) if verifier_evidence is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        value = _jsonable(asdict(self))
        if self.condition is None:
            value.pop("condition", None)
        if self.harness_id is None:
            value.pop("harness_id", None)
        if self.verifier_evidence is None:
            value.pop("verifier_evidence", None)
        return value

    @property
    def method_id(self) -> str:
        if self.harness_id:
            return self.harness_id
        if self.condition is not None:
            return self.condition.value
        raise ValueError("trajectory request has neither harness_id nor condition")


@dataclass(frozen=True)
class CallRecord:
    index: int
    request_id: str
    role: Role
    cycle: int
    label: str
    seed: int
    max_tokens: int
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    response: ChatCompletion | None
    usage: TokenUsage | None
    error: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CallRecord:
        response_value = value.get("response")
        usage_value = value.get("usage")
        error = value.get("error")
        messages = _required_sequence(value.get("messages"), "call.messages")
        tools = _required_sequence(value.get("tools", []), "call.tools")
        return cls(
            index=_required_int(value, "index"),
            request_id=_required_string(value, "request_id"),
            role=Role(_required_string(value, "role")),
            cycle=_required_int(value, "cycle"),
            label=_required_string(value, "label"),
            seed=_required_int(value, "seed"),
            max_tokens=_required_int(value, "max_tokens"),
            messages=tuple(_required_mapping(item, "call message") for item in messages),
            tools=tuple(_required_mapping(item, "call tool") for item in tools),
            response=(
                ChatCompletion.from_dict(_required_mapping(response_value, "call.response"))
                if response_value is not None
                else None
            ),
            usage=(
                TokenUsage.from_dict(_required_mapping(usage_value, "call.usage"))
                if usage_value is not None
                else None
            ),
            error=error if isinstance(error, str) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class CandidateRecord:
    cycle: int
    role: Role
    content: str
    reasoning: str | None
    call_index: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateRecord:
        reasoning = value.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("candidate.reasoning must be a string or null")
        return cls(
            cycle=_required_int(value, "cycle"),
            role=Role(_required_string(value, "role")),
            content=_required_string(value, "content", allow_empty=True),
            reasoning=reasoning,
            call_index=_required_int(value, "call_index"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class VerdictRecord:
    cycle: int
    verdict: Verdict
    critique: str
    fault_category: str
    candidate_excerpt: str
    call_index: int
    success_probability: float | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        probability = self.success_probability
        if probability is not None and (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(
                "verdict success_probability must be a finite number between zero and one"
            )
        if (probability is None) != (not self.rationale):
            raise ValueError(
                "verdict rationale and success_probability must either both be present "
                "or both be absent"
            )
        if len(self.rationale) > RATIONALE_MAX_CHARS:
            raise ValueError("verdict rationale exceeds its character limit")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VerdictRecord:
        probability = value.get("success_probability")
        if probability is not None and (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(
                "verdict success_probability must be a finite number between zero and one"
            )
        return cls(
            cycle=_required_int(value, "cycle"),
            verdict=Verdict(_required_string(value, "verdict")),
            critique=_required_string(value, "critique", allow_empty=True),
            fault_category=_required_string(value, "fault_category", allow_empty=True),
            candidate_excerpt=_required_string(value, "candidate_excerpt", allow_empty=True),
            call_index=_required_int(value, "call_index"),
            success_probability=(
                float(probability) if probability is not None else None
            ),
            rationale=_required_string(value, "rationale", allow_empty=True)
            if "rationale" in value
            else "",
        )

    def to_dict(self) -> dict[str, Any]:
        value = _jsonable(asdict(self))
        if self.success_probability is None:
            value.pop("success_probability", None)
            value.pop("rationale", None)
        return value


@dataclass(frozen=True)
class SubagentRecord:
    cycle: int
    task_index: int
    task: str
    context_excerpt: str
    seed: int
    output: str
    reasoning: str | None
    call_index: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SubagentRecord:
        reasoning = value.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("subagent.reasoning must be a string or null")
        return cls(
            cycle=_required_int(value, "cycle"),
            task_index=_required_int(value, "task_index"),
            task=_required_string(value, "task"),
            context_excerpt=_required_string(value, "context_excerpt", allow_empty=True),
            seed=_required_int(value, "seed"),
            output=_required_string(value, "output", allow_empty=True),
            reasoning=reasoning,
            call_index=_required_int(value, "call_index"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ValueEstimateRecord:
    query_index: int
    probability: float
    solver_call_index: int
    verifier_call_index: int
    tool_call_id: str
    trace_sha256: str
    rationale: str = ""

    def __post_init__(self) -> None:
        if len(self.rationale) > RATIONALE_MAX_CHARS:
            raise ValueError("value-estimate rationale exceeds its character limit")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ValueEstimateRecord:
        expected_fields = {
            "query_index",
            "probability",
            "solver_call_index",
            "verifier_call_index",
            "tool_call_id",
            "trace_sha256",
            "rationale",
        }
        unexpected_fields = set(value) - expected_fields
        if unexpected_fields:
            names = ", ".join(sorted(str(field) for field in unexpected_fields))
            raise ValueError(f"value estimate contains unexpected fields: {names}")
        query_index = _required_int(value, "query_index")
        solver_call_index = _required_int(value, "solver_call_index")
        verifier_call_index = _required_int(value, "verifier_call_index")
        if min(query_index, solver_call_index, verifier_call_index) < 0:
            raise ValueError("value-estimate indices must be non-negative")
        probability = value.get("probability")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("probability must be a finite number between zero and one")
        trace_sha256 = _required_string(value, "trace_sha256")
        if len(trace_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in trace_sha256
        ):
            raise ValueError("trace_sha256 must be a lowercase SHA-256 digest")
        return cls(
            query_index=query_index,
            probability=float(probability),
            solver_call_index=solver_call_index,
            verifier_call_index=verifier_call_index,
            tool_call_id=_required_string(value, "tool_call_id"),
            trace_sha256=trace_sha256,
            rationale=_required_string(value, "rationale", allow_empty=True)
            if "rationale" in value
            else "",
        )

    def to_dict(self) -> dict[str, Any]:
        value = _jsonable(asdict(self))
        if not self.rationale:
            value.pop("rationale", None)
        return value


@dataclass(frozen=True)
class TransitionRecord:
    cycle: int
    source: str
    action: str
    target: str
    detail: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TransitionRecord:
        return cls(
            cycle=_required_int(value, "cycle"),
            source=_required_string(value, "source"),
            action=_required_string(value, "action"),
            target=_required_string(value, "target"),
            detail=_required_string(value, "detail", allow_empty=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class TrajectoryResult:
    request: TrajectoryRequest
    status: TrajectoryStatus = TrajectoryStatus.FAILED
    final_output: str | None = None
    calls: list[CallRecord] = field(default_factory=list)
    candidates: list[CandidateRecord] = field(default_factory=list)
    verdicts: list[VerdictRecord] = field(default_factory=list)
    subagents: list[SubagentRecord] = field(default_factory=list)
    value_estimates: list[ValueEstimateRecord] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    budget: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrajectoryResult:
        calls = _required_sequence(value.get("calls", []), "calls")
        candidates = _required_sequence(value.get("candidates", []), "candidates")
        verdicts = _required_sequence(value.get("verdicts", []), "verdicts")
        subagents = _required_sequence(value.get("subagents", []), "subagents")
        value_estimates = _required_sequence(
            value.get("value_estimates", []), "value_estimates"
        )
        transitions = _required_sequence(value.get("transitions", []), "transitions")
        budget = value.get("budget", {})
        if not isinstance(budget, Mapping):
            raise ValueError("budget must be an object")
        error = value.get("error")
        return cls(
            request=TrajectoryRequest.from_dict(_required_mapping(value.get("request"), "request")),
            status=TrajectoryStatus(_required_string(value, "status")),
            final_output=(
                value.get("final_output") if isinstance(value.get("final_output"), str) else None
            ),
            calls=[CallRecord.from_dict(_required_mapping(item, "call")) for item in calls],
            candidates=[
                CandidateRecord.from_dict(_required_mapping(item, "candidate"))
                for item in candidates
            ],
            verdicts=[
                VerdictRecord.from_dict(_required_mapping(item, "verdict")) for item in verdicts
            ],
            subagents=[
                SubagentRecord.from_dict(_required_mapping(item, "subagent")) for item in subagents
            ],
            value_estimates=[
                ValueEstimateRecord.from_dict(_required_mapping(item, "value estimate"))
                for item in value_estimates
            ],
            transitions=[
                TransitionRecord.from_dict(_required_mapping(item, "transition"))
                for item in transitions
            ],
            usage=TokenUsage.from_dict(_required_mapping(value.get("usage", {}), "usage")),
            budget=dict(budget),
            error=error if isinstance(error, str) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        value = _jsonable(asdict(self))
        if self.request.verifier_evidence is None:
            value["request"].pop("verifier_evidence", None)
        for verdict in value.get("verdicts", []):
            if isinstance(verdict, dict) and verdict.get("success_probability") is None:
                verdict.pop("success_probability", None)
                verdict.pop("rationale", None)
        for estimate in value.get("value_estimates", []):
            if isinstance(estimate, dict) and not estimate.get("rationale"):
                estimate.pop("rationale", None)
        # Transport ``raw`` objects duplicate the complete assistant message,
        # tool arguments, and usage for every call. Keep those on the in-memory
        # client objects for diagnostics, but make trajectory artifacts compact
        # and canonical.
        usage = value.get("usage")
        if isinstance(usage, dict):
            usage.pop("raw", None)
        for call in value.get("calls", []):
            call_usage = call.get("usage")
            if isinstance(call_usage, dict):
                call_usage.pop("raw", None)
            response = call.get("response")
            if not isinstance(response, dict):
                continue
            response.pop("raw", None)
            response_usage = response.get("usage")
            if isinstance(response_usage, dict):
                response_usage.pop("raw", None)
            message = response.get("message")
            if isinstance(message, dict):
                message.pop("raw", None)
                for tool_call in message.get("tool_calls", []):
                    if isinstance(tool_call, dict):
                        tool_call.pop("raw", None)
        budget = value.get("budget")
        if isinstance(budget, dict):
            for charge in budget.get("charges", []):
                if isinstance(charge, dict) and isinstance(charge.get("usage"), dict):
                    charge["usage"].pop("raw", None)
        return value


def _required_int(value: Mapping[str, Any], key: str) -> int:
    parsed = _optional_int(value.get(key))
    if parsed is None:
        raise ValueError(f"usage.{key} must be an integer")
    return parsed


def _required_string(value: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    item = value.get(key)
    if not isinstance(item, str) or (not allow_empty and not item):
        suffix = "" if allow_empty else " nonempty"
        raise ValueError(f"{key} must be a{suffix} string")
    return item


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _required_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _optional_int(value: Any) -> int | None:
    # bool is an int subclass but never a meaningful token count.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "AssistantMessage",
    "CallRecord",
    "CandidateRecord",
    "ChatCompletion",
    "Condition",
    "Role",
    "SubagentRecord",
    "TokenUsage",
    "ToolCall",
    "TrajectoryRequest",
    "TrajectoryResult",
    "TrajectoryStatus",
    "TransitionRecord",
    "ValueEstimateRecord",
    "Verdict",
    "VerdictRecord",
]
