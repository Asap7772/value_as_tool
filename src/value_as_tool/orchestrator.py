"""Direct and Aletheia-style Generator/Verifier/Reviser trajectories."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .budget import BudgetAccountingError, BudgetExhausted, TokenBudget
from .client import ChatClient, ChatClientError
from .schemas import (
    CallRecord,
    CandidateRecord,
    ChatCompletion,
    Condition,
    Role,
    SubagentRecord,
    TokenUsage,
    TrajectoryRequest,
    TrajectoryResult,
    TrajectoryStatus,
    TransitionRecord,
    Verdict,
    VerdictRecord,
)

GENERATOR_SYSTEM_PROMPT = """You are the Generator in a mathematical solution system.
Develop a complete, rigorous response to the task. Check every substantive
claim, address edge cases, and obey the user's requested answer format. Do not
discuss the multi-agent protocol in the proposed solution."""

REVISER_SYSTEM_PROMPT = """You are the Reviser in a mathematical solution system.
Repair the candidate using the verifier's bounded diagnosis. Return a complete,
self-contained replacement that obeys the original requested answer format,
not a patch or a discussion of edits. Independently check the corrected work."""

VERIFIER_SYSTEM_PROMPT = """You are the Verifier in a mathematical proof system.
Assess whether the candidate is a complete rigorous solution of the stated
problem. Use exactly one submit_verdict tool call. Choose correct only if no
substantive gap remains; minor_fix for a locally repairable issue; and
critical_flaw when a new approach or major reconstruction is needed. Keep the
diagnosis specific and concise."""

REFERENCE_VERIFIER_SYSTEM_PROMPT = """You are the Verifier in a mathematical proof
system and may consult a reference proof. Assess the candidate itself, not its
wording similarity to the reference. Use exactly one submit_verdict tool call.
Do not quote or reproduce the reference proof in any field. Cite only passages
that occur in the candidate. Choose correct, minor_fix, or critical_flaw."""

SUBAGENT_SYSTEM_PROMPT = """You are a fresh-context mathematical research
subagent. Work only on the focused task. Return concise, rigorous findings for
the parent solver; do not attempt to invoke tools or delegate further."""


# SGLang deserializes this with ``dill.loads`` and applies it as a custom logit
# processor, forcing ``</think>`` once a request has spent
# ``custom_params["thinking_budget"]`` tokens inside an open thinking block.
#
# We cannot use SGLang's own ``Qwen3ThinkingBudgetLogitProcessor``: it hardcodes
# the Qwen3 think token ids (151667/151668), but Qwen3.5 renumbered them to
# 248068/248069.  With the wrong ids its ``_open_thinking_start`` never finds an
# open block, hits ``continue``, and the processor is a silent no-op -- thinking
# then runs to the full per-call cap exactly as if no budget were set.  So we
# ship a subclass that overrides only the three token ids.
#
# dill serializes a class defined outside an importable module *by value*, so
# this payload carries the attributes while still naming SGLang's base class by
# reference (the server supplies ``__call__``).  ``pickletools.dis`` shows only
# ``_create_type`` over three ints plus a ``setattr`` for ``__qualname__``: no
# code objects.  Regenerate with:
#
#   python -c "
#   import dill, json
#   from sglang.srt.sampling.custom_logit_processor import ThinkingBudgetLogitProcessor
#   print(json.dumps({'callable': dill.dumps(type(
#       'Qwen35ThinkingBudgetLogitProcessor', (ThinkingBudgetLogitProcessor,), {
#           'THINKING_START_TOKEN_ID': 248068,
#           'THINKING_END_TOKEN_ID': 248069,
#           'NEW_LINE_TOKEN_ID': 198,
#           '__doc__': 'Thinking budget for Qwen3.5 think token ids.'})).hex()}))"
#
# The server only honors it when started with --enable-custom-logit-processor.
QWEN3_THINKING_BUDGET_PROCESSOR = (
    '{"callable": "80049590010000000000008c0a64696c6c2e5f64696c6c948c0c5f637265'
    "6174655f74797065949394288c03616263948c074142434d6574619493948c225177656e33"
    "355468696e6b696e674275646765744c6f67697450726f636573736f72948c2a73676c616e"
    "672e7372742e73616d706c696e672e637573746f6d5f6c6f6769745f70726f636573736f72"
    "948c1c5468696e6b696e674275646765744c6f67697450726f636573736f7294939485947d"
    "94288c175448494e4b494e475f53544152545f544f4b454e5f4944944a04c903008c155448"
    "494e4b494e475f454e445f544f4b454e5f4944944a05c903008c114e45575f4c494e455f54"
    "4f4b454e5f4944944bc68c075f5f646f635f5f948c2c5468696e6b696e6720627564676574"
    "20666f72205177656e332e35207468696e6b20746f6b656e206964732e948c0a5f5f6d6f64"
    "756c655f5f9468038c135f5f61627374726163746d6574686f64735f5f9428919475749452"
    "948c086275696c74696e73948c07736574617474729493946815"
    '8c0c5f5f7175616c6e616d655f5f946806879452302e"}'
)

# Qwen3.5 think-token ids the pinned payload above encodes.  The cross-check in
# tests/test_core_runtime.py asserts these still match the solver tokenizer, so
# a model bump that renumbers them fails loudly instead of silently disabling
# the budget again.
QWEN35_THINK_TOKEN_IDS = (248068, 248069)


SUBMIT_VERDICT_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "Submit the verifier's routing decision and bounded diagnosis.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": ["correct", "minor_fix", "critical_flaw"],
                },
                "critique": {"type": "string", "maxLength": 600},
                "fault_category": {"type": "string", "maxLength": 80},
                "candidate_excerpt": {"type": "string", "maxLength": 240},
            },
            "required": [
                "outcome",
                "critique",
                "fault_category",
                "candidate_excerpt",
            ],
        },
    },
}

SPAWN_SUBAGENTS_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "spawn_subagents",
        "description": (
            "Run up to three independent fresh-context investigations in parallel. "
            "Use one focused task per child and then synthesize their returned findings."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "task": {"type": "string"},
                            "context_excerpt": {"type": "string"},
                        },
                        "required": ["task"],
                    },
                }
            },
            "required": ["tasks"],
        },
    },
}


Checkpoint = Callable[[TrajectoryResult], Awaitable[None] | None]


class RequestLifecycle(Protocol):
    """Optional crash-safety hooks implemented by the artifact layer."""

    def begin_request(
        self,
        request_id: str,
        *,
        role: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> object | Awaitable[object]: ...

    def complete_request(
        self,
        request_id: str,
        *,
        state: str,
        payload: Mapping[str, Any],
        usage: Mapping[str, Any],
    ) -> object | Awaitable[object]: ...


class TokenCounter(Protocol):
    """Tokenizer adapter used to enforce the per-request context window."""

    def count_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> int: ...


CountMessages = Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]] | None], int]


@dataclass(frozen=True)
class OrchestratorConfig:
    model: str = "Qwen/Qwen3.5-9B"
    total_generated_tokens: int = 229_376
    context_tokens: int = 262_144
    context_headroom_tokens: int = 1_024
    initial_generator_cap: int = 98_304
    verifier_cap: int = 32_768
    correction_pool: int = 98_304
    minimum_call_tokens: int = 1_024
    max_cycles: int = 3
    subagent_cap: int = 16_384
    max_subagents: int = 3
    final_candidate_reserve_tokens: int = 32_768
    max_tool_rounds_per_candidate: int = 4
    critique_max_chars: int = 600
    candidate_excerpt_max_chars: int = 240
    subagent_context_max_chars: int = 4_000
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    extra_body: Mapping[str, Any] = field(
        default_factory=lambda: {
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )
    thinking_content_reserve_tokens: int = 0
    thinking_budget_processor: str | None = None
    direct_system_prompt: str | None = None
    generator_system_prompt: str = GENERATOR_SYSTEM_PROMPT
    reviser_system_prompt: str = REVISER_SYSTEM_PROMPT
    verifier_system_prompt: str = VERIFIER_SYSTEM_PROMPT
    reference_verifier_system_prompt: str = REFERENCE_VERIFIER_SYSTEM_PROMPT
    subagent_system_prompt: str = SUBAGENT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        positive = {
            "total_generated_tokens": self.total_generated_tokens,
            "context_tokens": self.context_tokens,
            "context_headroom_tokens": self.context_headroom_tokens,
            "initial_generator_cap": self.initial_generator_cap,
            "verifier_cap": self.verifier_cap,
            "correction_pool": self.correction_pool,
            "minimum_call_tokens": self.minimum_call_tokens,
            "max_cycles": self.max_cycles,
            "subagent_cap": self.subagent_cap,
            "max_subagents": self.max_subagents,
            "final_candidate_reserve_tokens": self.final_candidate_reserve_tokens,
            "max_tool_rounds_per_candidate": self.max_tool_rounds_per_candidate,
            "critique_max_chars": self.critique_max_chars,
            "candidate_excerpt_max_chars": self.candidate_excerpt_max_chars,
            "subagent_context_max_chars": self.subagent_context_max_chars,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        reserved = self.initial_generator_cap + self.verifier_cap + self.correction_pool
        if reserved > self.total_generated_tokens:
            raise ValueError(
                "initial_generator_cap + verifier_cap + correction_pool exceeds "
                "total_generated_tokens"
            )
        if self.context_headroom_tokens >= self.context_tokens:
            raise ValueError("context_headroom_tokens must be smaller than context_tokens")
        if self.total_generated_tokens + self.context_headroom_tokens > self.context_tokens:
            raise ValueError(
                "total_generated_tokens plus context headroom exceeds the context window"
            )
        if self.max_cycles > 3:
            raise ValueError("the Aletheia-style protocol permits at most three cycles")
        if self.max_subagents > 3:
            raise ValueError("the protocol permits at most three subagents per trajectory")
        required_correction_pool = (self.max_cycles - 1) * (
            self.verifier_cap + self.minimum_call_tokens
        )
        if self.correction_pool < required_correction_pool:
            raise ValueError(
                "correction_pool cannot fit the minimum candidate and verifier calls "
                "for every configured correction cycle"
            )
        capped_roles = {
            "initial_generator_cap": self.initial_generator_cap,
            "verifier_cap": self.verifier_cap,
            "subagent_cap": self.subagent_cap,
        }
        for name, cap in capped_roles.items():
            if self.minimum_call_tokens > cap:
                raise ValueError(f"minimum_call_tokens exceeds {name}")
        if self.thinking_content_reserve_tokens < 0:
            raise ValueError("thinking_content_reserve_tokens must be non-negative")
        if self.thinking_content_reserve_tokens:
            if self.thinking_budget_processor is None:
                raise ValueError(
                    "thinking_content_reserve_tokens requires thinking_budget_processor"
                )
            # Every capped role must keep a usable thinking block once the
            # reserve is carved out, or the smallest calls would be forced to
            # answer with no reasoning at all.
            for name, cap in capped_roles.items():
                if self.thinking_content_reserve_tokens >= cap - self.minimum_call_tokens:
                    raise ValueError(
                        f"thinking_content_reserve_tokens leaves no thinking room in {name}"
                    )

    def thinking_budget_for(self, cap: int) -> int | None:
        """Thinking tokens allowed so a ``cap``-token call can still answer.

        Qwen3.5 otherwise spends the whole per-call cap inside ``<think>`` and
        is truncated at ``finish_reason=length`` with empty content, which the
        protocol scores as a failure rather than a wrong answer.
        """

        if not self.thinking_content_reserve_tokens or self.thinking_budget_processor is None:
            return None
        return max(self.minimum_call_tokens, cap - self.thinking_content_reserve_tokens)


class _RunAbort(RuntimeError):
    def __init__(self, status: TrajectoryStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class NonResumableTrajectoryError(RuntimeError):
    """A checkpoint cannot be resumed without duplicating or losing usage."""


@dataclass
class _RunState:
    result: TrajectoryResult
    budget: TokenBudget
    checkpoint: Checkpoint | None
    next_call_index: int = 0
    subagent_count: int = 0
    checkpoint_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def refresh(self) -> None:
        self.result.usage = self.budget.total_usage()
        self.result.budget = self.budget.snapshot()

    async def sync(self) -> None:
        self.refresh()
        if self.checkpoint is None:
            return
        # A snapshot prevents later in-memory mutations from changing an
        # artifact store's view while an async write is in flight.
        snapshot = copy.deepcopy(self.result)
        async with self.checkpoint_lock:
            outcome = self.checkpoint(snapshot)
            if inspect.isawaitable(outcome):
                await outcome

    async def transition(
        self,
        cycle: int,
        source: str,
        action: str,
        target: str,
        detail: str = "",
    ) -> None:
        self.result.transitions.append(
            TransitionRecord(
                cycle=cycle,
                source=source,
                action=action,
                target=target,
                detail=detail,
            )
        )
        await self.sync()


class AletheiaOrchestrator:
    """Run direct or bounded Generator/Verifier/Reviser trajectories."""

    def __init__(
        self,
        client: ChatClient,
        config: OrchestratorConfig | None = None,
        *,
        checkpoint: Checkpoint | None = None,
        request_lifecycle: RequestLifecycle | None = None,
        token_counter: TokenCounter | None = None,
        count_messages: CountMessages | None = None,
    ) -> None:
        if token_counter is not None and count_messages is not None:
            raise ValueError("provide token_counter or count_messages, not both")
        self.client = client
        self.config = config or OrchestratorConfig()
        self.checkpoint = checkpoint
        self.request_lifecycle = request_lifecycle
        self.count_messages = (
            token_counter.count_messages if token_counter is not None else count_messages
        )

    async def run(
        self,
        request: TrajectoryRequest,
        *,
        resume: TrajectoryResult | Mapping[str, Any] | None = None,
    ) -> TrajectoryResult:
        try:
            condition = Condition(request.condition)
        except ValueError as exc:
            raise ValueError(f"unknown condition: {request.condition!r}") from exc
        if not isinstance(request.condition, Condition):
            request = replace(request, condition=condition)
        if resume is None:
            result = TrajectoryResult(request=request, status=TrajectoryStatus.RUNNING)
            budget = TokenBudget(self.config.total_generated_tokens)
        else:
            result = self._restore_result(request, resume)
            budget = self._restore_budget(result)
        state = _RunState(
            result=result,
            budget=budget,
            checkpoint=self.checkpoint,
            next_call_index=(max((call.index for call in result.calls), default=-1) + 1),
            subagent_count=sum(call.role is Role.SUBAGENT for call in result.calls),
        )
        if resume is None:
            await state.transition(0, "scheduled", "start", condition.value)
        elif result.status is not TrajectoryStatus.RUNNING:
            return result

        failed_calls = [call for call in result.calls if call.error]
        if failed_calls:
            result.status = TrajectoryStatus.PROTOCOL_ERROR
            result.error = f"completed call failed: {failed_calls[0].error}"
            await state.sync()
            return result

        if condition is Condition.GVR_REFERENCE and not request.reference_proof:
            result.status = TrajectoryStatus.FAILED
            result.error = "gvr_reference requires a nonempty reference_proof"
            await state.sync()
            return result

        try:
            if condition is Condition.DIRECT:
                await self._run_direct(state)
            else:
                await self._run_gvr(state)
        except _RunAbort as exc:
            result.status = exc.status
            result.error = str(exc)
            # Preserve the most recent nonempty candidate on every terminal
            # failure, including verifier/protocol failures.
            if result.final_output is None and result.candidates:
                result.final_output = result.candidates[-1].content
            await state.sync()
        except BudgetExhausted as exc:
            result.status = TrajectoryStatus.BUDGET_EXHAUSTED
            result.error = str(exc)
            if result.candidates:
                result.final_output = result.candidates[-1].content
            await state.sync()
        except asyncio.CancelledError:
            # _call marks an in-flight permit unknown before cancellation gets
            # here. Do not convert task cancellation into an ordinary result.
            await state.sync()
            raise
        except NonResumableTrajectoryError:
            raise
        except Exception as exc:  # keep one malformed item from killing a shard
            result.status = TrajectoryStatus.FAILED
            result.error = f"{type(exc).__name__}: {exc}"
            if result.candidates:
                result.final_output = result.candidates[-1].content
            await state.sync()
        return result

    def _restore_result(
        self,
        request: TrajectoryRequest,
        resume: TrajectoryResult | Mapping[str, Any],
    ) -> TrajectoryResult:
        if isinstance(resume, TrajectoryResult):
            result = copy.deepcopy(resume)
        else:
            active_many = resume.get("in_flight_requests")
            active_one = resume.get("in_flight_request")
            if (isinstance(active_many, Mapping) and active_many) or isinstance(
                active_one, Mapping
            ):
                raise NonResumableTrajectoryError(
                    "checkpoint contains an in-flight request with unknown usage"
                )
            payload = resume.get("payload") if "payload" in resume else resume
            if not isinstance(payload, Mapping):
                raise NonResumableTrajectoryError("checkpoint payload is not an object")
            try:
                result = TrajectoryResult.from_dict(payload)
            except (TypeError, ValueError) as exc:
                raise NonResumableTrajectoryError(f"invalid checkpoint payload: {exc}") from exc

        if result.request.to_dict() != request.to_dict():
            raise NonResumableTrajectoryError(
                "checkpoint trajectory identity does not match the requested trajectory"
            )
        unknown = result.budget.get("unknown_usage") if result.budget else None
        unknown_upper = result.budget.get("unknown_usage_upper_bound") if result.budget else 0
        if unknown or (isinstance(unknown_upper, int) and unknown_upper > 0):
            raise NonResumableTrajectoryError("checkpoint contains unknown provider usage")
        return result

    def _restore_budget(self, result: TrajectoryResult) -> TokenBudget:
        configured = result.budget.get("max_generated_tokens") if result.budget else None
        if configured is not None and configured != self.config.total_generated_tokens:
            raise NonResumableTrajectoryError(
                "checkpoint generated-token budget differs from current configuration"
            )
        indices = [call.index for call in result.calls]
        if indices != list(range(len(indices))):
            raise NonResumableTrajectoryError(
                "checkpoint call indices must be unique, ordered, and contiguous"
            )
        request_ids = [call.request_id for call in result.calls]
        if len(request_ids) != len(set(request_ids)):
            raise NonResumableTrajectoryError("checkpoint contains duplicate request IDs")

        budget = TokenBudget(self.config.total_generated_tokens)
        for call in result.calls:
            if call.usage is None:
                raise NonResumableTrajectoryError(f"call {call.request_id!r} has unknown usage")
            if call.response is not None and call.response.usage != call.usage:
                raise NonResumableTrajectoryError(
                    f"call {call.request_id!r} has inconsistent response usage"
                )
            try:
                permit = budget.reserve(call.max_tokens, label=call.label)
                if permit.max_tokens != call.max_tokens:
                    raise NonResumableTrajectoryError(
                        f"call {call.request_id!r} could not have fit in the shared budget"
                    )
                budget.settle(permit, call.usage)
            except (BudgetAccountingError, BudgetExhausted) as exc:
                raise NonResumableTrajectoryError(
                    f"invalid budget history at call {call.request_id!r}: {exc}"
                ) from exc
        restored = budget.total_usage()
        if (
            restored.prompt_tokens != result.usage.prompt_tokens
            or restored.completion_tokens != result.usage.completion_tokens
            or restored.total_tokens != result.usage.total_tokens
        ):
            raise NonResumableTrajectoryError(
                "checkpoint aggregate usage does not equal its call records"
            )
        return budget

    async def _run_direct(self, state: _RunState) -> None:
        request = state.result.request
        if state.result.candidates:
            if len(state.result.candidates) != 1:
                raise NonResumableTrajectoryError(
                    "direct checkpoint contains more than one candidate"
                )
            state.result.final_output = state.result.candidates[0].content
            producing_call = next(
                (
                    call
                    for call in state.result.calls
                    if call.index == state.result.candidates[0].call_index
                ),
                None,
            )
            if (
                producing_call is not None
                and producing_call.response is not None
                and producing_call.response.finish_reason == "length"
            ):
                raise _RunAbort(
                    TrajectoryStatus.BUDGET_EXHAUSTED,
                    "direct completion reached its generation limit",
                )
            state.result.status = TrajectoryStatus.COMPLETED
            await state.transition(0, "direct", "resume", "final_output")
            return
        prior_calls = [call for call in state.result.calls if call.role is Role.DIRECT]
        if prior_calls:
            if len(prior_calls) != 1 or len(state.result.calls) != 1:
                raise NonResumableTrajectoryError(
                    "direct checkpoint has an ambiguous completed-call history"
                )
            recovered = self._candidate_from_call(prior_calls[0], Role.DIRECT, 0)
            state.result.candidates.append(recovered)
            state.result.final_output = recovered.content
            if (
                prior_calls[0].response is not None
                and prior_calls[0].response.finish_reason == "length"
            ):
                raise _RunAbort(
                    TrajectoryStatus.BUDGET_EXHAUSTED,
                    "direct completion reached its generation limit",
                )
            state.result.status = TrajectoryStatus.COMPLETED
            await state.transition(0, "direct", "recover", "final_output")
            return

        messages: list[Mapping[str, Any]] = []
        if self.config.direct_system_prompt:
            messages.append({"role": "system", "content": self.config.direct_system_prompt})
        messages.append({"role": "user", "content": request.solver_prompt or request.problem})
        completion, call_index = await self._call(
            state,
            role=Role.DIRECT,
            cycle=0,
            label="direct",
            messages=messages,
            cap=self.config.total_generated_tokens,
            keep=0,
            seed=request.seed,
        )
        output = (completion.message.content or "").strip()
        if not output:
            raise _RunAbort(
                TrajectoryStatus.PROTOCOL_ERROR,
                "direct completion returned no final content",
            )
        state.result.candidates.append(
            CandidateRecord(
                cycle=0,
                role=Role.DIRECT,
                content=output,
                reasoning=completion.message.reasoning,
                call_index=call_index,
            )
        )
        state.result.final_output = output
        if completion.finish_reason == "length":
            raise _RunAbort(
                TrajectoryStatus.BUDGET_EXHAUSTED,
                "direct completion reached its generation limit",
            )
        state.result.status = TrajectoryStatus.COMPLETED
        await state.transition(0, "direct", "return", "final_output")

    async def _run_gvr(self, state: _RunState) -> None:
        request = state.result.request
        allow_subagents = request.condition is Condition.GVR_SUBAGENTS
        self._validate_gvr_boundaries(state.result)
        if state.result.candidates:
            candidate_record = state.result.candidates[-1]
        else:
            candidate, reasoning, call_index = await self._produce_candidate(
                state,
                cycle=1,
                role=Role.GENERATOR,
                prior_candidate=None,
                feedback=None,
                phase_cap=self.config.initial_generator_cap,
                keep=self.config.verifier_cap + self.config.correction_pool,
                allow_subagents=allow_subagents,
            )
            recovered = CandidateRecord(1, Role.GENERATOR, candidate, reasoning, call_index)
            state.result.candidates.append(recovered)
            state.result.final_output = recovered.content
            await state.transition(1, "generator", "candidate", "verifier")
            candidate_record = recovered

        while True:
            cycle = candidate_record.cycle
            candidate = candidate_record.content
            existing = next(
                (item for item in state.result.verdicts if item.cycle == cycle),
                None,
            )
            if existing is None:
                verdict = self._recover_verdict(state, cycle=cycle, candidate=candidate)
                if verdict is None:
                    verifier_keep = (
                        self.config.correction_pool
                        if cycle == 1
                        else self._future_correction_reserve(cycle)
                    )
                    verdict = await self._verify(
                        state,
                        cycle=cycle,
                        candidate=candidate,
                        keep=verifier_keep,
                    )
                state.result.verdicts.append(verdict)
            else:
                verdict = existing
            if verdict.verdict is Verdict.CORRECT:
                state.result.status = TrajectoryStatus.ACCEPTED
                state.result.final_output = candidate
                await state.transition(cycle, "verifier", "correct", "final_output")
                return

            if cycle >= self.config.max_cycles:
                state.result.status = TrajectoryStatus.CYCLE_LIMIT
                state.result.final_output = candidate
                await state.transition(
                    cycle,
                    "verifier",
                    verdict.verdict.value,
                    "cycle_limit",
                )
                return

            next_role = Role.REVISER if verdict.verdict is Verdict.MINOR_FIX else Role.GENERATOR
            next_candidate = next(
                (item for item in state.result.candidates if item.cycle == cycle + 1),
                None,
            )
            if next_candidate is None:
                await self._transition_once(
                    state,
                    cycle,
                    "verifier",
                    verdict.verdict.value,
                    next_role.value,
                )
                correction_cycle = cycle + 1
                prior_correction_spend = sum(
                    call.usage.completion_tokens
                    for call in state.result.calls
                    if 2 <= call.cycle < correction_cycle
                    and call.usage is not None
                )
                reserved_after_candidate = (
                    self.config.verifier_cap
                    + self._future_correction_reserve(correction_cycle)
                )
                correction_cap = (
                    self.config.correction_pool
                    - prior_correction_spend
                    - reserved_after_candidate
                )
                if correction_cap < self.config.minimum_call_tokens:
                    raise BudgetExhausted(
                        "the cumulative correction pool cannot fit another candidate"
                    )
                feedback = self._feedback_for_solver(verdict)
                candidate, reasoning, call_index = await self._produce_candidate(
                    state,
                    cycle=correction_cycle,
                    role=next_role,
                    prior_candidate=candidate,
                    feedback=feedback,
                    phase_cap=correction_cap,
                    keep=reserved_after_candidate,
                    allow_subagents=allow_subagents,
                )
                next_candidate = CandidateRecord(
                    correction_cycle,
                    next_role,
                    candidate,
                    reasoning,
                    call_index,
                )
                state.result.candidates.append(next_candidate)
                state.result.final_output = next_candidate.content
                await state.transition(
                    cycle + 1,
                    next_role.value,
                    "candidate",
                    "verifier",
                )
            elif next_candidate.role is not next_role:
                raise NonResumableTrajectoryError(
                    f"candidate {cycle + 1} has role {next_candidate.role.value}, "
                    f"expected {next_role.value}"
                )
            candidate_record = next_candidate

    def _future_correction_reserve(self, completed_cycle: int) -> int:
        remaining_cycles = max(0, self.config.max_cycles - completed_cycle)
        return remaining_cycles * (
            self.config.minimum_call_tokens + self.config.verifier_cap
        )

    async def _transition_once(
        self,
        state: _RunState,
        cycle: int,
        source: str,
        action: str,
        target: str,
        detail: str = "",
    ) -> None:
        proposed = TransitionRecord(cycle, source, action, target, detail)
        if proposed not in state.result.transitions:
            await state.transition(cycle, source, action, target, detail)

    def _validate_gvr_boundaries(self, result: TrajectoryResult) -> None:
        candidate_cycles = [candidate.cycle for candidate in result.candidates]
        verdict_cycles = [verdict.cycle for verdict in result.verdicts]
        if candidate_cycles != list(range(1, len(candidate_cycles) + 1)):
            raise NonResumableTrajectoryError(
                "candidate checkpoints must have contiguous cycles beginning at one"
            )
        if verdict_cycles != list(range(1, len(verdict_cycles) + 1)):
            raise NonResumableTrajectoryError(
                "verdict checkpoints must have contiguous cycles beginning at one"
            )
        if len(result.verdicts) > len(result.candidates) or (
            len(result.candidates) - len(result.verdicts) > 1
        ):
            raise NonResumableTrajectoryError("checkpoint is not at a candidate/verdict boundary")
        if len(result.candidates) > self.config.max_cycles:
            raise NonResumableTrajectoryError("checkpoint exceeds the configured cycle limit")
        call_by_index = {call.index: call for call in result.calls}
        for candidate in result.candidates:
            call = call_by_index.get(candidate.call_index)
            if call is None or call.role is not candidate.role or call.cycle != candidate.cycle:
                raise NonResumableTrajectoryError(
                    f"candidate {candidate.cycle} does not reference its producing call"
                )
        for verdict in result.verdicts:
            call = call_by_index.get(verdict.call_index)
            if call is None or call.role is not Role.VERIFIER or call.cycle != verdict.cycle:
                raise NonResumableTrajectoryError(
                    f"verdict {verdict.cycle} does not reference its verifier call"
                )

    def _candidate_from_call(
        self,
        call: CallRecord,
        role: Role,
        cycle: int,
    ) -> CandidateRecord:
        if call.response is None or call.response.message.tool_calls:
            raise NonResumableTrajectoryError(
                f"{role.value} cycle {cycle} has no final assistant response"
            )
        content = (call.response.message.content or "").strip()
        if not content:
            raise NonResumableTrajectoryError(
                f"cycle {cycle} completed parent call has no candidate content"
            )
        return CandidateRecord(
            cycle=cycle,
            role=role,
            content=content,
            reasoning=call.response.message.reasoning,
            call_index=call.index,
        )

    def _recover_verdict(
        self,
        state: _RunState,
        *,
        cycle: int,
        candidate: str,
    ) -> VerdictRecord | None:
        calls = [
            call
            for call in state.result.calls
            if call.cycle == cycle and call.role is Role.VERIFIER
        ]
        if not calls:
            return None
        if len(calls) != 1 or calls[0].response is None:
            raise NonResumableTrajectoryError(
                f"cycle {cycle} has an ambiguous verifier call history"
            )
        try:
            return self._parse_verdict_completion(
                state.result.request,
                cycle=cycle,
                candidate=candidate,
                completion=calls[0].response,
                call_index=calls[0].index,
            )
        except _RunAbort as exc:
            raise NonResumableTrajectoryError(
                f"cycle {cycle} completed verifier response is invalid: {exc}"
            ) from exc

    async def _produce_candidate(
        self,
        state: _RunState,
        *,
        cycle: int,
        role: Role,
        prior_candidate: str | None,
        feedback: str | None,
        phase_cap: int,
        keep: int,
        allow_subagents: bool,
    ) -> tuple[str, str | None, int]:
        request = state.result.request
        base_task = request.solver_prompt or f"Problem:\n{request.problem}"
        if role is Role.REVISER:
            system = self.config.reviser_system_prompt
            user = (
                f"Original task and required output format:\n{base_task}\n\n"
                f"Candidate to replace:\n{prior_candidate}\n\n"
                f"Verifier feedback:\n{feedback}\n\n"
                "Return a complete replacement response that follows the original task."
            )
        elif prior_candidate is not None:
            system = self.config.generator_system_prompt
            user = (
                f"Original task and required output format:\n{base_task}\n\n"
                f"Rejected candidate:\n{prior_candidate}\n\n"
                f"Verifier feedback:\n{feedback}\n\n"
                "Develop a substantially corrected response that follows the original task."
            )
        else:
            system = self.config.generator_system_prompt
            user = base_task
        messages: list[Mapping[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tools = [SPAWN_SUBAGENTS_TOOL] if allow_subagents else None
        parent_calls = sorted(
            (call for call in state.result.calls if call.cycle == cycle and call.role is role),
            key=lambda call: call.index,
        )
        child_calls = sorted(
            (
                call
                for call in state.result.calls
                if call.cycle == cycle and call.role is Role.SUBAGENT
            ),
            key=lambda call: call.index,
        )
        phase_calls = sorted([*parent_calls, *child_calls], key=lambda call: call.index)
        if phase_calls and (not parent_calls or phase_calls[0].role is Role.SUBAGENT):
            raise NonResumableTrajectoryError(
                f"cycle {cycle} contains a subagent call before its parent call"
            )
        if phase_calls:
            phase_start = self._spent_before_call(state.result, phase_calls[0].index)
        else:
            phase_start = state.budget.spent_generated_tokens
        wave_used = False
        tool_rounds = 0
        parent_position = 0

        while True:
            call_seed = (
                request.seed
                if cycle == 1 and tool_rounds == 0
                else _stable_seed(request.seed, request.problem_id, role.value, cycle, tool_rounds)
            )
            label = f"{role.value}.cycle_{cycle}.turn_{tool_rounds}"
            next_parent_index: int | None = None
            if parent_position < len(parent_calls):
                recorded = parent_calls[parent_position]
                if parent_position + 1 < len(parent_calls):
                    next_parent_index = parent_calls[parent_position + 1].index
                self._validate_replayed_parent_call(
                    recorded,
                    messages=messages,
                    tools=tools,
                    label=label,
                    seed=call_seed,
                )
                if recorded.response is None:
                    raise NonResumableTrajectoryError(
                        f"completed call {recorded.request_id!r} has no response"
                    )
                completion = recorded.response
                call_index = recorded.index
                parent_position += 1
            else:
                phase_spent = state.budget.spent_generated_tokens - phase_start
                available = min(
                    max(0, phase_cap - phase_spent),
                    state.budget.allowance(self.config.total_generated_tokens, keep=keep),
                )
                if available <= 0:
                    raise BudgetExhausted(f"{role.value} candidate phase exhausted its budget")
                completion, call_index = await self._call(
                    state,
                    role=role,
                    cycle=cycle,
                    label=label,
                    messages=messages,
                    cap=available,
                    keep=keep,
                    seed=call_seed,
                    tools=tools,
                )

            interval_children = [
                call
                for call in child_calls
                if call.index > call_index
                and (next_parent_index is None or call.index < next_parent_index)
            ]
            if not completion.message.tool_calls:
                if interval_children:
                    raise NonResumableTrajectoryError(
                        f"cycle {cycle} has child calls after a parent response without tools"
                    )
                if parent_position < len(parent_calls):
                    raise NonResumableTrajectoryError(
                        f"cycle {cycle} has parent calls after a final response"
                    )
                content = (completion.message.content or "").strip()
                if not content:
                    raise _RunAbort(
                        TrajectoryStatus.PROTOCOL_ERROR,
                        f"{role.value} final synthesis returned no candidate content",
                    )
                return content, completion.message.reasoning, call_index

            tool_rounds += 1
            if tool_rounds > self.config.max_tool_rounds_per_candidate:
                raise _RunAbort(
                    TrajectoryStatus.PROTOCOL_ERROR,
                    f"{role.value} exceeded the tool-round limit without a candidate",
                )

            messages.append(completion.message.to_api_dict())
            used_child_interval = False
            for tool_call in completion.message.tool_calls:
                if tool_call.name != "spawn_subagents" or not allow_subagents:
                    reason = (
                        f"unknown tool {tool_call.name!r}"
                        if tool_call.name != "spawn_subagents"
                        else "subagents are disabled for this condition"
                    )
                    tool_result: Mapping[str, Any] = {
                        "ok": False,
                        "error": reason,
                    }
                elif wave_used:
                    tool_result = {
                        "ok": False,
                        "error": "only one subagent wave is allowed per candidate",
                    }
                else:
                    wave_used = True
                    try:
                        arguments = tool_call.parsed_arguments()
                        raw_tasks = arguments.get("tasks")
                        if not isinstance(raw_tasks, list) or not raw_tasks:
                            raise ValueError("tasks must be a nonempty array")
                        tasks = self._parse_subagent_tasks(raw_tasks)
                        tool_result = await self._run_subagents(
                            state,
                            cycle=cycle,
                            tasks=tasks,
                            phase_start=phase_start,
                            phase_cap=phase_cap,
                            keep=keep,
                            parent_call_index=call_index,
                            next_parent_index=next_parent_index,
                            existing_calls=interval_children,
                        )
                        used_child_interval = True
                    except (json.JSONDecodeError, ValueError) as exc:
                        tool_result = {"ok": False, "error": str(exc)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
            if interval_children and not used_child_interval:
                raise NonResumableTrajectoryError(
                    f"cycle {cycle} has child calls without a valid spawn_subagents request"
                )

    async def _run_subagents(
        self,
        state: _RunState,
        *,
        cycle: int,
        tasks: Sequence[tuple[str, str]],
        phase_start: int,
        phase_cap: int,
        keep: int,
        parent_call_index: int,
        next_parent_index: int | None,
        existing_calls: Sequence[CallRecord],
    ) -> Mapping[str, Any]:
        prior_children = sorted(
            (
                call
                for call in state.result.calls
                if call.role is Role.SUBAGENT and call.index < parent_call_index
            ),
            key=lambda call: call.index,
        )
        remaining_slots = self.config.max_subagents - len(prior_children)
        accepted = list(tasks[:remaining_slots])
        rejected_count = len(tasks) - len(accepted)
        if not accepted:
            if existing_calls:
                raise NonResumableTrajectoryError(
                    f"cycle {cycle} contains child calls after the subagent limit was reached"
                )
            return {
                "ok": False,
                "error": "the trajectory-wide subagent limit has been reached",
                "rejected_tasks": rejected_count,
            }

        spent_through_parent = self._spent_through_call(state.result, parent_call_index)
        phase_spent = spent_through_parent - phase_start
        available_before_synthesis = min(
            max(0, phase_cap - phase_spent),
            max(
                0,
                self.config.total_generated_tokens - spent_through_parent - keep,
            ),
        )
        available = max(
            0,
            available_before_synthesis - self.config.final_candidate_reserve_tokens,
        )
        # Each parallel request gets a disjoint advertised allowance. Budget
        # permits enforce the same invariant globally when they are dispatched;
        # the subtraction above guarantees a parent synthesis turn remains.
        count = min(len(accepted), available // self.config.minimum_call_tokens)
        rejected_count += len(accepted) - count
        accepted = accepted[:count]
        if not accepted:
            if existing_calls:
                raise NonResumableTrajectoryError(
                    f"cycle {cycle} has child calls that could not fit the configured budget"
                )
            return {
                "ok": False,
                "error": "insufficient budget after reserving parent synthesis",
                "rejected_tasks": len(tasks),
            }
        base = min(self.config.subagent_cap, available // len(accepted))
        caps = [base] * len(accepted)
        spare = min(
            available - sum(caps),
            len(caps) * self.config.subagent_cap - sum(caps),
        )
        for index in range(len(caps)):
            addition = min(spare, self.config.subagent_cap - caps[index])
            caps[index] += addition
            spare -= addition

        start_index = len(prior_children)
        state.subagent_count = max(state.subagent_count, start_index + len(accepted))
        coroutines = []
        task_metadata: list[tuple[int, str, str, int]] = []
        completions: list[tuple[ChatCompletion, int] | BaseException | None] = [None] * len(
            accepted
        )
        existing = sorted(existing_calls, key=lambda call: call.index)
        if len(existing) > len(accepted):
            raise NonResumableTrajectoryError(
                f"cycle {cycle} contains more child calls than the requested wave"
            )
        for offset, ((task, context), cap) in enumerate(zip(accepted, caps, strict=True)):
            task_index = start_index + offset
            seed = _stable_seed(
                state.result.request.seed,
                state.result.request.problem_id,
                "subagent",
                cycle,
                task_index,
            )
            user = f"Problem:\n{state.result.request.problem}\n\nFocused task:\n{task}"
            if context:
                user += f"\n\nOptional context excerpt:\n{context}"
            messages = [
                {"role": "system", "content": self.config.subagent_system_prompt},
                {"role": "user", "content": user},
            ]
            task_metadata.append((task_index, task, context, seed))
            label = f"subagent.cycle_{cycle}.{task_index}"
            if offset < len(existing):
                recorded = existing[offset]
                if next_parent_index is not None and recorded.index >= next_parent_index:
                    raise NonResumableTrajectoryError(
                        f"child call {recorded.request_id!r} follows its synthesis call"
                    )
                self._validate_replayed_subagent_call(
                    recorded,
                    messages=messages,
                    label=label,
                    seed=seed,
                    cap=cap,
                )
                if recorded.response is None:
                    raise NonResumableTrajectoryError(
                        f"completed child call {recorded.request_id!r} has no response"
                    )
                completions[offset] = (recorded.response, recorded.index)
            else:
                if next_parent_index is not None:
                    raise NonResumableTrajectoryError(
                        f"cycle {cycle} persisted a synthesis call before its child wave completed"
                    )
                coroutines.append(
                    (
                        offset,
                        self._call(
                            state,
                            role=Role.SUBAGENT,
                            cycle=cycle,
                            label=label,
                            messages=messages,
                            cap=cap,
                            keep=keep,
                            seed=seed,
                        ),
                    )
                )
        gathered = await asyncio.gather(
            *(coroutine for _, coroutine in coroutines),
            return_exceptions=True,
        )
        for (offset, _), outcome in zip(coroutines, gathered, strict=True):
            completions[offset] = outcome
        first_error: BaseException | None = None
        findings: list[Mapping[str, Any]] = []
        for metadata, outcome in zip(task_metadata, completions, strict=True):
            if isinstance(outcome, BaseException):
                first_error = first_error or outcome
                continue
            if outcome is None:
                # A missing slot must correspond to an exception returned by
                # gather; retain a defensive error instead of dropping a task.
                first_error = first_error or RuntimeError("subagent call produced no outcome")
                continue
            completion, call_index = outcome
            task_index, task, context, seed = metadata
            output = (completion.message.content or "").strip()
            record = SubagentRecord(
                cycle=cycle,
                task_index=task_index,
                task=task,
                context_excerpt=context,
                seed=seed,
                output=output,
                reasoning=completion.message.reasoning,
                call_index=call_index,
            )
            prior_record = next(
                (item for item in state.result.subagents if item.call_index == call_index),
                None,
            )
            if prior_record is None:
                state.result.subagents.append(record)
            elif prior_record != record:
                raise NonResumableTrajectoryError(
                    f"subagent record for call {call_index} does not match its response"
                )
            findings.append({"task": task, "finding": output})
        state.result.subagents.sort(key=lambda record: record.task_index)
        await state.sync()
        if first_error is not None:
            raise first_error
        return {
            "ok": True,
            "findings": findings,
            "rejected_tasks": rejected_count,
        }

    def _validate_replayed_parent_call(
        self,
        call: CallRecord,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        label: str,
        seed: int,
    ) -> None:
        expected_messages = tuple(copy.deepcopy(dict(message)) for message in messages)
        expected_tools = tuple(copy.deepcopy(dict(tool)) for tool in (tools or ()))
        if call.label != label or call.seed != seed:
            raise NonResumableTrajectoryError(
                f"completed parent call {call.request_id!r} has inconsistent routing metadata"
            )
        if call.messages != expected_messages or call.tools != expected_tools:
            raise NonResumableTrajectoryError(
                f"completed parent call {call.request_id!r} does not match reconstructed context"
            )

    def _validate_replayed_subagent_call(
        self,
        call: CallRecord,
        *,
        messages: Sequence[Mapping[str, Any]],
        label: str,
        seed: int,
        cap: int,
    ) -> None:
        expected_messages = tuple(copy.deepcopy(dict(message)) for message in messages)
        if (
            call.label != label
            or call.seed != seed
            or call.max_tokens != cap
            or call.messages != expected_messages
            or call.tools
        ):
            raise NonResumableTrajectoryError(
                f"completed child call {call.request_id!r} does not match its task"
            )

    @staticmethod
    def _spent_before_call(result: TrajectoryResult, call_index: int) -> int:
        return sum(
            call.usage.completion_tokens
            for call in result.calls
            if call.index < call_index and call.usage is not None
        )

    @staticmethod
    def _spent_through_call(result: TrajectoryResult, call_index: int) -> int:
        return sum(
            call.usage.completion_tokens
            for call in result.calls
            if call.index <= call_index and call.usage is not None
        )

    def _parse_subagent_tasks(self, raw_tasks: Sequence[Any]) -> list[tuple[str, str]]:
        tasks: list[tuple[str, str]] = []
        for item in raw_tasks:
            if not isinstance(item, Mapping):
                raise ValueError("each subagent task must be an object")
            task = item.get("task")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("each subagent task requires nonempty task text")
            context = item.get("context_excerpt", "")
            if not isinstance(context, str):
                raise ValueError("context_excerpt must be a string")
            tasks.append(
                (
                    task.strip(),
                    context.strip()[: self.config.subagent_context_max_chars],
                )
            )
        return tasks

    async def _verify(
        self,
        state: _RunState,
        *,
        cycle: int,
        candidate: str,
        keep: int,
    ) -> VerdictRecord:
        request = state.result.request
        reference_mode = request.condition is Condition.GVR_REFERENCE
        system = (
            self.config.reference_verifier_system_prompt
            if reference_mode
            else self.config.verifier_system_prompt
        )
        user = f"Problem:\n{request.problem}\n\nCandidate solution:\n{candidate}"
        if reference_mode:
            user += f"\n\nReference proof (verifier-only):\n{request.reference_proof}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        completion, call_index = await self._call(
            state,
            role=Role.VERIFIER,
            cycle=cycle,
            label=f"verifier.cycle_{cycle}",
            messages=messages,
            cap=self.config.verifier_cap,
            keep=keep,
            seed=_stable_seed(request.seed, request.problem_id, "verifier", cycle),
            tools=[SUBMIT_VERDICT_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "submit_verdict"},
            },
        )
        return self._parse_verdict_completion(
            request,
            cycle=cycle,
            candidate=candidate,
            completion=completion,
            call_index=call_index,
        )

    def _parse_verdict_completion(
        self,
        request: TrajectoryRequest,
        *,
        cycle: int,
        candidate: str,
        completion: ChatCompletion,
        call_index: int,
    ) -> VerdictRecord:
        reference_mode = request.condition is Condition.GVR_REFERENCE
        calls = completion.message.tool_calls
        if len(calls) != 1 or calls[0].name != "submit_verdict":
            raise _RunAbort(
                TrajectoryStatus.PROTOCOL_ERROR,
                "verifier must make exactly one submit_verdict tool call",
            )
        try:
            arguments = calls[0].parsed_arguments()
            required_fields = ("outcome", "critique", "fault_category", "candidate_excerpt")
            if any(not isinstance(arguments.get(name), str) for name in required_fields):
                raise ValueError(
                    "outcome, critique, fault_category, and candidate_excerpt must all be strings"
                )
            verdict = Verdict(arguments.get("outcome"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise _RunAbort(
                TrajectoryStatus.PROTOCOL_ERROR,
                f"invalid verifier verdict: {exc}",
            ) from exc
        critique = _clean_text(arguments.get("critique"), self.config.critique_max_chars)
        category = _clean_text(arguments.get("fault_category"), 80)
        excerpt = self._candidate_local_excerpt(arguments.get("candidate_excerpt"), candidate)
        if reference_mode:
            critique = _remove_reference_only_copy(
                critique,
                reference=request.reference_proof or "",
                candidate=candidate,
            )[: self.config.critique_max_chars]
            category = _remove_reference_only_copy(
                category,
                reference=request.reference_proof or "",
                candidate=candidate,
            )[:80]
            if not critique and verdict is not Verdict.CORRECT:
                label = category or "substantive"
                critique = (
                    f"The verifier identified a {label} issue; "
                    "re-check the cited candidate passage."
                )
        return VerdictRecord(
            cycle=cycle,
            verdict=verdict,
            critique=critique,
            fault_category=category,
            candidate_excerpt=excerpt,
            call_index=call_index,
        )

    def _candidate_local_excerpt(self, value: Any, candidate: str) -> str:
        excerpt = _clean_text(value, self.config.candidate_excerpt_max_chars)
        if not excerpt:
            return ""
        location = candidate.casefold().find(excerpt.casefold())
        if location < 0:
            return ""
        return candidate[location : location + len(excerpt)]

    def _feedback_for_solver(self, verdict: VerdictRecord) -> str:
        pieces = []
        if verdict.fault_category:
            pieces.append(f"Category: {verdict.fault_category}")
        if verdict.critique:
            pieces.append(f"Diagnosis: {verdict.critique}")
        if verdict.candidate_excerpt:
            pieces.append(f"Candidate passage: {verdict.candidate_excerpt}")
        return "\n".join(pieces)[: self.config.critique_max_chars]

    async def _call(
        self,
        state: _RunState,
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
    ) -> tuple[ChatCompletion, int]:
        context_cap = self._context_completion_cap(messages, tools)
        if context_cap < self.config.minimum_call_tokens:
            raise _RunAbort(
                TrajectoryStatus.CONTEXT_EXHAUSTED,
                f"{label} has only {context_cap} context tokens available, below the "
                f"minimum call size {self.config.minimum_call_tokens}",
            )
        cap = min(cap, context_cap)
        allowance = state.budget.allowance(cap, keep=keep)
        if allowance < self.config.minimum_call_tokens:
            raise BudgetExhausted(
                f"{label} has only {allowance} generated tokens available, below the "
                f"minimum call size {self.config.minimum_call_tokens}"
            )
        permit = state.budget.reserve(cap, keep=keep, label=label)
        call_index = state.next_call_index
        state.next_call_index += 1
        request_id = (
            f"call-{call_index:06d}-"
            f"{_stable_seed(state.result.request.seed, state.result.request.problem_id, label):08x}"
        )
        frozen_messages = tuple(copy.deepcopy(dict(message)) for message in messages)
        frozen_tools = tuple(copy.deepcopy(dict(tool)) for tool in (tools or ()))
        metadata = {
            "index": call_index,
            "label": label,
            "cycle": cycle,
            "seed": seed,
            "max_tokens": permit.max_tokens,
            "problem_id": state.result.request.problem_id,
            "condition": state.result.request.condition.value,
        }
        if self.request_lifecycle is not None:
            try:
                started = self.request_lifecycle.begin_request(
                    request_id,
                    role=role.value,
                    metadata=metadata,
                )
                if inspect.isawaitable(started):
                    await started
            except Exception as exc:
                state.budget.cancel(permit)
                state.result.calls.append(
                    CallRecord(
                        index=call_index,
                        request_id=request_id,
                        role=role,
                        cycle=cycle,
                        label=label,
                        seed=seed,
                        max_tokens=permit.max_tokens,
                        messages=frozen_messages,
                        tools=frozen_tools,
                        response=None,
                        usage=None,
                        error=f"request lifecycle begin failed: {type(exc).__name__}: {exc}",
                    )
                )
                state.result.calls.sort(key=lambda record: record.index)
                await state.sync()
                raise _RunAbort(
                    TrajectoryStatus.FAILED,
                    f"could not persist {label} request start: {exc}",
                ) from exc
        try:
            completion = await self.client.complete(
                messages,
                model=self.config.model,
                max_tokens=permit.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                min_p=self.config.min_p,
                presence_penalty=self.config.presence_penalty,
                repetition_penalty=self.config.repetition_penalty,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                extra_body=self._extra_body_for(permit.max_tokens),
            )
        except asyncio.CancelledError:
            state.budget.invalidate(permit, "request cancelled with unknown usage")
            await state.sync()
            raise
        except Exception as exc:
            known_usage = exc.usage if isinstance(exc, ChatClientError) else None
            if known_usage is not None:
                try:
                    state.budget.settle(permit, known_usage)
                except BudgetAccountingError as budget_exc:
                    exc = budget_exc
                status = TrajectoryStatus.PROTOCOL_ERROR
            else:
                state.budget.invalidate(permit, str(exc))
                status = TrajectoryStatus.INVALID_USAGE
            state.result.calls.append(
                CallRecord(
                    index=call_index,
                    request_id=request_id,
                    role=role,
                    cycle=cycle,
                    label=label,
                    seed=seed,
                    max_tokens=permit.max_tokens,
                    messages=frozen_messages,
                    tools=frozen_tools,
                    response=None,
                    usage=known_usage,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            state.result.calls.sort(key=lambda record: record.index)
            state.refresh()
            if known_usage is not None:
                await self._complete_lifecycle(
                    state,
                    request_id=request_id,
                    label=label,
                    usage=known_usage,
                )
            await state.sync()
            raise _RunAbort(status, f"{label} failed: {exc}") from exc

        try:
            state.budget.settle(permit, completion.usage)
        except BudgetAccountingError as exc:
            state.result.calls.append(
                CallRecord(
                    index=call_index,
                    request_id=request_id,
                    role=role,
                    cycle=cycle,
                    label=label,
                    seed=seed,
                    max_tokens=permit.max_tokens,
                    messages=frozen_messages,
                    tools=frozen_tools,
                    response=completion,
                    usage=completion.usage,
                    error=str(exc),
                )
            )
            state.result.calls.sort(key=lambda record: record.index)
            state.refresh()
            await self._complete_lifecycle(
                state,
                request_id=request_id,
                label=label,
                usage=completion.usage,
            )
            await state.sync()
            raise _RunAbort(TrajectoryStatus.PROTOCOL_ERROR, str(exc)) from exc
        state.result.calls.append(
            CallRecord(
                index=call_index,
                request_id=request_id,
                role=role,
                cycle=cycle,
                label=label,
                seed=seed,
                max_tokens=permit.max_tokens,
                messages=frozen_messages,
                tools=frozen_tools,
                response=completion,
                usage=completion.usage,
            )
        )
        state.result.calls.sort(key=lambda record: record.index)
        state.refresh()
        await self._complete_lifecycle(
            state,
            request_id=request_id,
            label=label,
            usage=completion.usage,
        )
        await state.sync()
        return completion, call_index

    def _context_completion_cap(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
    ) -> int:
        if self.count_messages is None:
            prompt_tokens = 0
        else:
            prompt_tokens = self.count_messages(messages, tools)
            if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
                raise TypeError("count_messages must return an integer")
            if prompt_tokens < 0:
                raise ValueError("count_messages returned a negative token count")
        return max(
            0,
            self.config.context_tokens - self.config.context_headroom_tokens - prompt_tokens,
        )

    def _extra_body_for(self, max_tokens: int) -> Mapping[str, Any]:
        """Per-call request extras, including the thinking budget when enabled.

        The budget is derived from the granted cap rather than configured
        outright, because the correction pool hands later cycles a much smaller
        cap than the initial generator call.
        """

        budget = self.config.thinking_budget_for(max_tokens)
        if budget is None:
            return self.config.extra_body
        return {
            **self.config.extra_body,
            "custom_logit_processor": self.config.thinking_budget_processor,
            "custom_params": {"thinking_budget": budget},
        }

    async def _complete_lifecycle(
        self,
        state: _RunState,
        *,
        request_id: str,
        label: str,
        usage: TokenUsage,
    ) -> None:
        if self.request_lifecycle is None:
            return
        try:
            completed = self.request_lifecycle.complete_request(
                request_id,
                state=f"{label}.complete",
                payload=state.result.to_dict(),
                usage=usage.to_dict(),
            )
            if inspect.isawaitable(completed):
                await completed
        except Exception as exc:
            raise _RunAbort(
                TrajectoryStatus.FAILED,
                f"could not persist {label} request completion: {exc}",
            ) from exc


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    # OpenAI-compatible servers generally accept signed 32-bit seeds.
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


def _clean_text(value: Any, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:maximum]


def _remove_reference_only_copy(text: str, *, reference: str, candidate: str) -> str:
    """Drop clauses containing long spans copied only from the reference.

    This is a leakage guard, not a semantic similarity detector.  Eight-word
    normalized n-grams are long enough to permit ordinary mathematical phrases
    while rejecting verbatim proof transfer. Any same span already present in
    the candidate is safe because feedback may quote the candidate itself.
    """

    if not text or not reference:
        return text
    reference_ngrams = _word_ngrams(reference, 8)
    candidate_ngrams = _word_ngrams(candidate, 8)
    forbidden = reference_ngrams - candidate_ngrams
    if not forbidden:
        return text
    clauses = re.split(r"(?<=[.!?;])\s+|\n+", text)
    retained: list[str] = []
    for clause in clauses:
        if not (_word_ngrams(clause, 8) & forbidden):
            retained.append(clause.strip())
    return " ".join(part for part in retained if part)


def _word_ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


__all__ = [
    "AletheiaOrchestrator",
    "CountMessages",
    "GENERATOR_SYSTEM_PROMPT",
    "NonResumableTrajectoryError",
    "OrchestratorConfig",
    "REFERENCE_VERIFIER_SYSTEM_PROMPT",
    "REVISER_SYSTEM_PROMPT",
    "RequestLifecycle",
    "SPAWN_SUBAGENTS_TOOL",
    "SUBAGENT_SYSTEM_PROMPT",
    "SUBMIT_VERDICT_TOOL",
    "TokenCounter",
    "VERIFIER_SYSTEM_PROMPT",
]
