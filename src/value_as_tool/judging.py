"""Condition-blind QED-Nano judging and strict response parsers."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import regex

from value_as_tool.benchmarks import BenchmarkItem, QEDPromptSet, benchmark_spec
from value_as_tool.client import ChatClientError

SELF_EVALUATION = re.compile(
    r"(?ms)^#+\s*[*\s]*(?:(?:Self|Final)\s*[*\s]*){0,2}Evaluation\b.*$"
)

# GPT-OSS-20B exposes a 131,072-token context.  Normal prompts retain the
# released QED-Nano 95,000 output allowance.  Only oversized proof candidates
# use the auditable recovery budget below.
JUDGE_CONTEXT_TOKENS = 131_072
JUDGE_CONTEXT_INPUT_LIMIT = 97_280
JUDGE_CONTEXT_RECOVERY_OUTPUT_TOKENS = 32_768
JUDGE_CONTEXT_HEADROOM_TOKENS = 1_024
QED_JUDGE_MAX_TOKENS = 95_000
JUDGE_CONTEXT_RECOVERY_POLICY_VERSION = 1
JUDGE_CONTEXT_TRUNCATION_MARKER = (
    "\n\n[... middle of proposed solution omitted deterministically to fit "
    "the GPT-OSS judge context ...]\n\n"
)
_RECOVERY_POLICY = {
    "algorithm": "balanced_token_middle_truncation",
    "context_tokens": JUDGE_CONTEXT_TOKENS,
    "input_limit_tokens": JUDGE_CONTEXT_INPUT_LIMIT,
    "marker": JUDGE_CONTEXT_TRUNCATION_MARKER,
    "output_tokens": JUDGE_CONTEXT_RECOVERY_OUTPUT_TOKENS,
    "solution_preprocessing": "qed_remove_self_evaluation",
    "template_headroom_tokens": JUDGE_CONTEXT_HEADROOM_TOKENS,
    "tie_break": "one_extra_head_token",
    "version": JUDGE_CONTEXT_RECOVERY_POLICY_VERSION,
}
JUDGE_CONTEXT_RECOVERY_POLICY_SHA256 = hashlib.sha256(
    json.dumps(
        _RECOVERY_POLICY,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


class TokenCounter(Protocol):
    def count_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> int: ...

    def count_text(self, text: str) -> int: ...

    def encode_text(self, text: str) -> Sequence[int]: ...

    def decode_tokens(self, token_ids: Sequence[int]) -> str: ...


class HuggingFaceTokenCounter:
    """Small adapter around a Transformers tokenizer, loaded by the caller."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return str(
            self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def count_text(self, text: str) -> int:
        return len(self.encode_text(text))

    def count_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = list(tools)
        encoded = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        return len(encoded)


def remove_self_evaluation(text: str) -> str:
    """Apply the preprocessing used by QED-Nano's public evaluator."""

    return SELF_EVALUATION.sub("", text).strip()


def _remove_inner_boxed(value: str) -> str:
    pattern = r"(\\boxed|\\fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    for match in list(regex.finditer(pattern, value)):
        value = value.replace(match.group(0), match.group(2))
    return value


def find_last_boxed_content(text: str) -> str | None:
    """Extract QED-Nano's last boxed answer, including multi-box lines."""

    pattern = r"(boxed|fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None
    if len(matches) > 1:
        for line in reversed(text.split("\n")):
            line_matches = list(regex.finditer(pattern, line))
            if line_matches:
                joined = ",".join(match.group(2) for match in line_matches)
                return _remove_inner_boxed(joined)
    return _remove_inner_boxed(matches[-1].group(2))


def extract_candidate_answer(solution: str) -> str:
    parsed = find_last_boxed_content(solution)
    if parsed:
        return parsed
    lines = solution.strip().split("\n")
    return "..." + "\n".join(lines[-5:]) if len(lines) >= 5 else solution


def parse_proof_score(text: str) -> int | None:
    """Parse the last QED ``points`` tag, preserving its one-digit contract."""

    matches = re.findall(r"<points.*?>(.*?)</points>", text, re.DOTALL)
    if not matches:
        return None
    value = matches[-1].strip()
    if not value or not value[0].isdigit():
        return None
    score = int(value[0])
    return score if 0 <= score <= 7 else None


def parse_answer_correct(text: str) -> bool:
    """Match QED-Nano: a missing or malformed boxed grade is incorrect."""

    parsed = find_last_boxed_content(text)
    return bool(parsed and "incorrect" not in parsed.casefold())


@dataclass(frozen=True, slots=True)
class JudgePromptPlan:
    prompt: str
    input_tokens: int | None
    max_tokens: int
    context_recovery: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class JudgeResult:
    raw_response: str
    score: int | None
    correct: bool
    status: str
    usage: dict[str, Any]
    messages: tuple[dict[str, Any], ...]
    context_recovery: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _middle_truncated_solution(
    counter: TokenCounter,
    token_ids: tuple[int, ...],
    kept_tokens: int,
) -> tuple[str, int, int]:
    head_tokens = (kept_tokens + 1) // 2
    tail_tokens = kept_tokens // 2
    head = counter.decode_tokens(token_ids[:head_tokens])
    tail = counter.decode_tokens(token_ids[-tail_tokens:]) if tail_tokens else ""
    return head + JUDGE_CONTEXT_TRUNCATION_MARKER + tail, head_tokens, tail_tokens


def prepare_judge_prompt(
    prompts: QEDPromptSet,
    item: BenchmarkItem,
    model_solution: str,
    *,
    default_max_tokens: int = QED_JUDGE_MAX_TOKENS,
    token_counter: TokenCounter | None = None,
) -> JudgePromptPlan:
    """Render a blind prompt and middle-truncate only an oversized proof.

    The problem, reference proof, grading rubric, and instructions are never
    truncated.  A binary search retains the largest balanced prefix/suffix of
    the submitted proof that fits the recovery input budget.
    """

    original_prompt = prompts.judge_prompt(item, model_solution)
    if token_counter is None:
        return JudgePromptPlan(original_prompt, None, default_max_tokens)
    original_tokens = token_counter.count_messages(
        ({"role": "user", "content": original_prompt},)
    )
    if original_tokens <= JUDGE_CONTEXT_INPUT_LIMIT:
        return JudgePromptPlan(original_prompt, original_tokens, default_max_tokens)
    if benchmark_spec(item.benchmark).kind != "proof":
        raise ValueError(
            "oversized answer judge prompt cannot be reduced without changing "
            "QED answer-extraction semantics"
        )

    normalized = remove_self_evaluation(model_solution)
    solution_tokens = tuple(token_counter.encode_text(normalized))
    marker_prompt = prompts.judge_prompt(item, JUDGE_CONTEXT_TRUNCATION_MARKER)
    marker_tokens = token_counter.count_messages(
        ({"role": "user", "content": marker_prompt},)
    )
    if marker_tokens > JUDGE_CONTEXT_INPUT_LIMIT:
        raise ValueError(
            "judge instructions/reference/rubric exceed the context limit even "
            "after omitting the proposed proof"
        )

    low, high = 0, len(solution_tokens) - 1
    best = (JUDGE_CONTEXT_TRUNCATION_MARKER, marker_prompt, marker_tokens, 0, 0)
    while low <= high:
        kept = (low + high) // 2
        candidate, head, tail = _middle_truncated_solution(
            token_counter, solution_tokens, kept
        )
        prompt = prompts.judge_prompt(item, candidate)
        prompt_tokens = token_counter.count_messages(
            ({"role": "user", "content": prompt},)
        )
        if prompt_tokens <= JUDGE_CONTEXT_INPUT_LIMIT:
            best = candidate, prompt, prompt_tokens, head, tail
            low = kept + 1
        else:
            high = kept - 1

    candidate, prompt, prompt_tokens, head, tail = best
    kept = head + tail
    metadata = {
        "applied": True,
        "context_tokens": JUDGE_CONTEXT_TOKENS,
        "effective_prompt_sha256": _sha256_text(prompt),
        "effective_prompt_tokens": prompt_tokens,
        "effective_solution_sha256": _sha256_text(candidate),
        "effective_solution_tokens": token_counter.count_text(candidate),
        "input_limit_tokens": JUDGE_CONTEXT_INPUT_LIMIT,
        "kept_head_solution_tokens": head,
        "kept_solution_tokens": kept,
        "kept_tail_solution_tokens": tail,
        "marker_sha256": _sha256_text(JUDGE_CONTEXT_TRUNCATION_MARKER),
        "normalized_solution_sha256": _sha256_text(normalized),
        "omitted_solution_tokens": len(solution_tokens) - kept,
        "original_prompt_sha256": _sha256_text(original_prompt),
        "original_prompt_tokens": original_tokens,
        "original_solution_tokens": len(solution_tokens),
        "policy_sha256": JUDGE_CONTEXT_RECOVERY_POLICY_SHA256,
        "policy_version": JUDGE_CONTEXT_RECOVERY_POLICY_VERSION,
        "requested_output_tokens": JUDGE_CONTEXT_RECOVERY_OUTPUT_TOKENS,
        "solution_preprocessing": "qed_remove_self_evaluation",
        "template_headroom_tokens": JUDGE_CONTEXT_HEADROOM_TOKENS,
    }
    return JudgePromptPlan(
        prompt=prompt,
        input_tokens=prompt_tokens,
        max_tokens=JUDGE_CONTEXT_RECOVERY_OUTPUT_TOKENS,
        context_recovery=metadata,
    )


def validate_judge_context_recovery(
    row: Mapping[str, Any],
    prompts: QEDPromptSet,
    item: BenchmarkItem,
    model_solution: str,
    *,
    default_max_tokens: int = QED_JUDGE_MAX_TOKENS,
    token_counter: TokenCounter,
) -> None:
    """Verify that a persisted truncated judgment is exactly reproducible."""

    expected = prepare_judge_prompt(
        prompts,
        item,
        model_solution,
        default_max_tokens=default_max_tokens,
        token_counter=token_counter,
    )
    recorded = row.get("context_recovery", row.get("judge_context_recovery"))
    terminal = str(row.get("status", row.get("judge_status", ""))) in {
        "completed",
        "empty_answer",
    }
    if recorded is None:
        if terminal and expected.context_recovery is not None:
            raise ValueError("terminal oversized judgment lacks recovery metadata")
        return
    if recorded != expected.context_recovery:
        raise ValueError("judge context-recovery metadata does not match the answer")
    messages = row.get("messages", row.get("judge_messages"))
    if not (
        isinstance(messages, Sequence)
        and messages
        and messages[0] == {"role": "user", "content": expected.prompt}
    ):
        raise ValueError("recovered judgment does not preserve its effective prompt")


def build_judge_request(
    item: BenchmarkItem,
    final_text: str,
    *,
    prompts: QEDPromptSet | None = None,
    max_tokens: int = QED_JUDGE_MAX_TOKENS,
    token_counter: TokenCounter | None = None,
) -> tuple[list[dict[str, Any]], JudgePromptPlan]:
    """Build the sole condition-blind message sent to the external judge."""

    prompt_set = prompts or QEDPromptSet()
    plan = prepare_judge_prompt(
        prompt_set,
        item,
        final_text,
        default_max_tokens=max_tokens,
        token_counter=token_counter,
    )
    return [{"role": "user", "content": plan.prompt}], plan


def _usage_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "usage_exact": False}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif not isinstance(value, Mapping) and hasattr(value, "model_dump"):
        value = value.model_dump()
    elif not isinstance(value, Mapping) and hasattr(value, "__dict__"):
        value = vars(value)
    if not isinstance(value, Mapping):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "usage_exact": False}
    usage = dict(value)
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    exact = (
        isinstance(prompt, int)
        and not isinstance(prompt, bool)
        and isinstance(completion, int)
        and not isinstance(completion, bool)
        and prompt >= 0
        and completion >= 0
    )
    prompt_int = int(prompt) if exact else 0
    completion_int = int(completion) if exact else 0
    usage.update(
        prompt_tokens=prompt_int,
        completion_tokens=completion_int,
        total_tokens=prompt_int + completion_int,
        usage_exact=exact,
    )
    return usage


def _completion_parts(value: Any) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Normalize local clients, OpenAI dictionaries, and simple test doubles."""

    if isinstance(value, tuple) and len(value) == 2:
        message, usage = value
        if isinstance(message, Mapping):
            return str(message.get("content") or ""), _usage_dict(usage), dict(message)
    if hasattr(value, "message") and hasattr(value, "usage"):
        message = value.message
        content = getattr(message, "content", None)
        raw_message = (
            message.to_api_dict()
            if hasattr(message, "to_api_dict")
            else {"role": "assistant", "content": content}
        )
        return str(content or ""), _usage_dict(value.usage), raw_message
    if not isinstance(value, Mapping) and hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, Mapping):
        choices = value.get("choices")
        if isinstance(choices, Sequence) and choices:
            first = choices[0]
            message = first.get("message", {}) if isinstance(first, Mapping) else {}
        else:
            message = value.get("message", value)
        if not isinstance(message, Mapping):
            message = {"role": "assistant", "content": str(message)}
        return (
            str(message.get("content") or ""),
            _usage_dict(value.get("usage")),
            dict(message),
        )
    raise TypeError(f"unsupported judge completion type: {type(value).__name__}")


async def _call_client(client: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    if hasattr(client, "complete"):
        method = client.complete
    elif hasattr(client, "chat") and hasattr(client.chat, "completions"):
        method = client.chat.completions.create
    else:
        raise TypeError("judge client must expose complete() or chat.completions.create()")
    try:
        parameters = inspect.signature(method).parameters.values()
        accepts_kwargs = any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters)
        accepted = {parameter.name for parameter in parameters}
    except (TypeError, ValueError):
        accepts_kwargs, accepted = True, set()
    call_kwargs = {
        key: value for key, value in kwargs.items() if accepts_kwargs or key in accepted
    }
    if accepts_kwargs or "messages" in accepted:
        result = method(messages=messages, **call_kwargs)
    else:
        result = method(messages, **call_kwargs)
    return await result if inspect.isawaitable(result) else result


class JudgeRunner:
    """Asynchronous adapter for a GPT-OSS OpenAI-compatible client."""

    def __init__(
        self,
        prompts: QEDPromptSet | None = None,
        *,
        max_tokens: int = QED_JUDGE_MAX_TOKENS,
        token_counter: TokenCounter | None = None,
        reasoning_effort: str = "medium",
        model: str | None = "openai/gpt-oss-20b",
    ):
        self.prompts = prompts or QEDPromptSet()
        self.max_tokens = max_tokens
        self.token_counter = token_counter
        self.reasoning_effort = reasoning_effort
        self.model = model

    async def judge(
        self,
        item: BenchmarkItem,
        final_text: str,
        client: Any,
    ) -> JudgeResult:
        try:
            messages, plan = build_judge_request(
                item,
                final_text,
                prompts=self.prompts,
                max_tokens=self.max_tokens,
                token_counter=self.token_counter,
            )
        except Exception as exc:  # tokenizer and template failures are durable
            return JudgeResult(
                raw_response="",
                score=None,
                correct=False,
                status="context_error",
                usage=_usage_dict(None),
                messages=(),
                error=str(exc),
            )

        kwargs: dict[str, Any] = {
            "max_tokens": plan.max_tokens,
            # The released QED GPT-OSS-medium profile leaves all sampling
            # fields unset, preserving provider defaults.
            "use_sampling": None,
            "reasoning_effort": self.reasoning_effort,
        }
        if self.model is not None:
            kwargs["model"] = self.model
        try:
            completion = await _call_client(client, messages, **kwargs)
            content, usage, assistant = _completion_parts(completion)
        except ChatClientError as exc:
            return JudgeResult(
                raw_response="",
                score=None,
                correct=False,
                status="api_error",
                usage=_usage_dict(exc.usage),
                messages=tuple(messages),
                context_recovery=plan.context_recovery,
                error=str(exc),
            )
        except Exception as exc:  # provider exception classes vary
            return JudgeResult(
                raw_response="",
                score=None,
                correct=False,
                status="api_error",
                usage=_usage_dict(None),
                messages=tuple(messages),
                context_recovery=plan.context_recovery,
                error=str(exc),
            )

        transcript = (*messages, assistant)
        if benchmark_spec(item.benchmark).kind == "answer":
            correct = parse_answer_correct(content)
            return JudgeResult(
                content,
                int(correct),
                correct,
                "completed",
                usage,
                transcript,
                plan.context_recovery,
            )
        score = parse_proof_score(content)
        return JudgeResult(
            content,
            score,
            score == 7,
            "completed" if score is not None else "parse_error",
            usage,
            transcript,
            plan.context_recovery,
        )


__all__ = [
    "HuggingFaceTokenCounter",
    "JUDGE_CONTEXT_HEADROOM_TOKENS",
    "JUDGE_CONTEXT_INPUT_LIMIT",
    "JUDGE_CONTEXT_RECOVERY_OUTPUT_TOKENS",
    "JUDGE_CONTEXT_RECOVERY_POLICY_SHA256",
    "JUDGE_CONTEXT_TOKENS",
    "JUDGE_CONTEXT_TRUNCATION_MARKER",
    "JudgePromptPlan",
    "JudgeResult",
    "JudgeRunner",
    "QED_JUDGE_MAX_TOKENS",
    "TokenCounter",
    "build_judge_request",
    "extract_candidate_answer",
    "find_last_boxed_content",
    "parse_answer_correct",
    "parse_proof_score",
    "prepare_judge_prompt",
    "remove_self_evaluation",
    "validate_judge_context_recovery",
]
