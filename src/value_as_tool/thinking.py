"""Model-derived SGLang thinking-budget processor profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True, slots=True)
class ThinkingTokenProfile:
    start_token_id: int
    end_token_id: int
    newline_token_id: int


def _single_token_id(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    values = [int(value) for value in encoded]
    if len(values) != 1 or values[0] < 0:
        raise ValueError(f"tokenizer does not encode {text!r} as one token: {values!r}")
    try:
        decoded = str(
            tokenizer.decode(
                values,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        decoded = str(tokenizer.decode(values, skip_special_tokens=False))
    if decoded != text:
        raise ValueError(
            f"tokenizer round-trip mismatch for {text!r}: id={values[0]}, decoded={decoded!r}"
        )
    return values[0]


def derive_thinking_token_profile(tokenizer: Any) -> ThinkingTokenProfile:
    """Derive and round-trip-check the model's thinking control tokens."""

    return ThinkingTokenProfile(
        start_token_id=_single_token_id(tokenizer, "<think>"),
        end_token_id=_single_token_id(tokenizer, "</think>"),
        newline_token_id=_single_token_id(tokenizer, "\n"),
    )


def _processor_call(self: Any, logits: Any, custom_param_list: Any) -> Any:
    if not custom_param_list:
        return logits
    for index, parameters in enumerate(custom_param_list):
        if not parameters:
            continue
        thinking_budget = parameters.get("thinking_budget")
        if (
            thinking_budget is None
            or not isinstance(thinking_budget, int)
            or thinking_budget < 0
        ):
            continue
        request = parameters.get("__req__")
        if request is None:
            continue
        token_ids = [*request.origin_input_ids, *request.output_ids]
        open_start = -1
        for position in reversed(range(len(token_ids))):
            if token_ids[position] == self.THINKING_START_TOKEN_ID:
                open_start = position
                break
            if token_ids[position] == self.THINKING_END_TOKEN_ID:
                break
        if open_start < 0 or len(token_ids) - open_start - 1 < thinking_budget:
            continue
        next_token = (
            self.THINKING_END_TOKEN_ID
            if request.output_ids
            and request.output_ids[-1] == self.NEW_LINE_TOKEN_ID
            else self.NEW_LINE_TOKEN_ID
        )
        logits[index, :] = -float("inf")
        logits[index, next_token] = 0.0
    return logits


@lru_cache(maxsize=16)
def build_thinking_budget_processor(profile: ThinkingTokenProfile) -> str:
    """Serialize a model-specific processor in SGLang's wire format."""

    import dill

    processor = type(
        "ModelThinkingBudgetLogitProcessor",
        (object,),
        {
            "THINKING_START_TOKEN_ID": profile.start_token_id,
            "THINKING_END_TOKEN_ID": profile.end_token_id,
            "NEW_LINE_TOKEN_ID": profile.newline_token_id,
            "__call__": _processor_call,
        },
    )
    return json.dumps(
        {"callable": dill.dumps(processor).hex()},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "ThinkingTokenProfile",
    "build_thinking_budget_processor",
    "derive_thinking_token_profile",
]
