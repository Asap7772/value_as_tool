"""Hugging Face chat-template token counting used for context safety."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


class HuggingFaceTokenCounter:
    def __init__(self, model_or_path: str, *, enable_thinking: bool = False) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=True)
        self.enable_thinking = enable_thinking

    def encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(
            int(token)
            for token in self.tokenizer.encode(text, add_special_tokens=False)
        )

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        kwargs = {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        try:
            return str(self.tokenizer.decode(list(token_ids), **kwargs))
        except TypeError:
            kwargs.pop("clean_up_tokenization_spaces")
            return str(self.tokenizer.decode(list(token_ids), **kwargs))

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
            "enable_thinking": self.enable_thinking,
        }
        if tools:
            kwargs["tools"] = [dict(tool) for tool in tools]
        try:
            encoded = self.tokenizer.apply_chat_template(
                [dict(message) for message in messages], **kwargs
            )
            return _encoded_length(encoded)
        except (TypeError, ValueError):
            # Context safety still needs a conservative fallback for custom
            # tokenizer templates. Provider usage remains authoritative after
            # a successful call.
            serialized = json.dumps(
                {"messages": list(messages), "tools": list(tools or ())},
                ensure_ascii=False,
                default=str,
            )
            return self.count_text(serialized)


def _encoded_length(value: Any) -> int:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("chat template output has no input_ids")
        return _encoded_length(value["input_ids"])
    shape = getattr(value, "shape", None)
    if shape is not None:
        dimensions = tuple(int(item) for item in shape)
        product = 1
        for dimension in dimensions:
            product *= dimension
        return product
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return 0
        first = value[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            return sum(_encoded_length(item) for item in value)
        return len(value)
    raise TypeError(f"unsupported tokenizer output: {type(value).__name__}")


__all__ = ["HuggingFaceTokenCounter"]
