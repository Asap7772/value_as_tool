"""Minimal asynchronous OpenAI Chat Completions client for Qwen servers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx

from .schemas import AssistantMessage, ChatCompletion, TokenUsage


class ChatClientError(RuntimeError):
    """A request or response failure.

    ``usage`` is populated when the server reported exact usage before a later
    response-shape error was discovered.  With no usage, callers must treat a
    dispatched request as unaccounted and invalidate the matched-budget run.
    """

    def __init__(
        self,
        message: str,
        *,
        usage: TokenUsage | None = None,
        request_may_have_run: bool = True,
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.request_may_have_run = request_may_have_run


class MissingUsageError(ChatClientError):
    pass


class ChatClient(Protocol):
    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        seed: int = 0,
        use_sampling: bool | None = True,
        reasoning_effort: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatCompletion: ...


class OpenAIChatClient:
    """Async client for SGLang/vLLM's OpenAI-compatible endpoint.

    The client intentionally performs no automatic retries: after a timeout or
    disconnect it is impossible to know how many tokens the server generated.
    The orchestrator records that attempt as invalid and a higher-level
    scheduler may create a new trajectory attempt.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float | httpx.Timeout = 3600.0,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.endpoint = _chat_completions_url(base_url)
        request_headers = {"Content-Type": "application/json"}
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            request_headers.update(headers)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            headers=request_headers,
        )
        self._headers = request_headers if client is not None else None

    async def __aenter__(self) -> OpenAIChatClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        seed: int = 0,
        use_sampling: bool | None = True,
        reasoning_effort: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatCompletion:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "n": 1,
            "stream": False,
        }
        if use_sampling is True:
            payload.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "presence_penalty": presence_penalty,
                    "repetition_penalty": repetition_penalty,
                    "seed": seed,
                }
            )
        elif use_sampling is False:
            payload["temperature"] = 0
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"
            payload["parallel_tool_calls"] = True
        elif tool_choice is not None:
            raise ValueError("tool_choice was supplied without tools")
        if extra_body:
            protected = {"model", "messages", "max_tokens", "n", "stream"}
            if tools:
                protected.update({"tools", "tool_choice", "parallel_tool_calls"})
            overlap = protected.intersection(extra_body)
            if overlap:
                names = ", ".join(sorted(overlap))
                raise ValueError(f"extra_body cannot override protected fields: {names}")
            payload.update(extra_body)

        try:
            response = await self._client.post(
                self.endpoint,
                json=payload,
                headers=self._headers,
            )
        except httpx.HTTPError as exc:
            raise ChatClientError(f"chat completion request failed: {exc}") from exc

        if not response.is_success:
            # Limit captured bodies: proxy error pages can be very large and
            # have no place in every trajectory artifact.
            body = response.text[:2000]
            raise ChatClientError(f"chat completion returned HTTP {response.status_code}: {body}")
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ChatClientError("chat completion response was not valid JSON") from exc
        if not isinstance(data, Mapping):
            raise ChatClientError("chat completion response must be a JSON object")
        return parse_chat_completion(data)


def parse_chat_completion(data: Mapping[str, Any]) -> ChatCompletion:
    """Parse one non-streaming completion and normalize Qwen reasoning fields."""

    raw_usage = data.get("usage")
    if not isinstance(raw_usage, Mapping):
        raise MissingUsageError("chat completion did not report exact usage")
    try:
        usage = TokenUsage.from_api(raw_usage)
    except ValueError as exc:
        raise MissingUsageError(str(exc)) from exc

    choices = data.get("choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or len(choices) != 1
        or not isinstance(choices[0], Mapping)
    ):
        raise ChatClientError("chat completion must contain exactly one choice", usage=usage)
    choice = choices[0]
    raw_message = choice.get("message")
    if not isinstance(raw_message, Mapping):
        raise ChatClientError("completion choice lacks a message", usage=usage)
    try:
        message = AssistantMessage.from_api(raw_message)
    except ValueError as exc:
        raise ChatClientError(f"invalid assistant message: {exc}", usage=usage) from exc

    identifier = data.get("id")
    model = data.get("model")
    created = data.get("created")
    finish_reason = choice.get("finish_reason")
    return ChatCompletion(
        id=identifier if isinstance(identifier, str) else None,
        model=model if isinstance(model, str) else None,
        message=message,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        usage=usage,
        created=created if isinstance(created, int) and not isinstance(created, bool) else None,
        raw=dict(data),
    )


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base:
        raise ValueError("base_url must be nonempty")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


__all__ = [
    "ChatClient",
    "ChatClientError",
    "MissingUsageError",
    "OpenAIChatClient",
    "parse_chat_completion",
]
