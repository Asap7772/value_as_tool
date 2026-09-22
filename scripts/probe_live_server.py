"""Probe a *running* production solver for whether the thinking budget clamps.

The earlier probe booted its own server, which costs a GPU allocation and a
long wait.  The solve shards already have 48 servers up with the exact config
under test, so this asks one of them directly: the same prompt twice, once
with a small ``thinking_budget`` and once without.  Two extra requests are
noise against a server already running continuous batches.

Usage: probe_live_server.py http://127.0.0.1:PORT/v1
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from value_as_tool.orchestrator import QWEN3_THINKING_BUDGET_PROCESSOR  # noqa: E402

PROMPT = (
    "Prove or disprove: every positive integer greater than 2 can be written as "
    "the sum of two primes. Work carefully through the cases and explain your "
    "reasoning in full before giving a final answer."
)
BUDGET = 256
MAX_TOKENS = 4096


def _one(base_url: str, model: str, budget: int | None) -> dict:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if budget is not None:
        payload["custom_logit_processor"] = QWEN3_THINKING_BUDGET_PROCESSOR
        payload["custom_params"] = {"thinking_budget": budget}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        value = json.load(response)
    usage = value.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    choice = (value.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or ""
    return {
        "reasoning_tokens": details.get("reasoning_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(message.get("content") or ""),
    }


def main() -> int:
    base_url = sys.argv[1]
    models = json.load(
        urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=30)
    )
    model = models["data"][0]["id"]
    print(f"model: {model}", flush=True)

    off = _one(base_url, model, None)
    print(f"no budget : {off}", flush=True)
    on = _one(base_url, model, BUDGET)
    print(f"budget={BUDGET}: {on}", flush=True)

    # reasoning_chars is the fallback signal: if the server does not report
    # reasoning_tokens, a clamped run still produces a far shorter think block.
    off_r, on_r = off["reasoning_tokens"], on["reasoning_tokens"]
    if off_r is None or on_r is None:
        clamped = on["reasoning_chars"] < off["reasoning_chars"] / 2
        print("(no reasoning_tokens reported; comparing reasoning_chars)")
    else:
        clamped = on_r <= BUDGET + 64
    print(f"\nreasoning without budget : {off_r} tokens / {off['reasoning_chars']} chars")
    print(f"reasoning with budget={BUDGET} : {on_r} tokens / {on['reasoning_chars']} chars")
    print(f"CLAMP WORKING: {clamped}")
    return 0 if clamped else 1


if __name__ == "__main__":
    raise SystemExit(main())
