"""Decide whether SGLang runs custom logit processors at all on a live server.

The Qwen3.5 thinking-budget payload provably clamps when its ``__call__`` is
invoked on synthetic logits, yet production shows no clamping.  That leaves two
possibilities: the ids never match in a real request, or the processor is never
called.  This separates them by sending a processor whose start token is a word
the model is certain to emit -- if the mechanism is alive that clamps almost
immediately and reasoning collapses; if reasoning is unchanged the mechanism is
dead and no choice of token ids can matter.

Every branch of ThinkingBudgetLogitProcessor writes ``logits[i, :]``, so this
only ever perturbs its own row and is safe to run against a busy server.

Usage: probe_processor_alive.py http://127.0.0.1:PORT/v1 /path/to/model
"""

from __future__ import annotations

import json
import sys
import urllib.request

import dill
from sglang.srt.sampling.custom_logit_processor import ThinkingBudgetLogitProcessor
from transformers import AutoTokenizer

PROMPT = (
    "Prove or disprove: every positive integer greater than 2 can be written as "
    "the sum of two primes. Work carefully through the cases and explain your "
    "reasoning in full before giving a final answer."
)
MAX_TOKENS = 2048
QWEN35_THINK = (248068, 248069)


def payload_for(start: int, end: int, newline: int) -> str:
    processor = type(
        "ProbeThinkingBudgetLogitProcessor",
        (ThinkingBudgetLogitProcessor,),
        {
            "THINKING_START_TOKEN_ID": start,
            "THINKING_END_TOKEN_ID": end,
            "NEW_LINE_TOKEN_ID": newline,
        },
    )
    return json.dumps({"callable": dill.dumps(processor).hex()})


def ask(base_url: str, model: str, processor: str | None, budget: int | None) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if processor is not None:
        body["custom_logit_processor"] = processor
        body["custom_params"] = {"thinking_budget": budget}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        value = json.load(response)
    choice = value["choices"][0]
    message = choice.get("message") or {}
    return {
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "content_chars": len(message.get("content") or ""),
        "completion_tokens": (value.get("usage") or {}).get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }


def main() -> int:
    base_url, model_path = sys.argv[1], sys.argv[2]
    model = json.load(
        urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=30)
    )["data"][0]["id"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    common = tokenizer.encode(" the", add_special_tokens=False)[0]
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    think_in_prompt = QWEN35_THINK[0] in prompt_ids
    print(f"model={model}")
    print(f"' the' id={common}  newline id={newline}")
    print(f"<think>({QWEN35_THINK[0]}) in prompt: {think_in_prompt}")
    if think_in_prompt:
        print(f"  at index {prompt_ids.index(QWEN35_THINK[0])} of {len(prompt_ids)}")
    print(f"prompt tail: {prompt_ids[-8:]}", flush=True)

    baseline = ask(base_url, model, None, None)
    print(f"\nno processor            : {baseline}", flush=True)

    real = ask(base_url, model, payload_for(*QWEN35_THINK, newline), 128)
    print(f"real think ids, budget128: {real}", flush=True)

    # ' the' is emitted within the first few dozen tokens of any such answer, so
    # a live mechanism trips this budget almost at once.
    forced = ask(base_url, model, payload_for(common, QWEN35_THINK[1], newline), 8)
    print(f"start=' the',   budget  8: {forced}", flush=True)

    alive = forced["reasoning_chars"] < baseline["reasoning_chars"] / 2
    print(f"\nMECHANISM ALIVE: {alive}")
    if alive:
        print("-> processors do run; the Qwen3.5 think ids are not matching in production")
    else:
        print("-> processors are never invoked on this server; token ids are irrelevant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
