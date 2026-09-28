"""Boot a solver server and check whether the thinking budget actually clamps.

The budget is applied by a custom logit processor, and SGLang degrades to a
silent no-op whenever any link in the chain is wrong (wrong think-token ids,
processor dropped, flag not reaching the scheduler).  Nothing in the harness
surfaces that: calls simply keep spending the whole cap on reasoning.  This
probe is the end-to-end check -- the same prompt twice, once with a small
budget and once without, comparing the reasoning tokens the server reports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from value_as_tool.client import OpenAIChatClient  # noqa: E402
from value_as_tool.config import load_config  # noqa: E402
from value_as_tool.harnesses.cch_plan_work_review import SUBMIT_PLAN_TOOL  # noqa: E402
from value_as_tool.server import ModelServer, ServerSpec  # noqa: E402
from value_as_tool.thinking import (  # noqa: E402
    build_thinking_budget_processor,
    derive_thinking_token_profile,
)

# Deliberately open-ended so the model reasons well past the small budget below.
PROMPT = (
    "Analyze Goldbach's conjecture in detail. Before your final answer, spend at "
    "least 1,500 words privately checking candidate proof strategies, known "
    "obstructions, and whether the requested claim is currently established."
)
BUDGET = 64
MAX_TOKENS = 4096


async def _one(
    client: OpenAIChatClient,
    model: str,
    processor: str,
    tokenizer: object,
    budget: int | None,
) -> dict:
    extra: dict = {"chat_template_kwargs": {"enable_thinking": True}}
    if budget is not None:
        extra["custom_logit_processor"] = processor
        extra["custom_params"] = {"thinking_budget": budget}
    completion = await client.complete(
        [{"role": "user", "content": PROMPT}],
        model=model,
        max_tokens=MAX_TOKENS,
        seed=0,
        extra_body=extra,
    )
    reasoning = completion.message.reasoning or ""
    measured_reasoning_tokens = len(
        tokenizer.encode(reasoning, add_special_tokens=False)  # type: ignore[attr-defined]
    )
    return {
        "reasoning_tokens": completion.usage.reasoning_tokens,
        "measured_reasoning_tokens": measured_reasoning_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "finish_reason": completion.finish_reason,
        "content_chars": len(completion.message.content or ""),
    }


async def _probe(
    base_url: str,
    model: str,
    processor: str,
    tokenizer: object,
) -> tuple[dict, dict, dict]:
    client = OpenAIChatClient(base_url, timeout=1800.0)
    try:
        off = await _one(client, model, processor, tokenizer, None)
        on = await _one(client, model, processor, tokenizer, BUDGET)
        tool_completion = await client.complete(
            [
                {
                    "role": "system",
                    "content": "Use submit_plan exactly once to plan a proof.",
                },
                {"role": "user", "content": "Prove that 1 + 1 = 2."},
            ],
            model=model,
            max_tokens=2048,
            seed=1,
            tools=(SUBMIT_PLAN_TOOL,),
            tool_choice={"type": "function", "function": {"name": "submit_plan"}},
            parallel_tool_calls=False,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "custom_logit_processor": processor,
                "custom_params": {"thinking_budget": 256},
            },
        )
    finally:
        await client.aclose()
    calls = tool_completion.message.tool_calls
    tool = {
        "count": len(calls),
        "name": calls[0].name if len(calls) == 1 else None,
        "arguments": calls[0].parsed_arguments() if len(calls) == 1 else None,
    }
    return off, on, tool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment.yaml")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    solver = config.models.solver
    model_dir = (
        Path(config.paths.asset_root)
        / "models"
        / solver.name.replace("/", "--")
        / solver.revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    processor = build_thinking_budget_processor(
        derive_thinking_token_profile(tokenizer)
    )
    base_url = f"http://127.0.0.1:{args.port}/v1"
    spec = ServerSpec(
        backend=config.runtime.solver_backend,
        role="solver",
        model_path=str(model_dir),
        served_model_name=solver.name,
        base_url=base_url,
        log_path=Path("probe-server.log"),
        context_length=config.budget.context_tokens,
        tensor_parallel_size=config.runtime.solver_tensor_parallel_size,
    )
    print("booting server ...", flush=True)
    with ModelServer(spec):
        off, on, tool = asyncio.run(
            _probe(base_url, solver.name, processor, tokenizer)
        )

    print(
        json.dumps(
            {"no_budget": off, f"budget_{BUDGET}": on, "forced_tool": tool},
            indent=2,
        )
    )
    clamped = (
        on["measured_reasoning_tokens"] <= BUDGET + 64
        and off["measured_reasoning_tokens"] > BUDGET + 64
    )
    tool_ok = tool["count"] == 1 and tool["name"] == "submit_plan"
    print(f"\nreasoning without budget : {off['measured_reasoning_tokens']}")
    print(f"reasoning with budget={BUDGET:<4}: {on['measured_reasoning_tokens']}")
    print(f"CLAMP WORKING: {clamped}")
    print(f"TOOL PARSER WORKING: {tool_ok}")
    return 0 if clamped and tool_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
