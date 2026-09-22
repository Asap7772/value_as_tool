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

from value_as_tool.client import OpenAIChatClient  # noqa: E402
from value_as_tool.config import load_config  # noqa: E402
from value_as_tool.orchestrator import QWEN3_THINKING_BUDGET_PROCESSOR  # noqa: E402
from value_as_tool.server import ModelServer, ServerSpec  # noqa: E402

# Deliberately open-ended so the model reasons well past the small budget below.
PROMPT = (
    "Prove or disprove: every positive integer greater than 2 can be written as "
    "the sum of two primes. Work carefully through the cases and explain your "
    "reasoning in full before giving a final answer."
)
BUDGET = 256
MAX_TOKENS = 4096


async def _one(client: OpenAIChatClient, model: str, budget: int | None) -> dict:
    extra: dict = {"chat_template_kwargs": {"enable_thinking": True}}
    if budget is not None:
        extra["custom_logit_processor"] = QWEN3_THINKING_BUDGET_PROCESSOR
        extra["custom_params"] = {"thinking_budget": budget}
    completion = await client.complete(
        [{"role": "user", "content": PROMPT}],
        model=model,
        max_tokens=MAX_TOKENS,
        seed=0,
        extra_body=extra,
    )
    return {
        "reasoning_tokens": completion.usage.reasoning_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "finish_reason": completion.finish_reason,
        "content_chars": len(completion.message.content or ""),
    }


async def _probe(base_url: str, model: str) -> tuple[dict, dict]:
    client = OpenAIChatClient(base_url, timeout=1800.0)
    try:
        off = await _one(client, model, None)
        on = await _one(client, model, BUDGET)
    finally:
        await client.aclose()
    return off, on


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
    base_url = f"http://127.0.0.1:{args.port}/v1"
    spec = ServerSpec(
        backend=config.runtime.solver_backend,
        role="solver",
        model_path=str(model_dir),
        served_model_name=solver.name,
        base_url=base_url,
        log_path=Path("probe-server.log"),
        context_length=config.budget.context_tokens,
    )
    print("booting server ...", flush=True)
    with ModelServer(spec):
        off, on = asyncio.run(_probe(base_url, solver.name))

    print(json.dumps({"no_budget": off, f"budget_{BUDGET}": on}, indent=2))
    clamped = on["reasoning_tokens"] <= BUDGET + 64
    print(f"\nreasoning without budget : {off['reasoning_tokens']}")
    print(f"reasoning with budget={BUDGET:<4}: {on['reasoning_tokens']}")
    print(f"CLAMP WORKING: {clamped}")
    return 0 if clamped else 1


if __name__ == "__main__":
    raise SystemExit(main())
