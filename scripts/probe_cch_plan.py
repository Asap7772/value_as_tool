"""Run one real CCH planner call against a local prepared solver server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from value_as_tool.client import OpenAIChatClient  # noqa: E402
from value_as_tool.harnesses.cch_plan_work_review import (  # noqa: E402
    PLAN_CAP,
    PLANNER_SYSTEM_PROMPT,
    SUBMIT_PLAN_TOOL,
    _parse_plan,
)
from value_as_tool.pipeline import load_context, schedule_path  # noqa: E402
from value_as_tool.schedule import load_schedule  # noqa: E402
from value_as_tool.server import ModelServer, ServerSpec  # noqa: E402
from value_as_tool.thinking import (  # noqa: E402
    build_thinking_budget_processor,
    derive_thinking_token_profile,
)


async def _probe(
    *,
    base_url: str,
    model: str,
    problem: str,
    processor: str,
    thinking_budget: int,
) -> dict[str, object]:
    client = OpenAIChatClient(base_url, timeout=1800.0)
    try:
        completion = await client.complete(
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            model=model,
            max_tokens=PLAN_CAP,
            seed=0,
            tools=(SUBMIT_PLAN_TOOL,),
            tool_choice={"type": "function", "function": {"name": "submit_plan"}},
            parallel_tool_calls=False,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "custom_logit_processor": processor,
                "custom_params": {"thinking_budget": thinking_budget},
            },
        )
    finally:
        await client.aclose()
    parsed = _parse_plan(completion)
    tool_calls = completion.message.tool_calls
    raw_arguments = (
        tool_calls[0].parsed_arguments() if len(tool_calls) == 1 else {}
    )
    return {
        "finish_reason": completion.finish_reason,
        "completion_tokens": completion.usage.completion_tokens,
        "reasoning_tokens": completion.usage.reasoning_tokens,
        "tool_call_count": len(tool_calls),
        "argument_types": {
            key: type(value).__name__ for key, value in raw_arguments.items()
        },
        "parsed_plan_chars": len(parsed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment_qwen38_27b_harness.yaml")
    parser.add_argument("--problem-id", default="PB-Basic-012")
    parser.add_argument("--port", type=int, default=19081)
    args = parser.parse_args()

    context = load_context(Path(args.config))
    schedule = load_schedule(schedule_path(context))
    item = next(
        item
        for item in schedule
        if item.problem_id == args.problem_id
        and item.harness_id == "cch_plan_work_review"
    )
    solver = context.config.models.solver
    manifest = json.loads(
        (
            context.artifact_root / "prepared" / "models" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    model_dir = Path(manifest["models"]["solver"]["path"])
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
        backend=context.config.runtime.solver_backend,
        role="solver",
        model_path=str(model_dir),
        served_model_name=solver.name,
        base_url=base_url,
        log_path=Path("probe-cch-plan-server.log"),
        context_length=context.config.budget.context_tokens,
        tensor_parallel_size=context.config.runtime.solver_tensor_parallel_size,
    )
    thinking_budget = PLAN_CAP - context.config.sampling.thinking_content_reserve_tokens
    print(f"booting server for {item.problem_id} ...", flush=True)
    with ModelServer(spec):
        result = asyncio.run(
            _probe(
                base_url=base_url,
                model=solver.name,
                problem=item.problem,
                processor=processor,
                thinking_budget=thinking_budget,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
